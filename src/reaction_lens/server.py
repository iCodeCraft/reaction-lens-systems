"""Local FastAPI application. One process, one frozen model, explicit readiness."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .output import as_csv
from .pipeline import BusyError, ReactionLens


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    title: str = Field(default='', max_length=2000)
    abstract: str = Field(default='', max_length=50000)
    threshold: float | None = None

    @model_validator(mode='after')
    def nonempty(self):
        if not (self.title.strip() or self.abstract.strip()):
            raise ValueError('Provide a title or abstract')
        return self


def create_app(pipeline: ReactionLens | None = None, *, bundle: Path | None = None,
               encoder_dir: Path | None = None, device: str = 'cpu') -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = pipeline
        if pipeline is None and bundle is not None and encoder_dir is not None:
            app.state.pipeline = await run_in_threadpool(ReactionLens.from_bundle, bundle, encoder_dir, device)
        yield
        app.state.pipeline = None

    app = FastAPI(title='Reaction Lens Systems', version='0.1.0.dev0', lifespan=lifespan,
                  description='Full-catalog reaction scoring for human review. Scores are not biological probabilities.')
    static = Path(__file__).parent / 'static'
    app.mount('/static', StaticFiles(directory=static), name='static')

    def current():
        if app.state.pipeline is None:
            raise HTTPException(503, 'Model not configured. Start with --bundle and --encoder-dir.')
        return app.state.pipeline

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(static / 'index.html')

    @app.get('/healthz')
    def health():
        return {'status': 'alive'}

    @app.get('/readyz')
    def ready():
        current()
        return {'status': 'ready'}

    @app.get('/api/model')
    def model():
        return current().metadata()

    @app.post('/api/predict')
    def predict(request: PredictRequest, format: Literal['json', 'csv'] = 'json'):
        try:
            result = current().predict(request.title, request.abstract, request.threshold)
        except BusyError as error:
            raise HTTPException(503, str(error), headers={'Retry-After': '2'}) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if format == 'csv':
            return Response(as_csv(result), media_type='text/csv',
                            headers={'Content-Disposition': 'attachment; filename="reaction-lens.csv"'})
        return result

    return app


def app_factory():
    bundle = os.environ.get('REACTION_LENS_BUNDLE')
    encoder = os.environ.get('REACTION_LENS_ENCODER_DIR')
    return create_app(bundle=Path(bundle) if bundle else None,
                      encoder_dir=Path(encoder) if encoder else None,
                      device=os.environ.get('REACTION_LENS_DEVICE', 'cpu'))
