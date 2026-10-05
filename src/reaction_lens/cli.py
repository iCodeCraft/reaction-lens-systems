"""Command-line entrypoints; model downloads are always explicit."""
import argparse
from pathlib import Path

from .artifacts import download_bundle, download_encoder
from .output import as_csv, as_json


def main():
    parser = argparse.ArgumentParser(prog='reaction-lens')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('predict', 'serve'):
        p = sub.add_parser(command)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--encoder-dir', type=Path, required=True)
        p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
        p.add_argument('--threads', type=int, default=2)
        if command == 'predict':
            p.add_argument('--title', default='')
            p.add_argument('--abstract-file', type=Path, required=True)
            p.add_argument('--threshold', type=float)
            p.add_argument('--format', choices=['json', 'csv'], default='json')
            p.add_argument('--out', type=Path, required=True)
        else:
            p.add_argument('--host', default='127.0.0.1')
            p.add_argument('--port', type=int, default=8766)
    p = sub.add_parser('download-encoder')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('download-bundle')
    p.add_argument('--repo', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'download-encoder':
        download_encoder(args.bundle, args.out)
        return
    if args.command == 'download-bundle':
        download_bundle(args.repo, args.revision, args.out)
        return
    import torch
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'serve':
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(bundle=args.bundle, encoder_dir=args.encoder_dir, device=args.device),
                    host=args.host, port=args.port, workers=1)
    else:
        from .pipeline import ReactionLens
        if args.out.exists():
            parser.error('Output exists; choose a new filename')
        model = ReactionLens.from_bundle(args.bundle, args.encoder_dir, args.device)
        result = model.predict(args.title, args.abstract_file.read_text(), args.threshold)
        with args.out.open('x') as stream:
            stream.write(as_csv(result) if args.format == 'csv' else as_json(result))
        print(f"Scored {result['options_scored']} reactions; saved {args.out}")


if __name__ == '__main__':
    main()
