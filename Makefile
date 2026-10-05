PYTHON ?= python3
RUNTIME ?= .venv/bin/python
TECTONIC ?= tectonic

.PHONY: all table1 contracts timings verify setup test artifacts encoder check-model benchmark serve paper
all table1 contracts timings:
	$(PYTHON) scripts/render_results.py
verify:
	$(PYTHON) scripts/render_results.py --check
	$(PYTHON) scripts/verify_study.py
setup:
	uv sync --locked --extra encoder --extra serve --extra dev
test:
	PYTHONPATH=src $(RUNTIME) -m unittest discover -s tests -v
artifacts:
	$(RUNTIME) scripts/download_artifacts.py
encoder:
	$(RUNTIME) scripts/download_artifacts.py --encoder
check-model: artifacts
	$(RUNTIME) scripts/check_release.py --bundle artifacts/structured-seed23
benchmark: encoder
	mkdir -p .local/runs
	$(RUNTIME) scripts/benchmark_system.py --bundle artifacts/structured-seed23 --encoder artifacts/encoder --examples examples/latency-examples.json --output .local/runs/system-benchmark.json
serve: encoder
	./start.sh
paper: all verify
	cd paper && $(TECTONIC) --untrusted --keep-logs --keep-intermediates reaction-lens-systems.tex
	$(PYTHON) scripts/package_paper.py
