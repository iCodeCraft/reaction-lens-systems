<div align="center">

# Reaction Lens Systems

**Verifiable full-catalog retrieval from scientific text**

![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.12-blue)
![Code license](https://img.shields.io/badge/code-MIT-green)
![Paper license](https://img.shields.io/badge/paper-CC_BY_4.0-green)

[Paper](paper/reaction-lens-systems.pdf) · [Protocol](PROTOCOL.md) · [Numerical evidence](NUMBERS.csv)

</div>

Reaction Lens Systems ranks Reactome reactions from English scientific titles
and abstracts. This experience report evaluates a fixed implementation: whether
candidate transformations and scoring blocks preserve pairwise scores, and how
encoding and scoring contribute to local inference cost. Predictions are
proposals for human review, not verified biological conclusions.

## Verify the paper's numbers

From the repository root, with Python 3.11 or 3.12:

```sh
make verify
```

No model download, account or third-party Python package is required.
Validation also rejects missing/duplicate probes, incomplete catalog predictions,
changed artifact identities and relaxed acceptance tolerances. This checks
recorded observations, summary statistics, generated outputs and the hashes of
the measured implementation. Run `make all` to regenerate tables and numeric
macros from the committed observations. These commands do not rerun inference.

## Install and run

Install [uv](https://docs.astral.sh/uv/), then:

```sh
make setup
make serve
```

Open **http://127.0.0.1:8767/**. Try:
`STAT1 induces SMAD7 expression and inhibits TGF-beta signaling.`
The interface shows ranked reactions and their scores. API documentation is at
http://127.0.0.1:8767/docs. Stop with Ctrl+C; use `PORT=8768 make serve` for another
port. Subsequent runs reuse verified local model files.

The first run downloads the inference bundle and pinned upstream encoder files.
The [public model bundle](https://huggingface.co/imrangadzhiev/reaction-lens-systems)
and the pinned upstream encoder files can be downloaded without an HF account.
Anonymous downloads from an empty cache have been verified against their hashes.

The model repository, immutable revision and nine required file hashes are in
[artifact-manifest.json](artifact-manifest.json). Downloads select only those
files; unrelated model-repository contents are not needed. The bundle manifest
pins the upstream encoder revisions and hashes. Local models and environments
are excluded from version control.

## Reproduce inference and measurements

After `make setup`:

```sh
make test
make check-model
make benchmark
```

`make check-model` recomputes scores for all 2,377 development publications against
15,692 reactions using the released cached features and compares MAP, Hit@1 and
Recall@100 with recorded values. It does not retrain the model or reconstruct the
full feature cache from publication text.

`make benchmark` reruns candidate-transformation checks, scoring timings,
separate-process memory probes and complete-text local predictions. New results
are written to `.local/runs/system-benchmark.json`, preserving published
observations. Timing varies with hardware and load. The recorded experiment uses
CPU FP32 and two PyTorch threads; see [PROTOCOL.md](PROTOCOL.md) for sampling,
warmups and measurement boundaries. The local artifact files occupy about
583 MB; allow additional space for dependencies and download caches.

## Evidence map

| Paper result | Committed evidence | Regeneration |
|---|---|---|
| Table 1: blocking latency and memory | [Scoring table](results/table1_scoring.csv) | `make table1` |
| Section 5.1: candidate contracts | [All contract probes](results/table3_contracts.csv) | `make contracts` |
| Section 5.2: complete predictions | [Component timings](results/table2_end_to_end.csv) | `make timings` |
| Section 5.3: cached-feature ranking | [Fresh evaluation](results/fresh-evaluation.json) | `make check-model` |

The CSV regeneration targets share one deterministic renderer. Raw benchmark
observations are in [system-benchmark.json](results/system-benchmark.json).
[NUMBERS.csv](NUMBERS.csv) maps displayed values to source fields and
transformations; [measured-source-hashes.json](measured-source-hashes.json)
identifies the measured scorer, encoder, pipeline and benchmark script.

Candidate-independent scores do not imply stable ranks or calibrated biological
probabilities. The latency inputs form a small convenience sample, and the memory
comparison has one trial per configuration. The reference uses exact scoring;
it is not a competing retrieval baseline. See the paper for limitations.

## Build the paper

With [Tectonic](https://tectonic-typesetting.github.io/) installed and its TeX
bundle available:

```sh
make paper
```

This builds the English PDF. An auxiliary source archive is saved locally under
`.local/paper/` and is excluded from version control. The LaTeX sources remain
in `paper/` so the PDF can be rebuilt. Tectonic and its TeX
bundle are separate build dependencies; the first build may need network access.
The PDF evidence links are pinned to source commit
`2140cc40ba3dc2af9ad0600cd2cc1624a47a2bf8` in this repository.
`scripts/configure_release.py --repo URL --revision SHA` can bind a later revision
only after verifying that its evidence files match the working copy. Rebuild with
`make paper` after binding. This checks file identity, not remote availability;
readers need public access to the referenced repository and commit. Source code is licensed under
[MIT](LICENSE), manuscript text under [CC BY 4.0](paper/LICENSE). Model and data
terms are documented separately in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
