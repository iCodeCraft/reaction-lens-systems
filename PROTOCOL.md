# Systems audit protocol

This study evaluates a fixed single-process retrieval implementation. The
measurement parameters below were recorded before the runs; this document
has been edited for presentation without changing those parameters.

## Questions

1. Under fixed features and weights, does changing candidate order or membership
   preserve retained candidates' logits and fixed-threshold decisions?
2. Does blocking exact full-catalog scoring change its results, latency or memory?
3. What fraction of local request time is spent encoding text versus scoring?
4. Can an isolated installation reproduce the released checkpoint's cached-feature
   development metrics?

## Measurements

- Frozen structured seed23 epoch105 bundle, all 15,692 reactions. The reserved
  test is excluded, and no model is trained. Artifact hashes are verified before inference.
- CPU FP32, two PyTorch threads. Sample 64 development publications without
  replacement, NumPy seed 20261005. Five warmups, three recorded repetitions.
- Blocks: 128, 1,024 and entire catalog. Check permutations, half-catalog subsets,
  appended duplicate vectors, and unblocked reference. Absolute logit tolerance
  1e-5; retain every measured error and threshold flip, including failures.
- Candidate additions use duplicate existing vectors, not newly encoded biological
  reactions. The experiment tests candidate-axis execution, not biological validity
  of future catalog additions. Relative rank and top-k membership can change.
- Time only cached-feature scoring for these runs; exclude projection setup, input
  encoding and serialization. Sequential block order is a potential thermal bias.
- Measure memory in a fresh process for each block size using the same padded
  batch (first 16 sampled rows). Report whole-process high-water RSS, including
  dependencies and loaded arrays. Do not use processes that ran the unblocked
  reference to infer memory savings. No claims about GPU memory or service capacity.
- End-to-end local Python requests: existing released English comparison examples,
  one warmup, three repetitions, full returned payload plus JSON serialization.
  Report setup separately. This convenience sample is not a population latency
  benchmark; no HTTP/network overhead or concurrent-load claim.
- Recompute MAP, Hit@1 and Recall@100 from the released cached development features.
  This does not retrain the encoder/head or reconstruct features from source text.

## Interpretation

Candidate-independent scores are a property of the pointwise equation, not a novel
discovery. Empirical checks establish finite-precision behavior on these inputs.
The reference is exact unblocked scoring. It is not an approximate-search
competitor. All probes and raw timings are retained.
