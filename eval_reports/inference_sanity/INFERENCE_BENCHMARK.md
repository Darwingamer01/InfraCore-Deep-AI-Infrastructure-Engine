# Inference Sanity Findings

This note captures the narrow Ollama sanity matrix run against `llama3.2:1b` with 10 runs per config.

## Findings

- Throughput is flat across the tested batch sizes: `66.2` to `69.7` tokens/sec.
- Ollama CPU mode is effectively serialized here, so batch size is not a meaningful throughput lever.
- p99 latency is dominated by the generated completion behavior, not the batch size itself.
- The shorter prompt variant (`50` tokens) produced the highest p99 values, which is consistent with longer completions being generated from sparse prompts.
- The first-run memory delta of `-320.64 MB` is a cold-start artifact, not real memory reclamation.
- Subsequent memory deltas are essentially flat (`-2.25 MB` to `+0.33 MB`), which supports steady-state stability.
- TTFT mean stayed stable across all configs at about `99` to `108 ms`, indicating that warm-up and steady-state timing are working.

## Interpretation

- Treat the first run as warm-up and discard its memory delta if you are comparing steady-state configurations.
- For Ollama CPU benchmarks, use batch size as a control variable for experiment shape, not as an expected throughput optimization.
- When interpreting p99, focus on output behavior and queueing effects rather than prompt length alone.

## Notes

- The runbook lives in the top-level `INFERENCE_BENCHMARK.md` file.
- Benchmark artifacts for this run are stored in `eval_reports/inference_sanity/`.