# Inference Throughput Benchmark Summary

| engine | model | batch_size | tokens/sec | p99ms | memory_MB |
|---|---|---:|---:|---:|---:|
| ollama | llama3.2:1b | 1 | 67.34 | 11428.61 | -320.64 |
| ollama | llama3.2:1b | 1 | 66.22 | 3114.52 | -2.25 |
| ollama | llama3.2:1b | 4 | 67.16 | 11085.89 | 0.33 |
| ollama | llama3.2:1b | 4 | 69.70 | 3020.51 | -0.31 |
| hf | sshleifer/tiny-gpt2 | 1 | 3963.11 | 29.08 | 0.23 |
| hf | sshleifer/tiny-gpt2 | 1 | 9126.98 | 32.66 | 0.00 |
| hf | sshleifer/tiny-gpt2 | 4 | 13874.49 | 33.19 | 0.02 |
| hf | sshleifer/tiny-gpt2 | 4 | 29592.49 | 41.75 | 0.02 |
