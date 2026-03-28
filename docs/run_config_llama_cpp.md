# Run Config – llama.cpp (local)

## Model
- GGUF: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
- Path: models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

## Experiment
- Questions: data_processed/questions_200.jsonl (200 items, seed=42, NQ-Open train)
- Conditions: experiment_conditions.csv (10 conditions)

## Decoding (fixed)
- temperature: {0.0, 0.2, 0.5, 0.8, 1.0}
- top_p: 1.0
- top_k: 0
- max_tokens: 256
- seed: 42

## llama.cpp runtime
- n_ctx: 4096
- n_threads: 8
- n_gpu_layers: -1 (Metal if available)

## Output format
- JSON only: {answer, abstain, confidence}
- stop: ["\n\n", "```"]
