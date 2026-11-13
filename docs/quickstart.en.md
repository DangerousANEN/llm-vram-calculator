# Quickstart (English)

`llm-vram-calculator` is a byte-level memory ledger for **dense** transformer
inference. It answers one question honestly: how many bytes do the weights and
the KV cache of a specific, revision-pinned model take at a given context and
batch size?

It does **not** measure your GPU and it does not promise that a model fits.

## Install

Requirements: Python 3.9+, no runtime dependencies.

```bash
git clone https://github.com/DangerousANEN/llm-vram-calculator.git
cd llm-vram-calculator

# run from the checkout
python -m vram_calc models

# or install the package (entry points: vram-calc, llm-vram-calc)
pip install -e .
vram-calc models
```

## Use

```bash
# list the pinned catalogue
python -m vram_calc models

# Qwen2.5 7B Instruct, official Q4_K_M GGUF, 8192 tokens, one sequence
python -m vram_calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192

# Qwen3 8B in bf16, 32k context, batch 4, plus a manual 2 GiB overhead bound
python -m vram_calc calc --model qwen3-8b --quant bf16 --ctx 32768 --batch 4 --overhead-gib 2

# machine readable
python -m vram_calc calc --model qwen3-8b --quant q4_k_m --ctx 8192 --json
```

Sample output (trimmed):

```
 Memory ledger: Qwen2.5 7B Instruct  [q4_k_m]
 source            : Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28
 geometry          : 28 layers x 4 KV heads x 128 head_dim
 weights           :        4.362 GiB  (4683073632 bytes)
 KV cache          :        0.438 GiB  (469762048 bytes, 57344 B/token/sequence)
 runtime overhead  :      unknown  (unknown_not_estimated)
 LEDGER TOTAL      :        4.799 GiB  (arithmetic sum, NOT measured VRAM)
```

## The formula

```
kv_bytes_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * bytes_per_element
kv_bytes           = kv_bytes_per_token * context * batch
ledger_total       = weights_bytes + kv_bytes + manual_overhead_bytes
```

* `2` = one key tensor and one value tensor.
* `head_dim` comes from `config.head_dim` when present, otherwise
  `hidden_size / num_attention_heads`; the source is printed with the result.
* `bytes_per_element` is 2 for `fp16` and `bf16`; nothing else is modelled.
* Units are GiB (1 GiB = 1024^3 bytes) plus the exact byte count.

## Supported models

| id | repo | config revision | GGUF revision |
|---|---|---|---|
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` |
| `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `7c41481f57cb95916b40956ab2f0b139b296d974` |
| `qwen3-14b` | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | `530227a7d994db8eca5ab5ced2fb692b614357fd` |
| `qwen3-4b` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `bc640142c66e1fdd12af0bd68f40445458f3869b` |

Weights come from the official file listings at those revisions: the exact byte
sum of the safetensors shards (`bf16`/`fp16`) or of the official GGUF shards
(`q8_0`, `q6_k`, `q5_k_m`, `q5_0`, `q4_k_m`, …). If a requested quantisation has
no official listing, the CLI refuses and offers `--allow-estimate`: a labelled
bytes-per-parameter estimate whose parameter count is derived from the config
(attention Q/K/V including the output projection, the three MLP matrices,
embeddings) and is tested against the official `safetensors.total` with a 0.5%
tolerance. Regenerate or verify the
metadata (metadata only, no weights are downloaded):

```bash
python scripts/fetch_model_metadata.py           # regenerate vramcalc/data/models.json
python scripts/fetch_model_metadata.py --check    # verify the commit still matches upstream
```

## What is refused, and why

A refusal with an error code is the correct answer when the dense KV formula
would be wrong:

| input | error code |
|---|---|
| MoE config (`num_experts`, `n_routed_experts`, …) | `moe_not_supported` |
| MLA config (`kv_lora_rank`, `qk_nope_head_dim`, …) | `mla_not_supported` |
| active sliding window | `sliding_window_not_supported` |
| per-layer hybrid attention (`layer_types`) | `hybrid_attention_not_supported` |
| unknown `model_type` / `architectures` | `unknown_architecture` |
| quantised KV cache (`--kv-quant fp8`) | `unsupported_kv_dtype` |
| CPU offload (`--cpu-offload-gib`) | `offload_unsupported` |
| legacy parameter-count mode (`--params`) | `legacy_params_removed` |
| context above `max_position_embeddings` | `context_exceeds_model_max` |
| context × batch above 33 554 432 | `positions_too_large` |

Unsupported inputs exit with code `2` and print the reason with a migration hint
on stderr (or a JSON object with `error` when `--json` is used).

## Runtime overhead is unknown

CUDA context, allocator fragmentation, activations, workspaces, CUDA graphs and
paged-attention padding are runtime dependent. This tool reports them as
`unknown_not_estimated`, with `bytes`/`gib` set to `null` and `"unknown": true` -
an explicit unknown, never a measured zero. If you measured a value on your own
stack, add it with `--overhead-gib`; the result labels it as `manual_bound`
(an explicit `0` stays an explicit zero bound and is labelled as such).

`verification.gpu_runtime_checked` is always `false` in this release: no GPU run
was performed. Treat the ledger as an accounting model, not a measurement.

## Benchmark - temporarily disabled

The 2.0 release is a memory calculator, not a speed test. The old streaming
probe was removed after a release-safety review: it streamed from an arbitrary
user-supplied URL with `urllib`, followed redirects while forwarding the
`Authorization` header, reported tokens/s as `(N-1)/(last chunk - first chunk)`,
treated truncated or malformed SSE as success, and only checked its overall
deadline between chunks.

This release therefore ships **no network benchmark code at all**:

```bash
$ python -m vram_calc bench --url http://127.0.0.1:8000/v1 --model local-model
error [benchmark_disabled]: ... exit code 2, with a migration command
```

`run_bench` refuses immediately, a live endpoint observes zero requests, and no
tokens/s figure is claimed anywhere. A correct bounded local probe (pinned
endpoint, no redirect following, explicit stream terminator, wall-clock
deadline) requires a redesign and is not part of this release.

## Web UI

`index.html` is generated from `web/` + `vramcalc/data/models.json` and is fully
self-contained (no fetch, no external scripts, works from `file://`).

```bash
python scripts/build_web.py           # rebuild index.html
python scripts/build_web.py --check   # CI guard: committed page must match
```

Shares: parameters live in the URL (`?m=&q=&c=&b=&o=&k=`) and are validated on
load; the page offers "copy report", "copy link" and an official
`https://t.me/share/url` chooser. Telegram is optional - the calculator never
requires an account or a subscription, and it sends nothing automatically.

## Tests

```bash
python -m pytest -m "not ui"
node tests/node/state.test.cjs
python -m pytest -m ui tests/test_ui.py     # needs playwright + chromium AND firefox
```

Both UI browsers are required: if chromium or firefox cannot be launched the UI
module fails instead of skipping, so a whole browser cannot silently drop out of
verification. `UI_ALLOW_MISSING_BROWSERS=1` is a local-only opt-out; CI never
sets it.

Package build and smoke test:

```bash
python -m build
python -m venv /tmp/venv && /tmp/venv/bin/pip install dist/*.whl
/tmp/venv/bin/vram-calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192 --json
```

## License

MIT. Model metadata belongs to its authors (Qwen, apache-2.0).