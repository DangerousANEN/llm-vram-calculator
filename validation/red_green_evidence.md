# RED → GREEN evidence (P0-03/04/05/06)

All application tests were executed **inside the isolated sandbox container**
`zexkora-growth-sandbox` (image `zexkora-test-sandbox:20261107`, python 3.13,
pytest 9.2, node 20, chromium) — never on the host and never on the user's PC.
The repo is bind-mounted at `/workspace`.

## 1. RED: the new test suite against pre-change source

The tests were written before the implementation existed. First run, against the
pre-change source (`git archive HEAD` extracted to `/tmp/red` inside the
container, new `tests/` copied in), with the original `vram_calc.py` unchanged:

```
$ python3 -m pytest -q -m "not ui"          # in /tmp/red
ERROR tests/test_architecture_gate.py
ERROR tests/test_bench.py
ERROR tests/test_catalog_fixtures.py
ERROR tests/test_ledger.py
ERROR tests/test_ui.py
!!!!!!!!!!!!!!!!!!! Interrupted: 5 errors during collection !!!!!!!!!!!!!!!!!!!!
1 warning, 5 errors in 1.98s
E   ModuleNotFoundError: No module named 'vramcalc'
```

Behavioural red evidence for the same defect class (why those tests exist), taken
from the pre-change CLI:

```
$ python3 vram_calc.py calc --params 7 --quant q4_k_m --ctx 8192
 🧮 VRAM ALLOCATION REPORT: 7.0B (Q4_K_M)
 • Веса модели (Weights):         3.78 GB
 • KV-Кэш (8192 токенов, b=1):     1.06 GB (kv=fp16)
 • CUDA Overhead + Активации:    0.81 GB
 • ИТОГО Потребление:            5.65 GB
 • Рекомендуемый размер GPU:         7 GB VRAM
```

* layer/KV-head counts were derived from a parameter count, not from any config;
* "CUDA Overhead + Активации: 0.81 GB" was an invented constant presented as a
  precise figure;
* the report recommended a GPU size, i.e. implied a fit verdict;
* decimal `GB` was used while the arithmetic was binary.

Bench (pre-change source, `vram_calc.py` lines ~113-140): the code collected only
`delta["content"]` strings into `token_times`, never read `usage`, and computed
`tps = (total_tokens - 1) / (token_times[-1] - token_times[0])` where
`total_tokens = len(token_times)`. Stream chunks were reported as tokens/s.

## 2. GREEN: same tests against the implementation

```
### python3 -m pytest -m "not ui"
130 passed, 16 deselected in 14.58s

### node tests/node/state.test.cjs
state.test.cjs: OK (12 tests)

### python3 -m pytest -m ui tests/test_ui.py
9 passed, 7 skipped in 37.95s

### python3 scripts/build_web.py --check
OK: index.html is up to date (40914 bytes)

### python3 scripts/fetch_model_metadata.py --check
OK: /workspace/vramcalc/data/models.json matches upstream metadata for 2 models
```

Full log: `validation/sandbox_test_run.log`.

`9 passed` = the UI suite in chromium; the 7 skips are the firefox
parametrisations — the sandbox ships `/usr/bin/firefox` (a Debian build without
Playwright's Juggler protocol), so Playwright cannot drive it:
`BrowserType.launch: Failed to launch the browser process ... [pid] <process did
exit: exitCode=0>`. No UI claim in this phase rests on Firefox; CI installs the
Playwright firefox build (`python -m playwright install firefox`) and runs the
same suite there.

## 3. Package build and install smoke (inside the container)

```
$ /tmp/bvenv/bin/python -m build --outdir /tmp/dist
Successfully built llm_vram_calculator-2.0.0.tar.gz and llm_vram_calculator-2.0.0-py3-none-any.whl

$ /tmp/venv/bin/pip install /tmp/dist/llm_vram_calculator-2.0.0-py3-none-any.whl
$ /tmp/venv/bin/vram-calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192 --json
weights 4683073632
kv 469762048
total 5152835680 4.799 GiB
measured False gpu_checked False

$ /tmp/venv/bin/vram-calc calc --params 7 --ctx 8192 ; echo "exit=$?"
error [legacy_params_removed]: ... Use --model qwen2.5-7b-instruct ...
exit=2
```

Wheel contents include `vramcalc/data/models.json` (catalogue ships with the
package; no network at runtime).

## 4. Data provenance

Fixtures were generated from live Hugging Face metadata
(`scripts/fetch_model_metadata.py`), then independently re-verified inside the
sandbox with `--check` against the same upstream API:

| model | config revision | config sha256 | GGUF revision |
|---|---|---|---|
| qwen2.5-7b-instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | `7463bb0e…` | `bb5d59e0…` |
| qwen3-8b | `b968826d9c46dd6066d109eabc6255188de91218` | `f7c4eadf…` | `7c41481f…` |

Only `config.json` and public file listings (byte sizes) were downloaded. No
model weights were downloaded at any point.

## 5. Screenshots (real browser output)

Written by headless chromium during the UI run, inspected visually afterwards:

* `validation/screenshots/desktop-1280-default.png`
* `validation/screenshots/desktop-1280-qwen3-32k-batch2.png`
* `validation/screenshots/desktop-1280-invalid-url-notice.png`
* `validation/screenshots/mobile-360.png` (360 px viewport, no horizontal overflow)

Screenshots are inspection evidence; the assertions that matter (byte-for-byte
ledger values, URL round-trip, clipboard text, notice text, scroll width) are
DOM assertions in `tests/test_ui.py`.
