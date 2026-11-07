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

---

# Revision round (2.0.0 small safety/correctness revision)

Same sandbox (`zexkora-growth-sandbox`, `/workspace`). Unlike the first round,
the RED below is **behavioural**: the new tests were run against the unfixed
source and failed on real assertions. The earlier "5 collection errors" run is
not offered as TDD proof any more.

## 6. RED (this revision) — before the fixes

| run | command | real output |
|---|---|---|
| python | `python3 -m pytest -m "not ui"` | **11 failed, 114 passed, 19 deselected** |
| node | `node tests/node/state.test.cjs` | exit 1: `FAIL: defaults are applied when the query is empty` / `Expected 0 !== null` (an omitted overhead was a fake 0, not unknown) |
| UI (old suite) | `python3 -m pytest -m ui` on the committed `tests/test_ui.py` | **7 ERROR** at fixture setup, each `Error: Browser.new_context: Unknown permission: clipboard-read` (all Firefox parameterisations) |

Failing python tests (verbatim summary): 6 × `tests/test_bench.py` (probe still
active), `test_ledger.py::test_runtime_overhead_is_unknown_not_invented`
(`KeyError: 'unknown'`), `test_ledger.py::test_unknown_overhead_is_not_confused_with_an_explicit_zero_bound`,
`test_ledger.py::test_manual_overhead_is_labelled_and_added`, 
`test_legacy_regression.py::test_final_api_passes_the_same_requirement`,
`test_web_static.py::test_node_state_tests_pass`.

### Real pre-change behavioural harness

`tests/test_legacy_regression.py` loads the frozen public
`calculate_vram` from git `2dcb9ff` (`tests/fixtures/legacy_vram_calc_2dcb9ff.py`,
test fixture only, never imported by the package) and runs the same release
requirement the ledger must meet:

* KV bytes for Qwen2.5-7B @ 8192 tokens must be `57344 * 8192 = 469762048`;
* runtime overhead must be an explicit `unknown_not_estimated`.

Legacy output: KV ≈ 1.09 GiB (it guesses `ceil(14 + 7*log2(7.6)) = 35` layers and
8 KV heads instead of the real 28 × 4), and `cuda_overhead_gb = 0.65` +
`activations_gb = 0.08` are invented numbers with no unknown representation. The
shared requirement check raises a genuine `AssertionError` on that output; then
the same check passes on `compute_ledger`. That is the behavioural RED/GREEN,
not a collection error.

## 7. GREEN (this revision)

| run | result |
|---|---|
| `python3 -m pytest -m "not ui"` | **125 passed, 19 deselected in 7.80s** |
| `node tests/node/state.test.cjs` | **OK (19 tests)** |
| `python3 -m pytest -m ui tests/test_ui.py` | **19 passed in 75.75s** — 8 page tests × {chromium, firefox} + 3 screenshot tests, **zero skips** |
| `python3 scripts/build_web.py --check` | `OK: index.html is up to date (47869 bytes)` |
| `python3 scripts/fetch_model_metadata.py --check` | `OK: ... matches upstream metadata for 2 models` |
| `python -m build` + clean-venv wheel smoke | wheel installs; `calc --json` → weights 4683073632 B, total 4.799 GiB, `runtime_overhead.bytes = null` + `unknown = true`; `bench --json` exits **2** with `benchmark_disabled` |

### Firefox clipboard evidence

Firefox has no `clipboard-read` permission; passing it was the original crash.
The test fixture now launches Firefox with Firefox's own async-clipboard test
preference (`dom.events.testing.asyncClipboard = true`) and chromium with its
clipboard permissions. `test_copy_report_puts_exact_honest_text_on_the_clipboard`
clicks the real button and asserts the clipboard content is **byte-for-byte** the
expected report (read back through the real `navigator.clipboard`, with a DOM
paste fallback). No production code is stubbed.

### Firefox screenshots (real browser output, inspected with vision)

* `validation/screenshots/firefox-360-mobile.png` (360 px, no horizontal overflow)
* `validation/screenshots/firefox-1280-desktop.png`
* `validation/screenshots/firefox-1280-invalid-url-notice.png`

The 360 px and desktop renders show `Runtime overhead — unknown · not estimated`
(em dash, no numeric `0.000 GiB`), unchanged original styling and readable ledger
rows.

### Benchmark is gone, not merely relabelled

`tests/test_bench.py` proves the absence of a probe: a live local HTTP server
sees **0 requests**, the CLI exits 2 with `{"error": "benchmark_disabled"}`, and
`vramcalc/bench.py` carries no `urllib`/`socket`/`http` transport at all. No
tokens/s figure is claimed in README, docs, CLI or the manifest.
