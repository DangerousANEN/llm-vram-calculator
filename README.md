# ⚡ LLM VRAM Calculator — config-grounded ledger

[![Telegram](https://img.shields.io/badge/Telegram-@zexkora__hub-0088cc?logo=telegram&style=for-the-badge)](https://t.me/zexkora_hub)
[![Live WebApp](https://img.shields.io/badge/Live_WebApp-GitHub_Pages-00d2ff?style=for-the-badge)](https://dangerousanen.github.io/llm-vram-calculator/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Runtime%20Deps-Zero%20(Stdlib)-brightgreen?style=for-the-badge)](vramcalc)

> 🚀 Интерактивный калькулятор (GitHub Pages): [dangerousanen.github.io/llm-vram-calculator](https://dangerousanen.github.io/llm-vram-calculator/) — работает в браузере, без установки, без аналитики и без сбора идентификаторов.
> 📖 Подробный лонгрид с математикой: [Локальные LLM в 2026: рантаймы, расчёт VRAM и ловушка KV-кэша](https://telegra.ph/Lokalnye-LLM-v-2026-rantajmy-raschet-VRAM-i-lovushka-KV-kehsha-10-28)
> 📢 Инженерный канал: [**@zexkora_hub**](https://t.me/zexkora_hub) — шпаргалки по физике инференса, бенчмарки, разборы сорцов.

---

## Что это на самом деле считает

Калькулятор строит **бухгалтерию байт** для dense-модели: сколько занимают веса из официального файла и сколько занимает KV-кэш по геометрии самого конфига. Единица — **GiB (1 GiB = 1024³ байт)**, рядом всегда печатается точное число байт.

```
ledger_total   = weights_bytes + kv_cache_bytes + manual_overhead_bytes
kv_cache_bytes = 2 (key+value) × num_hidden_layers × num_key_value_heads
                 × head_dim × bytes_per_element × context × batch
```

Каждое слагаемое имеет источник:

| Слагаемое | Откуда берётся |
|---|---|
| Веса | точная сумма байт официальных файлов (safetensors-шарды или GGUF) на **закреплённой ревизии** Hugging Face |
| KV-кэш | `num_hidden_layers`, `num_key_value_heads`, `head_dim`, `max_position_embeddings` из `config.json` этой же ревизии |
| Runtime overhead | **не оценивается.** Неизвестен. Можно добавить свой измеренный запас флагом `--overhead-gib` — он будет помечен как ручная граница |

Итог **не является измеренной VRAM** и **не гарантия**, что модель влезет: CUDA-контекст, фрагментация аллокатора, активации, workspace-буферы, CUDA graphs и padding paged-attention зависят от рантайма. Это написано и в CLI, и в веб-интерфейсе.

## Поддерживаемые модели (ревизии закреплены)

| id | Репозиторий | Ревизия конфига | Ревизия GGUF | Лицензия |
|---|---|---|---|---|
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `bb5d59e06d9551d752d08b292a50eb208b07ab1f` | apache-2.0 |
| `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `7c41481f57cb95916b40956ab2f0b139b296d974` | apache-2.0 |
| `qwen3-14b` | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | `530227a7d994db8eca5ab5ced2fb692b614357fd` | apache-2.0 |
| `qwen3-4b` | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `bc640142c66e1fdd12af0bd68f40445458f3869b` | apache-2.0 |

Размеры файлов и `sha256` конфигов лежат в `vramcalc/data/models.json` и пересобираются скриптом `scripts/fetch_model_metadata.py` — он скачивает **только** `config.json` и публичный листинг файлов, никаких весов. Проверка, что цифры не разъехались с upstream:

```bash
python scripts/fetch_model_metadata.py --check
```

## Быстрый старт

```bash
git clone https://github.com/DangerousANEN/llm-vram-calculator.git
cd llm-vram-calculator
python -m vram_calc models

# Qwen2.5 7B, Q4_K_M, контекст 8192, одна последовательность
python -m vram_calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192

# Qwen3 8B, bf16, 32k, batch 4, плюс свой измеренный запас 2 GiB
python -m vram_calc calc --model qwen3-8b --quant bf16 --ctx 32768 --batch 4 --overhead-gib 2

# машинный вывод
python -m vram_calc calc --model qwen3-8b --quant q4_k_m --ctx 8192 --json
```

Установка как пакета (entry points `vram-calc` и `llm-vram-calc`):

```bash
pip install -e .     # или pip install dist/*.whl
vram-calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192
```

`python vram_calc.py ...` продолжает работать, но печатает в stderr уведомление о миграции на `python -m vram_calc`.

## Что калькулятор **отказывается** считать

Отказ — это фича. Вместо похожего числа выводится ошибка с кодом и причиной.

- **MoE** (`num_experts`, `num_local_experts`, `n_routed_experts`, …) — активные параметры не равны резидентным весам, dense-формула соврёт.
- **MLA** (DeepSeek-подобное латентное внимание, `kv_lora_rank` и т.п.) — своя геометрия сжатия, здесь не моделируется.
- **Sliding window** (`use_sliding_window: true`) — кэш перестаёт расти за окном, формула завысила бы память.
- **Гибридное внимание по слоям** (`layer_types` с `linear_attention` и т.п.) — рост кэша не одинаковый.
- **Неизвестная архитектура** — `model_type` вне `{qwen2, qwen3}`.
- **Квантованный KV-кэш** (`--kv-quant fp8`, любые q8/q4) — точного размера без знания блоков и рантайма нет; считается только `fp16`/`bf16`.
- **Частичный CPU-offload** (`--cpu-offload-gib`) — резидентный набор GPU тогда зависит от рантайма.
- **Context > `max_position_embeddings`** — YaRN/rope-экстраполяция не моделируется. У Qwen2.5-7B в конфиге 32768 (карточка модели упоминает расширение), калькулятор держится конфига.
- **Абсурдные входы** — нецелые/нечисловые/≤0 значения, `context × batch > 33 554 432`, KV > 1 TiB.

## Честные единицы и overhead

- Всё в GiB, байты печатаются точно (`4683073632 B`).
- Веса: `bf16`/`fp16` — точная сумма официальных safetensors-шардов; `q4_k_m`, `q5_k_m`, `q6_k`, `q8_0` и др. — точная сумма официальных GGUF-файлов. Если официального файла нужного кванта нет, CLI откажется и предложит `--allow-estimate`, который выдаёт **помеченную** оценку по байт-на-параметр.
- KV: только `fp16`/`bf16`, оба по 2 байта на элемент.
- Runtime overhead: `unknown_not_estimated`. В JSON это `bytes: null` / `gib: null` + `"unknown": true` — это именно «неизвестно», а не измеренный ноль. Ручной запас добавляется как `manual_bound` (явный `0` остаётся явным нулём и так и подписан).

## CLI

```bash
python -m vram_calc models                  # модели, ревизии, точные кванты
python -m vram_calc calc --help
python -m vram_calc calc --model ... --quant ... --ctx ... [--batch N] [--kv-dtype fp16|bf16] [--overhead-gib X] [--json]
# bench ВРЕМЕННО ОТКЛЮЧЁН: сетевого пробника в этом релизе нет (код выхода 2, benchmark_disabled)
python -m vram_calc bench --help
```

### Миграция со старого `--params`

Старый режим выводил число слоёв и KV-голов из количества параметров — это не конфиг. Он удалён как «точный»:

```bash
$ python -m vram_calc calc --params 7 --quant q4_k_m --ctx 8192
error [legacy_params_removed]: ... Use --model qwen2.5-7b-instruct ...
```

Если нужна именно грубая оценка для миграции скриптов — только явным флагом и с пометкой «NOT CONFIG-GROUNDED»:

```bash
python -m vram_calc calc --params 7 --quant q4_k_m --ctx 8192 --legacy-heuristic
```

### Бенчмарк: временно отключён

Релиз 2.0 — это калькулятор памяти, а не тест скорости. Старый потоковый
пробник убран после ревью безопасности: он ходил на произвольный пользовательский
URL через `urllib`, **шёл по редиректам и вёл за собой заголовок
`Authorization`**, считал tokens/s как `(N-1)/(последний чанк − первый чанк)` и
принимал обрезанный/битый SSE за успех, а общий таймаут проверялся только между
чанками.

Поэтому в этом релизе **нет сетевого кода бенчмарка вообще**: `bench`
немедленно отказывает с кодом `benchmark_disabled` (код выхода 2) и печатает
команду миграции. Никаких запросов, никаких платных вызовов и никаких заявлений
о «рабочем tokens/s». Корректный ограниченный локальный пробник требует
отдельного редизайна и в этот релиз не входит.

## Веб-интерфейс

- Выбор модели, точного кванта, контекста (ползунок + число), batch, типа KV, ручного overhead.
- Показывает источник и ревизию, sha256 конфига, лицензию, формулу KV с подставленными числами.
- Кнопки «Скопировать отчёт» и «Скопировать ссылку», официальный `t.me/share/url` chooser (Telegram-шеринг опционален, ничего не отправляется автоматически).
- Состояние в URL (`?m=&q=&c=&b=&o=&k=`) валидируется при загрузке: неизвестная модель/квант/тип KV и запредельный контекст не ломают страницу, а показывают уведомление и подставляют допустимое значение.
- Страница самодостаточна: каталог, стили и JS встроены в `index.html`, никаких внешних скриптов и `fetch` — работает и с GitHub Pages, и из `file://`.
- Мобильный от 360 px и десктоп, без горизонтальной прокрутки, управление доступно с клавиатуры.

Пересобрать страницу после изменения каталога:

```bash
python scripts/build_web.py          # перегенерировать index.html
python scripts/build_web.py --check  # проверить, что коммит актуален (делает CI)
```

## Тесты

```bash
python -m pytest -m "not ui"              # формулы, отказы, каталог, CLI, bench-отказ, статика
node tests/node/state.test.cjs            # валидация URL-state + сверка чисел JS ↔ Python
python -m pytest -m ui tests/test_ui.py   # headless chromium + firefox (оба обязательны), скриншоты в validation/screenshots
```

UI-тесты требуют **оба** браузера: если chromium или firefox не запускается,
модуль падает (`pytest.fail`), а не тихо пропускается — иначе целый браузер
может выпасть из проверки незаметно. Локальное послабление возможно только
явным `UI_ALLOW_MISSING_BROWSERS=1`; CI его не выставляет.

Покрыто: точные байты весов против официальных листингов, `kv_bytes_per_token` (57344 B для Qwen2.5-7B, 147456 B для Qwen3-8B и Qwen3-4B, 163840 B для Qwen3-14B), отказы по MoE/MLA/sliding window/гибриду/неизвестной архитектуре, границы контекста и batch, помеченные оценки, CLI exit-коды и миграция, отсутствие сетевого бенчмарка (сервер видит ноль запросов), явная семантика неизвестного overhead, поведенческая регрессия против legacy `calculate_vram`, совпадение чисел Python и JS, валидация URL и копирование/шеринг в реальном браузере (включая чтение настоящего буфера обмена в Firefox).

Сборка пакета и smoke из чистого окружения:

```bash
python -m build
python -m venv /tmp/venv && /tmp/venv/bin/pip install dist/*.whl
/tmp/venv/bin/vram-calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192 --json
```

CI (`.github/workflows/ci.yml`) гоняет юнит-тесты на Python 3.9/3.11/3.13, node-тесты (без «мягкого» fallback), проверку актуальности `index.html`, сборку wheel + smoke из чистого venv и UI-тесты в chromium и firefox; отсутствие любого из обязательных браузеров валит джобу.

## Чего в этом релизе нет

- Никакой проверки на реальной GPU: GPU-прогон этим релизом не выполнялся, поле `verification.gpu_runtime_checked` всегда `false`.
- Нет MoE/MLA/sliding-window/гибридных архитектур и квантованного KV.
- Нет tensor-parallel репликации, CUDA graphs, prefix caching, padding paged-attention.
- Нет телеметрии, аккаунтов и обязательной подписки: калькулятор полностью работает без Telegram.

Если цифры расходятся с вашим рантаймом — это ожидаемо: ledger описывает байты тензоров, а не поведение аллокатора. Присылайте измеренные значения — добавим как явную границу.

## Лицензия

MIT — см. [LICENSE](LICENSE). Метаданные моделей принадлежат их авторам (Qwen, apache-2.0).