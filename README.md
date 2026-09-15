# Hackathon Korkyt — AI-аудит государственных закупок Казахстана

MVP-платформа для внутреннего аудитора / ревизионной комиссии / аналитика акимата.
Принимает PDF договоров и актов госзакупок и автоматически:

- извлекает структурированные данные из документа (Gemini читает PDF нативно, без OCR);
- находит более дешёвые рыночные альтернативы той же услуге и считает потенциальную экономию;
- сравнивает цену закупки с медианой по категории услуг и присваивает уровень риска;
- выявляет признаки дробления закупки и несоответствия продукта ТЗ;
- проверяет целостность PDF (метаданные, ПО-редактор, инкрементальные сохранения, текстовый слой поверх скана) — без ИИ;
- собирает всё в Streamlit-дашборд: сводная панель, фильтры, таблица с цветовой индикацией риска и детальный разбор договора.

Результат работы системы — **индикатор риска для проверки**, а не юридический вердикт.

## Этапы pipeline

| # | Скрипт | Вход | Выход | Статус |
|---|--------|------|-------|--------|
| 1 | `extract_stage1.py` | папка с PDF | `contracts.json`, `contracts.csv` | готов |
| 2 | `find_alternatives.py` | `contracts.json` | `alternatives.json`, `market_database.json` | готов |
| 3 | `price_benchmark.py` | `contracts.json`, `market_database.json` | `price_benchmark.json`, `category_stats.json` | готов |
| 4 | `compliance_and_fragmentation.py` | `contracts.json` | `results/tor_compliance.json`, `results/fragmentation_clusters.json` | готов |
| 5 | `dashboard.py` (Streamlit) | папка с результатами 1–4 | веб-дашборд | готов (без проверки целостности PDF) |

Ключ стыковки записей между этапами — поле `source_file` (имя исходного PDF).

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY="ваш_ключ"   # https://aistudio.google.com/apikey
```

## Запуск

```bash
.venv/bin/python extract_stage1.py --input ./contracts --output contracts.json
.venv/bin/python find_alternatives.py --input contracts.json --output alternatives.json --mode synthetic
.venv/bin/python price_benchmark.py --contracts contracts.json --market market_database.json --output price_benchmark.json
```

```bash
.venv/bin/python compliance_and_fragmentation.py --input contracts.json --output-dir ./results
.venv/bin/streamlit run dashboard.py                      # http://localhost:8501
```

Дашборд читает файлы из `./results` (или из `./demo_results`, если результатов ещё нет; папку можно задать явно: `streamlit run dashboard.py -- --results ./папка`).
Без PDF и ключа Gemini можно посмотреть дашборд на вымышленных данных: `.venv/bin/python demo_data.py --output-dir ./demo_results`.

```bash
.venv/bin/python compliance_and_fragmentation.py --input contracts.json --output-dir ./results
.venv/bin/python integrity_check.py --input ./contracts --output ./results/integrity_check.json
.venv/bin/streamlit run dashboard.py -- --results ./results
```

У каждого скрипта есть `--help`. Пороги риска вынесены в константы в начале каждого скрипта
(Этап 3 — `price_benchmark.py`, Этап 4 — `compliance_and_fragmentation.py`, Этап 5А — `integrity_check.py`,
общий risk_level — `dashboard.py`).

## Дашборд

`streamlit run dashboard.py` открывает http://localhost:8501. Папка результатов: `./results`, а если её нет — `./demo_results`.

- **Сводная панель:** число договоров, общая сумма, сумма договоров с высоким риском, суммарная потенциальная экономия.
- **Фильтры:** уровень риска, категория услуг, диапазон суммы.
- **Таблица договоров** с цветовой индикацией общего risk_level = максимум из четырёх сигналов:
  цена vs медиана (Этап 3), дробление (4Б), соответствие ТЗ (4А), целостность PDF (5А).
- **Детали договора:** plotly-график «цена / медиана категории», альтернативные поставщики и экономия,
  кластер дробления, соответствие ТЗ с спорными формулировками, флаги целостности PDF.
- Если какой-то этап не запускался, его блок показывает «данные недоступны» — дашборд открывается уже после Этапа 1.

### Демо без PDF и ключа

```bash
.venv/bin/python demo_data.py                     # demo_results/*.json + demo_contracts/*.pdf (всё вымышленное)
.venv/bin/python integrity_check.py --input ./demo_contracts --output ./demo_results/integrity_check.json
.venv/bin/streamlit run dashboard.py
```

## Примечания

- Модель по умолчанию — `gemini-3.6-flash` (`gemini-2.0-flash` отключена Google и возвращает 404); меняется флагом `--model`.
- Все ответы модели получаются через structured output (`response_schema`) — JSON гарантированно валиден.
- Рыночная база (`market_database.json`) синтетическая: компании вымышлены и сгенерированы для демо, реальные и персональные данные не используются.
- Режим `--mode websearch` в Этапе 2 — заглушка с готовым интерфейсом; при пустом результате используется синтетический рынок.
