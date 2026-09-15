# Hackathon Korkyt — AI-аудит государственных закупок Казахстана

MVP-платформа для внутреннего аудитора / ревизионной комиссии / аналитика акимата.
Принимает PDF договоров и актов госзакупок и автоматически:

- извлекает структурированные данные из документа (Gemini читает PDF нативно, без OCR);
- находит более дешёвые рыночные альтернативы той же услуге и считает потенциальную экономию;
- сравнивает цену закупки с медианой по категории услуг и присваивает уровень риска;
- выявляет признаки дробления закупки и несоответствия продукта ТЗ;
- проверяет целостность PDF (редактирование после подписания) и собирает итоговый дашборд.

Результат работы системы — **индикатор риска для проверки**, а не юридический вердикт.

## Этапы pipeline

| # | Скрипт | Вход | Выход | Статус |
|---|--------|------|-------|--------|
| 1 | `extract_stage1.py` | папка с PDF | `contracts.json`, `contracts.csv` | готов |
| 2 | `find_alternatives.py` | `contracts.json` | `alternatives.json`, `market_database.json` | готов |
| 3 | `price_benchmark.py` | `contracts.json`, `market_database.json` | `price_benchmark.json`, `category_stats.json` | готов |
| 4 | соответствие ТЗ + дробление закупок | — | — | в работе |
| 5 | целостность PDF + Streamlit-дашборд | — | — | в работе |

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

У каждого скрипта есть `--help`. Пороги риска для Этапа 3 вынесены в константы в начале `price_benchmark.py`.

## Примечания

- Модель по умолчанию — `gemini-3.6-flash` (`gemini-2.0-flash` отключена Google и возвращает 404); меняется флагом `--model`.
- Все ответы модели получаются через structured output (`response_schema`) — JSON гарантированно валиден.
- Рыночная база (`market_database.json`) синтетическая: компании вымышлены и сгенерированы для демо, реальные и персональные данные не используются.
- Режим `--mode websearch` в Этапе 2 — заглушка с готовым интерфейсом; при пустом результате используется синтетический рынок.
