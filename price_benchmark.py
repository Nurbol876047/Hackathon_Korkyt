#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 3 из 5 — ценовой бенчмаркинг: сравнение цены каждого договора с рыночной
медианой по укрупнённой категории услуг.

В отличие от Этапа 2 (точечный подбор альтернатив под один договор) здесь считается
групповая статистика: все договоры разбиваются на категории, внутри каждой категории
собираются цены договоров + цены синтетического рынка, и по отклонению от медианы
каждому договору присваивается уровень риска.

Вход:   contracts.json        — Этап 1 (extract_stage1.py)
        market_database.json  — Этап 2 (find_alternatives.py), синтетический рынок
Выход:  price_benchmark.json  — по одной записи на договор (ключ стыковки: source_file)
        category_stats.json   — сводная статистика по категориям (для дашборда Этапа 5)

Запуск:
    export GEMINI_API_KEY="ваш_ключ"
    python price_benchmark.py --contracts contracts.json --market market_database.json \
                              --output price_benchmark.json
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=r"(?s).*google\.generativeai")

import google.generativeai as genai              # noqa: E402
from google.generativeai import protos           # noqa: E402
from google.api_core import exceptions as gexc   # noqa: E402


# ---------------------------------------------------------------------------
# Настройки — пороги риска вынесены сюда, чтобы легко подстроить на демо
# ---------------------------------------------------------------------------

# Отклонение цены договора от медианы категории, в процентах
RISK_THRESHOLD_CHECK_PCT = 15.0   # свыше этого — "требует проверки"
RISK_THRESHOLD_HIGH_PCT = 40.0    # свыше этого — "высокий риск"
LOW_PRICE_NOTE_PCT = -40.0        # цена НИЖЕ медианы сильнее этого — отдельная пометка (риск не меняется)

RISK_NORMAL = "норма"
RISK_CHECK = "требует проверки"
RISK_HIGH = "высокий риск"
RISK_NO_DATA = "недостаточно данных"   # категория слишком мала или у договора нет суммы

MIN_SAMPLE_SIZE = 3           # минимум цен в категории, чтобы медиана считалась осмысленной

# Контекст по районам: район считается "систематически дороже", если средн. отклонение
# его договоров превышает среднее по региону более чем на DISTRICT_DELTA_PP проц. пунктов
DISTRICT_MIN_CONTRACTS = 2
DISTRICT_DELTA_PP = 10.0

# Параметры Gemini (как в Этапах 1–2). gemini-2.0-flash отключена Google → gemini-3.6-flash
DEFAULT_MODEL = "gemini-3.6-flash"
MAX_RETRIES = 2
RETRY_BASE_DELAY = 5
RATE_LIMIT_DELAY = 30
REQUEST_TIMEOUT = 180

FALLBACK_CATEGORY = "прочие услуги"   # для договоров, которые модель не отнесла ни к одной категории

SYSTEM_PROMPT = """Ты — аналитик государственных закупок Республики Казахстан.
Помогаешь аудитору группировать закупки в укрупнённые категории услуг для ценового
сравнения. Отвечай только JSON по заданной схеме, на русском языке."""


# ---------------------------------------------------------------------------
# Запрос к Gemini со structured output и повторными попытками (как в Этапе 2)
# ---------------------------------------------------------------------------

def ask_json(model: genai.GenerativeModel, prompt: str, schema: protos.Schema,
             temperature: float = 0.0):
    total_attempts = MAX_RETRIES + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
                request_options={"timeout": REQUEST_TIMEOUT},
            )
            if not response.candidates:
                reason = getattr(response.prompt_feedback, "block_reason", None)
                raise RuntimeError(f"пустой ответ модели (block_reason={reason})")
            finish = response.candidates[0].finish_reason
            if finish.name != "STOP":
                raise RuntimeError(f"генерация прервана: finish_reason={finish.name}")
            return json.loads(response.text)
        except Exception as exc:  # noqa: BLE001
            if attempt == total_attempts:
                raise
            delay = (RATE_LIMIT_DELAY if isinstance(exc, gexc.ResourceExhausted)
                     else RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            print(f"  ⚠ попытка {attempt}/{total_attempts} не удалась: "
                  f"{type(exc).__name__}: {str(exc)[:150]} — повтор через {delay} с", flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Шаг 1. Категоризация услуг через Gemini
# ---------------------------------------------------------------------------

S, T = protos.Schema, protos.Type


def schema_categorization() -> protos.Schema:
    """
    Ответ модели: список укрупнённых категорий. Каждая категория содержит номера
    договоров (C-номера) и номера рыночных категорий из Этапа 2 (M-номера),
    которые к ней относятся. Модель работает с номерами, а не с текстами —
    так она не может исказить исходные данные.
    """
    return S(
        type=T.OBJECT,
        properties={
            "categories": S(
                type=T.ARRAY,
                items=S(
                    type=T.OBJECT,
                    properties={
                        "name": S(type=T.STRING,
                                  description="Короткое название категории (2–5 слов, нижний регистр)."),
                        "description": S(type=T.STRING,
                                         description="Что входит в категорию (1 предложение)."),
                        "contract_ids": S(type=T.ARRAY, items=S(type=T.INTEGER),
                                          description="Номера договоров, отнесённых к категории."),
                        "market_category_ids": S(type=T.ARRAY, items=S(type=T.INTEGER),
                                                 description="Номера рыночных категорий, "
                                                             "соответствующих этой категории."),
                    },
                    required=["name", "description", "contract_ids", "market_category_ids"],
                ),
            ),
        },
        required=["categories"],
    )


def categorize_services(model: genai.GenerativeModel, contracts: list,
                        market_categories: list) -> tuple[dict, dict]:
    """
    Через Gemini группирует ВСЕ тексты услуг в укрупнённые категории. Категории
    модель предлагает сама на основе реальных текстов, вручную ничего не задаётся.
    Одновременно модель привязывает к предложенным категориям рыночные категории
    из market_database.json (Этап 2), чтобы их цены попали в статистику.

    contracts          — записи contracts.json (индекс в списке = номер договора)
    market_categories  — список (name, description) из market_database.json

    Возвращает:
      assignments — {индекс договора: имя категории}
      categories  — {имя категории: {"description": str, "market_categories": [имена]}}
    """
    # В категоризацию идут только договоры с текстом услуги; без текста — не классифицируем
    indexed = [(i, c) for i, c in enumerate(contracts) if service_text_of(c)]
    if not indexed:
        return {}, {}

    contracts_block = "\n".join(f"C{i + 1}: {service_text_of(c)}" for i, c in indexed)
    market_block = ("\n".join(f"M{j + 1}: {name} — {desc}" for j, (name, desc) in enumerate(market_categories))
                    or "(нет)")

    prompt = f"""Сгруппируй закупки в укрупнённые категории услуг для ценового сравнения.

ДОГОВОРЫ (номер: описание предмета закупки):
{contracts_block}

РЫНОЧНЫЕ КАТЕГОРИИ из базы конкурентов (номер: название — описание):
{market_block}

Правила:
- Категории предложи сам, исходя из реальных текстов: объединяй услуги с одинаковым типом
  результата и сопоставимой ценовой природой (например: разработка мобильных приложений,
  разработка веб-систем, лицензии ПО, AV/конференц-оборудование, ремонт и монтаж сетей,
  поставка компьютерной техники, обучение/консалтинг). Это примеры формата, а не готовый список.
- Не смешивай разные типы: разработка ≠ лицензия ≠ поставка оборудования ≠ ремонт.
- Не дроби слишком мелко: категория из одного договора допустима только если он действительно
  ни на что не похож. Обычно категорий заметно меньше, чем договоров.
- КАЖДЫЙ номер договора должен попасть ровно в одну категорию (числа без буквы C).
- Каждой категории припиши подходящие рыночные категории (числа без буквы M). Рыночная категория
  может относиться только к одной категории; если не подходит ни к одной — не указывай её.
- name — 2–5 слов, нижний регистр."""

    result = ask_json(model, prompt, schema_categorization())

    assignments: dict = {}
    categories: dict = {}
    used_market: set = set()
    for cat in result.get("categories", []):
        name = str(cat.get("name", "")).strip().lower()
        if not name:
            continue
        # Одинаковые названия от модели сливаем в одну категорию
        entry = categories.setdefault(name, {"description": str(cat.get("description", "")).strip(),
                                             "market_categories": []})
        for cid in cat.get("contract_ids", []) or []:
            try:
                idx = int(cid) - 1
            except (TypeError, ValueError):
                continue
            # Первое назначение выигрывает; номера вне диапазона игнорируем
            if 0 <= idx < len(contracts) and idx not in assignments and service_text_of(contracts[idx]):
                assignments[idx] = name
        for mid in cat.get("market_category_ids", []) or []:
            try:
                j = int(mid) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= j < len(market_categories) and j not in used_market:
                used_market.add(j)
                entry["market_categories"].append(market_categories[j][0])

    # Договоры, которые модель пропустила, — в резервную категорию, чтобы не потерять
    missed = [i for i, _ in indexed if i not in assignments]
    if missed:
        print(f"  ⚠ модель не классифицировала {len(missed)} договор(ов) — "
              f"отнесены к «{FALLBACK_CATEGORY}»", flush=True)
        categories.setdefault(FALLBACK_CATEGORY, {"description": "договоры, не отнесённые моделью "
                                                                 "ни к одной категории",
                                                  "market_categories": []})
        for i in missed:
            assignments[i] = FALLBACK_CATEGORY

    # Пустые категории (без договоров) в статистике не нужны
    categories = {n: v for n, v in categories.items() if any(a == n for a in assignments.values())}
    return assignments, categories


def service_text_of(contract: dict) -> str:
    """Текст услуги для категоризации: нормализованный (Этап 1), иначе — исходный."""
    return (contract.get("service_text_normalized") or contract.get("service_text") or "").strip().lower()


# ---------------------------------------------------------------------------
# Шаг 2. Статистика по категориям (чистый Python, без модели)
# ---------------------------------------------------------------------------

def compute_stats(prices: list) -> dict:
    """Медиана, среднее, стандартное отклонение, min/max по списку цен."""
    if not prices:
        return {"median_price": 0, "mean_price": 0, "std_price": 0, "min_price": 0, "max_price": 0}
    return {
        "median_price": round(statistics.median(prices)),
        "mean_price": round(statistics.mean(prices)),
        "std_price": round(statistics.stdev(prices)) if len(prices) > 1 else 0,
        "min_price": round(min(prices)),
        "max_price": round(max(prices)),
    }


def amount_of(contract: dict) -> float:
    try:
        value = float(contract.get("amount_kzt") or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if value > 0 else 0.0


def build_category_stats(contracts: list, assignments: dict, categories: dict,
                         market_db: dict) -> dict:
    """
    Для каждой категории собирает цены: договоры (amount_kzt > 0) + компании из
    привязанных рыночных категорий Этапа 2 — и считает статистику.
    """
    stats = {}
    for name, meta in categories.items():
        contract_prices = [amount_of(contracts[i]) for i, cat in assignments.items()
                           if cat == name and amount_of(contracts[i]) > 0]
        market_prices = []
        for mcat in meta["market_categories"]:
            for company in market_db.get("categories", {}).get(mcat, {}).get("companies", []):
                try:
                    p = float(company.get("price_kzt") or 0)
                except (TypeError, ValueError):
                    continue
                if p > 0:
                    market_prices.append(p)

        all_prices = contract_prices + market_prices
        stats[name] = {
            "description": meta["description"],
            **compute_stats(all_prices),
            "contracts_count": sum(1 for cat in assignments.values() if cat == name),
            "contracts_with_price": len(contract_prices),
            "market_prices_count": len(market_prices),
            "sample_size": len(all_prices),
            "market_categories": meta["market_categories"],
            "risk_counts": {RISK_NORMAL: 0, RISK_CHECK: 0, RISK_HIGH: 0, RISK_NO_DATA: 0},
        }
    return stats


# ---------------------------------------------------------------------------
# Шаг 3–4. Отклонение от медианы и уровень риска
# ---------------------------------------------------------------------------

def price_deviation_score(contract_price: float, category_median: float) -> tuple[float, float]:
    """
    Возвращает (отклонение в %, во сколько раз):
        deviation_pct = (price - median) / median * 100   (+ дороже медианы, − дешевле)
        ratio         = price / median
    """
    if category_median <= 0 or contract_price <= 0:
        return 0.0, 0.0
    return (round((contract_price - category_median) / category_median * 100, 1),
            round(contract_price / category_median, 2))


def risk_level_by_deviation(deviation_pct: float) -> str:
    """Уровень риска по порогам из констант в начале файла."""
    if deviation_pct > RISK_THRESHOLD_HIGH_PCT:
        return RISK_HIGH
    if deviation_pct > RISK_THRESHOLD_CHECK_PCT:
        return RISK_CHECK
    return RISK_NORMAL


# ---------------------------------------------------------------------------
# Шаг 5. Контекст по районам
# ---------------------------------------------------------------------------

_DISTRICT_RE = re.compile(r"([\w-]+)\s+(?:район|ауданы)", re.IGNORECASE)
_CITY_RE = re.compile(r"(?:г\.|город|қ\.)\s*([\w-]+)", re.IGNORECASE)


def normalize_district(text: str) -> str:
    """
    Приводит свободный текст поля district к ключу группировки:
    'Актюбинская область, Мартукский район, с. Мартук' → 'мартукский район'.
    Если района нет — город, иначе вся строка целиком.
    """
    text = (text or "").strip().lower().replace("ё", "е")
    if not text:
        return ""
    m = _DISTRICT_RE.search(text)
    if m:
        return f"{m.group(1)} район"
    m = _CITY_RE.search(text)
    if m:
        return f"г. {m.group(1)}"
    return re.sub(r"\s+", " ", text)


def add_district_context(records: list) -> dict:
    """
    Сравнивает среднее отклонение от медианы по договорам района со средним по региону
    (все договоры). Уровень риска НЕ меняет — только заполняет district_context.
    Отклонения нормированы по категориям, поэтому их можно усреднять между категориями.
    Возвращает сводку по районам (для консоли).
    """
    scored = [r for r in records if r["risk_level"] != RISK_NO_DATA]
    if not scored:
        return {}
    region_avg = statistics.mean(r["price_deviation_pct"] for r in scored)

    by_district = defaultdict(list)
    for r in scored:
        if r["district_key"]:
            by_district[r["district_key"]].append(r["price_deviation_pct"])

    summary = {}
    for key, devs in by_district.items():
        avg = statistics.mean(devs)
        summary[key] = {"contracts": len(devs), "avg_deviation_pct": round(avg, 1),
                        "region_avg_pct": round(region_avg, 1),
                        "systematically_higher": len(devs) >= DISTRICT_MIN_CONTRACTS
                                                 and avg - region_avg > DISTRICT_DELTA_PP}

    for r in records:
        info = summary.get(r["district_key"])
        if not info or info["contracts"] < DISTRICT_MIN_CONTRACTS:
            r["district_context"] = ""
            continue
        base = (f"Район «{r['district_key']}»: среднее отклонение от медианы {info['avg_deviation_pct']:+.1f}% "
                f"при {info['region_avg_pct']:+.1f}% по региону в целом ({info['contracts']} договоров)")
        r["district_context"] = (base + " — цены систематически выше" if info["systematically_higher"]
                                 else base + " — без систематического завышения")
    return summary


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 3: ценовой бенчмаркинг договоров относительно медианы категории.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--contracts", default="contracts.json", help="contracts.json из Этапа 1")
    parser.add_argument("--market", default="market_database.json",
                        help="market_database.json из Этапа 2 (если нет — только цены договоров)")
    parser.add_argument("--output", default="price_benchmark.json", help="выходной JSON по договорам")
    parser.add_argument("--stats", default="category_stats.json", help="сводная статистика по категориям")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели Gemini")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ОШИБКА: не задана переменная окружения GEMINI_API_KEY.\n"
              "Получите ключ на https://aistudio.google.com/apikey и выполните:\n"
              '    export GEMINI_API_KEY="ваш_ключ"', file=sys.stderr)
        return 1

    contracts_path = Path(args.contracts)
    if not contracts_path.is_file():
        print(f"ОШИБКА: файл не найден: {contracts_path} (сначала запустите extract_stage1.py)",
              file=sys.stderr)
        return 1
    with open(contracts_path, encoding="utf-8") as f:
        contracts = json.load(f)
    if not contracts:
        print("ОШИБКА: в contracts.json нет записей", file=sys.stderr)
        return 1

    # База рынка необязательна: без неё статистика строится только по договорам
    market_db = {"categories": {}}
    market_path = Path(args.market)
    if market_path.is_file():
        with open(market_path, encoding="utf-8") as f:
            market_db = json.load(f)
    else:
        print(f"⚠ {market_path} не найден — статистика только по ценам договоров", flush=True)
    market_categories = [(name, str(v.get("description", ""))) for name, v in market_db.get("categories", {}).items()]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model, system_instruction=SYSTEM_PROMPT)

    print(f"Модель: {args.model}. Договоров: {len(contracts)}. "
          f"Рыночных категорий Этапа 2: {len(market_categories)}\n")

    # --- Шаг 1. Категоризация (один запрос к модели на весь набор) ---
    print("Шаг 1/4: категоризация услуг через Gemini...", flush=True)
    try:
        assignments, categories = categorize_services(model, contracts, market_categories)
    except Exception as exc:  # noqa: BLE001
        print(f"ОШИБКА: категоризация не удалась после {MAX_RETRIES + 1} попыток: "
              f"{type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 1
    print(f"  категорий: {len(categories)}")
    for name, meta in categories.items():
        n = sum(1 for c in assignments.values() if c == name)
        linked = ", ".join(meta["market_categories"]) or "—"
        print(f"  • {name} — договоров: {n}; рынок Этапа 2: {linked}")

    # --- Шаг 2. Статистика по категориям ---
    print("\nШаг 2/4: статистика по категориям...", flush=True)
    category_stats = build_category_stats(contracts, assignments, categories, market_db)
    for name, st in category_stats.items():
        print(f"  • {name}: медиана {st['median_price']:,} тг, среднее {st['mean_price']:,} тг, "
              f"σ {st['std_price']:,}, min {st['min_price']:,}, max {st['max_price']:,} "
              f"(выборка {st['sample_size']}: договоров {st['contracts_with_price']} + рынок {st['market_prices_count']})")

    # --- Шаг 3–4. Отклонение и риск по каждому договору ---
    print("\nШаг 3/4: отклонение от медианы и уровень риска...", flush=True)
    records = []
    total = len(contracts)
    for i, contract in enumerate(contracts):
        contract_id = contract.get("contract_number") or contract.get("source_file", "")
        category = assignments.get(i, "")
        price = amount_of(contract)
        st = category_stats.get(category)
        note = ""

        if not category:
            deviation, ratio, risk, median = 0.0, 0.0, RISK_NO_DATA, 0
            note = "нет описания услуги — категория не определена"
        elif price <= 0:
            deviation, ratio, risk, median = 0.0, 0.0, RISK_NO_DATA, st["median_price"]
            note = "сумма договора не извлечена"
        elif st["sample_size"] < MIN_SAMPLE_SIZE:
            deviation, ratio = price_deviation_score(price, st["median_price"])
            risk, median = RISK_NO_DATA, st["median_price"]
            note = f"в категории всего {st['sample_size']} цен(ы) — медиана ненадёжна"
        else:
            median = st["median_price"]
            deviation, ratio = price_deviation_score(price, median)
            risk = risk_level_by_deviation(deviation)
            if deviation < LOW_PRICE_NOTE_PCT:
                note = "цена значительно ниже медианы — проверить объём и качество исполнения"

        if st:
            st["risk_counts"][risk] += 1

        rec = {
            "source_file": contract.get("source_file", ""),
            "contract_id": contract_id,
            "supplier": contract.get("supplier", ""),
            "amount_kzt": contract.get("amount_kzt", 0),
            "category": category,
            "category_median_price": median,
            "price_deviation_pct": deviation,
            "price_ratio": ratio,
            "risk_level": risk,
            "sample_size": st["sample_size"] if st else 0,
            "district": contract.get("district", ""),
            "district_key": normalize_district(contract.get("district", "")),
            "district_context": "",
            "benchmark_note": note,
        }
        records.append(rec)
        print(f"  [{i + 1}/{total}] {contract_id} — {category or '—'} | "
              f"{int(price):,} тг vs медиана {int(median):,} тг → {deviation:+.1f}% | {risk}"
              + (f" ({note})" if note else ""), flush=True)

    # --- Шаг 5. Контекст по районам ---
    print("\nШаг 4/4: контекст по районам...", flush=True)
    district_summary = add_district_context(records)
    if district_summary:
        for key, info in sorted(district_summary.items(), key=lambda kv: -kv[1]["avg_deviation_pct"]):
            flag = " ⚠ систематически выше" if info["systematically_higher"] else ""
            print(f"  • {key}: {info['contracts']} дог., среднее отклонение "
                  f"{info['avg_deviation_pct']:+.1f}% (регион {info['region_avg_pct']:+.1f}%){flag}")
    else:
        print("  районы не указаны или нет оценённых договоров")

    # --- Сохранение ---
    output_path, stats_path = Path(args.output), Path(args.stats)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(category_stats, f, ensure_ascii=False, indent=2)

    counts = {lvl: sum(1 for r in records if r["risk_level"] == lvl)
              for lvl in (RISK_NORMAL, RISK_CHECK, RISK_HIGH, RISK_NO_DATA)}
    print(f"\nГотово: {len(records)} договоров. "
          + ", ".join(f"{lvl}: {n}" for lvl, n in counts.items()))
    print(f"Пороги: проверка > +{RISK_THRESHOLD_CHECK_PCT:.0f}%, высокий риск > +{RISK_THRESHOLD_HIGH_PCT:.0f}%")
    print(f"Результат: {output_path}\nСтатистика: {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
