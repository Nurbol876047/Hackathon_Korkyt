#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 3 из 5 — ценовой бенчмаркинг: сравнение цены каждого договора с рыночной
медианой по укрупнённой категории услуг.

Вход:   contracts.json        — Этап 1 (extract_stage1.py)
        market_database.json  — Этап 2 (find_alternatives.py), синтетический рынок
Выход:  price_benchmark.json  — по одной записи на договор (ключ стыковки: source_file)
        category_stats.json   — сводная статистика по категориям (для дашборда Этапа 5)

Запуск:
    export OPENAI_API_KEY="ваш_ключ"
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
from collections import defaultdict
from pathlib import Path
from typing import List

from openai import OpenAI
from openai import RateLimitError, APIError
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

RISK_THRESHOLD_CHECK_PCT = 15.0
RISK_THRESHOLD_HIGH_PCT = 40.0
LOW_PRICE_NOTE_PCT = -40.0

RISK_NORMAL = "норма"
RISK_CHECK = "требует проверки"
RISK_HIGH = "высокий риск"
RISK_NO_DATA = "недостаточно данных"

MIN_SAMPLE_SIZE = 3

DISTRICT_MIN_CONTRACTS = 2
DISTRICT_DELTA_PP = 10.0

DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 2
RETRY_BASE_DELAY = 5
RATE_LIMIT_DELAY = 30
REQUEST_TIMEOUT = 180

FALLBACK_CATEGORY = "прочие услуги"

SYSTEM_PROMPT = """Ты — аналитик государственных закупок Республики Казахстан.
Помогаешь аудитору группировать закупки в укрупнённые категории услуг для ценового
сравнения. Отвечай только JSON по заданной схеме, на русском языке."""

# ---------------------------------------------------------------------------
# JSON-схемы ответов (Pydantic)
# ---------------------------------------------------------------------------

class CategoryItem(BaseModel):
    name: str = Field(description="Короткое название категории (2–5 слов, нижний регистр).")
    description: str = Field(description="Что входит в категорию (1 предложение).")
    contract_ids: List[int] = Field(description="Номера договоров, отнесённых к категории.")
    market_category_ids: List[int] = Field(description="Номера рыночных категорий, соответствующих этой категории.")

class CategorizationResponse(BaseModel):
    categories: List[CategoryItem]

# ---------------------------------------------------------------------------
# Запрос к OpenAI со structured output
# ---------------------------------------------------------------------------

def ask_json(client: OpenAI, model_name: str, prompt: str, schema_class: type[BaseModel], temperature: float = 0.0):
    total_attempts = MAX_RETRIES + 1
    for attempt in range(1, total_attempts + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=schema_class,
                temperature=temperature,
                timeout=REQUEST_TIMEOUT
            )
            return response.choices[0].message.parsed.model_dump()
        except Exception as exc:
            if attempt == total_attempts:
                raise
            delay = RATE_LIMIT_DELAY if isinstance(exc, RateLimitError) else RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"  ⚠ попытка {attempt}/{total_attempts} не удалась: "
                  f"{type(exc).__name__}: {str(exc)[:150]} — повтор через {delay} с", flush=True)
            time.sleep(delay)

# ---------------------------------------------------------------------------
# Шаг 1. Категоризация услуг
# ---------------------------------------------------------------------------

def categorize_services(client: OpenAI, model_name: str, contracts: list,
                        market_categories: list) -> tuple[dict, dict]:
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
- Категории предложи сам, исходя из реальных текстов.
- Не смешивай разные типы: разработка ≠ лицензия ≠ поставка оборудования ≠ ремонт.
- КАЖДЫЙ номер договора должен попасть ровно в одну категорию (числа без буквы C).
- Каждой категории припиши подходящие рыночные категории (числа без буквы M).
- name — 2–5 слов, нижний регистр."""

    result = ask_json(client, model_name, prompt, CategorizationResponse)

    assignments: dict = {}
    categories: dict = {}
    used_market: set = set()
    for cat in result.get("categories", []):
        name = str(cat.get("name", "")).strip().lower()
        if not name:
            continue
        entry = categories.setdefault(name, {"description": str(cat.get("description", "")).strip(),
                                             "market_categories": []})
        for cid in cat.get("contract_ids", []) or []:
            try:
                idx = int(cid) - 1
            except (TypeError, ValueError):
                continue
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

    missed = [i for i, _ in indexed if i not in assignments]
    if missed:
        print(f"  ⚠ модель не классифицировала {len(missed)} договор(ов) — "
              f"отнесены к «{FALLBACK_CATEGORY}»", flush=True)
        categories.setdefault(FALLBACK_CATEGORY, {"description": "договоры, не отнесённые моделью ни к одной категории",
                                                  "market_categories": []})
        for i in missed:
            assignments[i] = FALLBACK_CATEGORY

    categories = {n: v for n, v in categories.items() if any(a == n for a in assignments.values())}
    return assignments, categories


def service_text_of(contract: dict) -> str:
    return (contract.get("service_text_normalized") or contract.get("service_text") or "").strip().lower()


# ---------------------------------------------------------------------------
# Шаг 2. Статистика по категориям
# ---------------------------------------------------------------------------

def compute_stats(prices: list) -> dict:
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
# Шаг 3–5. Расчёты отклонения и риска (чистый Python)
# ---------------------------------------------------------------------------

def price_deviation_score(price: float, median: float) -> tuple[float, float]:
    if median <= 0:
        return 0.0, 0.0
    diff = price - median
    pct = (diff / median) * 100.0
    ratio = price / median
    return round(pct, 1), round(ratio, 2)


def risk_level_by_deviation(pct: float) -> str:
    if pct > RISK_THRESHOLD_HIGH_PCT:
        return RISK_HIGH
    if pct > RISK_THRESHOLD_CHECK_PCT:
        return RISK_CHECK
    return RISK_NORMAL

def normalize_district(raw: str) -> str:
    s = raw.lower()
    s = re.sub(r"\b(г\.|город|обл\.|область|район|р-н)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def add_district_context(records: list) -> dict:
    d_stats = defaultdict(list)
    region_devs = []

    for r in records:
        key = r.get("district_key")
        dev = r.get("price_deviation_pct", 0)
        if key and r["risk_level"] != RISK_NO_DATA:
            d_stats[key].append(dev)
            region_devs.append(dev)

    if not region_devs:
        return {}

    region_avg = statistics.mean(region_devs)
    summary = {}

    for key, devs in d_stats.items():
        if len(devs) < DISTRICT_MIN_CONTRACTS:
            continue
        avg_dev = statistics.mean(devs)
        sys_high = (avg_dev - region_avg) > DISTRICT_DELTA_PP
        summary[key] = {
            "contracts": len(devs),
            "avg_deviation_pct": round(avg_dev, 1),
            "region_avg_pct": round(region_avg, 1),
            "systematically_higher": sys_high,
        }

    for r in records:
        key = r.get("district_key")
        if key and key in summary:
            info = summary[key]
            r["district_context"] = (
                f"Среднее отклонение по району {info['avg_deviation_pct']:+.1f}% "
                f"(по региону {info['region_avg_pct']:+.1f}%)."
            )
            if info["systematically_higher"]:
                r["district_context"] += " ⚠ Район систематически закупает дороже средних цен."

    return summary

# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 3: ценовой бенчмаркинг по укрупнённым категориям (OpenAI API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--contracts", default="contracts.json", help="Входной JSON Этапа 1")
    parser.add_argument("--market", default="market_database.json", help="Входная база Этапа 2")
    parser.add_argument("--output", default="price_benchmark.json", help="Выходной JSON")
    parser.add_argument("--stats", default="category_stats.json", help="Файл статистики по категориям")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель OpenAI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ОШИБКА: не задана переменная окружения OPENAI_API_KEY.\n", file=sys.stderr)
        return 1

    if not Path(args.contracts).is_file():
        print(f"ОШИБКА: нет файла договоров {args.contracts}", file=sys.stderr)
        return 1
    with open(args.contracts, encoding="utf-8") as f:
        contracts = json.load(f)

    market_db = {}
    if Path(args.market).is_file():
        with open(args.market, encoding="utf-8") as f:
            market_db = json.load(f)

    market_categories = [
        (name, data.get("description", ""))
        for name, data in market_db.get("categories", {}).items()
    ]

    client = OpenAI(api_key=api_key)

    print("Шаг 1/4: категоризация услуг через OpenAI...", flush=True)
    try:
        assignments, categories = categorize_services(client, args.model, contracts, market_categories)
    except Exception as exc:
        print(f"ОШИБКА: категоризация не удалась после {MAX_RETRIES + 1} попыток: "
              f"{type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 1
    print(f"  категорий: {len(categories)}")
    for name, meta in categories.items():
        n = sum(1 for c in assignments.values() if c == name)
        linked = ", ".join(meta["market_categories"]) or "—"
        print(f"  • {name} — договоров: {n}; рынок Этапа 2: {linked}")

    print("\nШаг 2/4: статистика по категориям...", flush=True)
    category_stats = build_category_stats(contracts, assignments, categories, market_db)
    for name, st in category_stats.items():
        print(f"  • {name}: медиана {st['median_price']:,} тг, среднее {st['mean_price']:,} тг, "
              f"σ {st['std_price']:,}, min {st['min_price']:,}, max {st['max_price']:,} "
              f"(выборка {st['sample_size']}: договоров {st['contracts_with_price']} + рынок {st['market_prices_count']})")

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
            note = f"сумма не указана, рыночная медиана: {int(st['median_price']):,} тг".replace(",", " ")
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

    print("\nШаг 4/4: контекст по районам...", flush=True)
    district_summary = add_district_context(records)
    if district_summary:
        for key, info in sorted(district_summary.items(), key=lambda kv: -kv[1]["avg_deviation_pct"]):
            flag = " ⚠ систематически выше" if info["systematically_higher"] else ""
            print(f"  • {key}: {info['contracts']} дог., среднее отклонение "
                  f"{info['avg_deviation_pct']:+.1f}% (регион {info['region_avg_pct']:+.1f}%){flag}")
    else:
        print("  районы не указаны или нет оценённых договоров")

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
