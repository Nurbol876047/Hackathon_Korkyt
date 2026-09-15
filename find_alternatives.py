#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 2 из 5 — поиск альтернативных поставщиков/решений для каждой закупки
и расчёт потенциальной экономии бюджета.

Вход:   contracts.json        — результат Этапа 1 (extract_stage1.py)
Выход:  alternatives.json     — по одному объекту на каждый договор
        market_database.json  — синтетическая база рынка по категориям услуг
                                (кэш: переиспользуется между запусками)

Режимы работы (--mode):
  synthetic  (по умолчанию) — для каждой категории услуг генерируются
                              15–20 вымышленных компаний-конкурентов с реалистичным
                              разбросом цен; база сохраняется в market_database.json.
  websearch                 — поиск реальных компаний в интернете. (TODO-заглушка)

Запуск:
    export OPENAI_API_KEY="ваш_ключ"
    python find_alternatives.py --input contracts.json --output alternatives.json --mode synthetic
"""

import argparse
import difflib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, List, Optional

from openai import OpenAI
from openai import RateLimitError, APIError
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini" # or gpt-4o

MAX_RETRIES = 2            # повторных попыток после первой неудачной (итого 3 запроса)
RETRY_BASE_DELAY = 5       # секунд; удваивается с каждой повторной попыткой
RATE_LIMIT_DELAY = 30      # секунд ожидания при ошибке 429
REQUEST_TIMEOUT = 180      # секунд на один запрос

DEFAULT_MARKET_SIZE = 18   # сколько компаний генерировать на категорию (ТЗ: 15–20)
DEFAULT_TOP_N = 5          # сколько альтернатив оставлять в выводе (ТЗ: 3–5)
MIN_ALTERNATIVES = 3       # если релевантных меньше — предупреждаем в консоли

SYSTEM_PROMPT = """Ты — аналитик рынка IT-услуг, оборудования и программного обеспечения
Республики Казахстан. Помогаешь аудитору государственных закупок сравнивать цену закупки
с рыночными альтернативами. Отвечай только JSON по заданной схеме, на русском языке.
Все суммы — в тенге (KZT), только числа."""

# ---------------------------------------------------------------------------
# JSON-схемы ответов (Pydantic)
# ---------------------------------------------------------------------------

class CategoryResponse(BaseModel):
    category: str = Field(description="Короткое название категории услуги (2–6 слов, нижний регистр, именительный падеж), например 'разработка мобильного приложения'.")
    matched_existing: bool = Field(description="true, если выбрана одна из уже существующих категорий.")
    category_description: str = Field(description="Что обычно входит в услугу этой категории (1–2 предложения).")
    typical_price_min_kzt: float = Field(description="Нижняя граница типичной рыночной цены в Казахстане, тенге.")
    typical_price_max_kzt: float = Field(description="Верхняя граница типичной рыночной цены в Казахстане, тенге.")

class MarketCompany(BaseModel):
    company: str = Field(description="Вымышленное название с ОПФ, например 'ТОО «Digital Dala»'.")
    city: str = Field(description="Город Казахстана.")
    company_size: Literal["малая", "средняя", "крупная"] = Field(description="Размер компании.")
    offer: str = Field(description="Что именно предлагает компания и в каком объёме (1 предложение).")
    tech_stack: str = Field(description="Технологии/платформы через запятую.")
    price_kzt: float = Field(description="Цена предложения в тенге, число.")

class MarketResponse(BaseModel):
    items: List[MarketCompany]

class RelevanceVerdict(BaseModel):
    candidate_id: int = Field(description="Номер кандидата из списка.")
    is_relevant: bool = Field(description="true, если кандидат реально может выполнить ту же услугу.")
    relevance_reason: str = Field(description="Краткое обоснование (1 предложение), почему подходит / не подходит.")

class RelevanceResponse(BaseModel):
    items: List[RelevanceVerdict]

class SearchQueryResponse(BaseModel):
    query: str = Field(description="Короткий поисковый запрос (3–8 слов).")

# ---------------------------------------------------------------------------
# Общий помощник: запрос к OpenAI со structured output
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
            print(f"    ⚠ попытка {attempt}/{total_attempts} не удалась: "
                  f"{type(exc).__name__}: {str(exc)[:150]} — повтор через {delay} с", flush=True)
            time.sleep(delay)

# ---------------------------------------------------------------------------
# База рынка (market_database.json)
# ---------------------------------------------------------------------------

def fmt_kzt(value) -> str:
    return f"{int(round(float(value or 0))):,}".replace(",", " ")

def load_market_database(path: Path) -> dict:
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            db = json.load(f)
        db.setdefault("categories", {})
        return db
    return {
        "is_synthetic": True,
        "note": "Синтетическая база: все компании вымышлены и сгенерированы для демо. "
                "Совпадения с реальными организациями случайны.",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "categories": {},
    }

def save_market_database(db: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def find_existing_category(name: str, existing: list) -> str | None:
    key = name.strip().lower()
    for cat in existing:
        if cat.lower() == key:
            return cat
    close = difflib.get_close_matches(key, [c.lower() for c in existing], n=1, cutoff=0.85)
    if close:
        return next(c for c in existing if c.lower() == close[0])
    return None

# ---------------------------------------------------------------------------
# Шаг 1. Классификация услуги в категорию
# ---------------------------------------------------------------------------

def classify_service_category(client: OpenAI, model_name: str, service_text: str,
                              tech_stack: str, existing_categories: list) -> dict:
    existing_block = ("\n".join(f"- {c}" for c in existing_categories)
                      if existing_categories else "(пока нет)")
    prompt = f"""Определи категорию услуги для закупки.

Описание предмета закупки: {service_text}
Упомянутые технологии: {tech_stack or "не указаны"}

Уже существующие категории в базе:
{existing_block}

Если закупка подходит под одну из существующих категорий, верни её название (буква в букву).
Если не подходит — придумай новую понятную категорию (2–6 слов) и оцени для неё типичный ценовой диапазон на рынке РК."""
    result = ask_json(client, model_name, prompt, CategoryResponse)
    
    cat_name = str(result.get("category", "")).strip()
    match = find_existing_category(cat_name, existing_categories)
    if match:
        result["category"] = match
        result["matched_existing"] = True
    else:
        result["matched_existing"] = False
    return result

# ---------------------------------------------------------------------------
# Шаг 2А. Синтетический режим (генерация вымышленных компаний)
# ---------------------------------------------------------------------------

def generate_synthetic_market(client: OpenAI, model_name: str, service_category: str,
                              category_desc: str, price_min: float, price_max: float,
                              count: int) -> list:
    prompt = f"""Сгенерируй список из {count} вымышленных IT-компаний в Казахстане, которые
предлагают услуги в категории: "{service_category}" ({category_desc}).

Требования:
- Цены предложений должны быть реалистичными и варьироваться от {fmt_kzt(price_min)} до {fmt_kzt(price_max)} тенге.
- Разные города (Астана, Алматы, Шымкент, Караганда и др.).
- Разные размеры компаний (малая, средняя, крупная) и пропорциональный стек технологий.
- Форматы компаний: ТОО, ИП. Названия должны звучать реалистично (как локальные, так и англоязычные).
- Описание оффера должно быть разнообразным."""
    
    companies = ask_json(client, model_name, prompt, MarketResponse, temperature=0.8)
    return companies.get("items", [])

# ---------------------------------------------------------------------------
# Шаг 2Б. Режим Web Search (поиск реальных альтернатив)
# ---------------------------------------------------------------------------

def build_search_query(client: OpenAI, model_name: str, service_text: str) -> str:
    prompt = f"""Сформулируй короткий поисковый запрос (3-8 слов) для поиска коммерческих
предложений или компаний в Казахстане, которые оказывают следующую услугу:
{service_text}"""
    return str(ask_json(client, model_name, prompt, SearchQueryResponse).get("query", "")).strip()

def search_market_alternatives(client: OpenAI, model_name: str, service_text: str) -> list:
    query = build_search_query(client, model_name, service_text)
    print(f"    (websearch) Поисковый запрос: [{query}]")
    return []

# ---------------------------------------------------------------------------
# Шаг 3. Фильтрация релевантных кандидатов
# ---------------------------------------------------------------------------

def filter_relevant_matches(client: OpenAI, model_name: str, service_text: str,
                            contract_price: float, candidates: list) -> list:
    if not candidates:
        return []
    
    c_list = ""
    for i, c in enumerate(candidates):
        c_list += f"[{i}] {c.get('company')}, {fmt_kzt(c.get('price_kzt'))} тг: {c.get('offer')}\n"

    prompt = f"""Оцени, подходят ли предложенные кандидаты в качестве альтернативы
для государственной закупки.

Предмет закупки: {service_text}
Сумма закупки: {fmt_kzt(contract_price)} тенге (используй только как ориентир масштаба).

Кандидаты:
{c_list}

Верни массив оценок для каждого кандидата по его номеру."""
    
    verdicts = ask_json(client, model_name, prompt, RelevanceResponse)
    
    verdict_map = {}
    for v in verdicts.get("items", []):
        try:
            vid = int(v.get("candidate_id", -1))
            verdict_map[vid] = v
        except ValueError:
            pass

    relevant = []
    for i, c in enumerate(candidates):
        v = verdict_map.get(i)
        if v and v.get("is_relevant"):
            out = dict(c)
            out["relevance_reason"] = v.get("relevance_reason", "")
            relevant.append(out)
    return relevant

# ---------------------------------------------------------------------------
# Основной конвейер одного договора
# ---------------------------------------------------------------------------

def make_result(contract: dict, category: str = "", alts: list = None,
                savings: float = 0.0, note: str = "") -> dict:
    res = dict(contract)
    res["market_category"] = category
    res["alternatives"] = alts or []
    res["potential_savings_kzt"] = savings
    res["note"] = note
    return res

def process_contract(client: OpenAI, model_name: str, contract: dict, db: dict, db_path: Path,
                     mode: str, market_size: int, top_n: int) -> dict:
    service_text = contract.get("service_text", "").strip()
    tech_stack = contract.get("tech_stack_mentioned", "").strip()
    contract_price = float(contract.get("amount_kzt") or 0.0)

    if not service_text or contract_price <= 0:
        return make_result(contract, note="Пропуск: нет описания услуги или цены")

    print("    • определение категории...", flush=True)
    existing_cats = list(db["categories"].keys())
    cat_info = classify_service_category(client, model_name, service_text, tech_stack, existing_cats)
    cat_name = cat_info.get("category") or "Разное"

    if cat_name not in db["categories"]:
        db["categories"][cat_name] = {
            "description": cat_info.get("category_description", ""),
            "typical_price_min_kzt": cat_info.get("typical_price_min_kzt", 0.0),
            "typical_price_max_kzt": cat_info.get("typical_price_max_kzt", 0.0),
            "companies": [],
        }

    cat_record = db["categories"][cat_name]
    candidates = []
    used_mode = mode

    if mode == "websearch":
        print("    • поиск в интернете...", flush=True)
        candidates = search_market_alternatives(client, model_name, service_text)
        if not candidates:
            print("    ⚠ веб-поиск не дал результатов, fallback на synthetic", flush=True)
            used_mode = "synthetic"

    if used_mode == "synthetic":
        if not cat_record.get("companies"):
            print(f"    • генерация {market_size} компаний для категории '{cat_name}'...", flush=True)
            cat_record["companies"] = generate_synthetic_market(
                client, model_name, cat_name, cat_record["description"],
                cat_record["typical_price_min_kzt"], cat_record["typical_price_max_kzt"],
                market_size
            )
            save_market_database(db, db_path)
        candidates = cat_record.get("companies", [])

    if not candidates:
        return make_result(contract, cat_name, note="Нет кандидатов для сравнения")

    print(f"    • фильтрация релевантных (кандидатов: {len(candidates)})...", flush=True)
    relevant = filter_relevant_matches(client, model_name, service_text, contract_price, candidates)

    for r in relevant:
        r["market_source"] = used_mode

    if not relevant:
        return make_result(contract, cat_name, note=f"Из {len(candidates)} ни один не подошел")

    if len(relevant) < MIN_ALTERNATIVES:
        print(f"    ⚠ найдено мало релевантных альтернатив: {len(relevant)} (желательно от {MIN_ALTERNATIVES})",
              flush=True)

    relevant.sort(key=lambda x: float(x.get("price_kzt", float('inf'))))
    cheaper = [r for r in relevant if float(r.get("price_kzt", 0)) < contract_price]

    savings = 0.0
    if cheaper:
        best_price = float(cheaper[0].get("price_kzt", 0))
        savings = contract_price - best_price

    final_alts = relevant[:top_n]
    for alt in final_alts:
        alt_price = float(alt.get("price_kzt", 0))
        diff = contract_price - alt_price
        diff_pct = (diff / contract_price * 100) if contract_price > 0 else 0
        alt["price_diff_kzt"] = diff
        alt["price_diff_pct"] = round(diff_pct, 1)

    print(f"    ✔ категория: {cat_name} | релевантных: {len(relevant)} | экономия: {fmt_kzt(savings)} тг",
          flush=True)
    return make_result(contract, cat_name, final_alts, savings,
                       note=f"Кандидатов проверено: {len(candidates)}")

# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 2: поиск альтернатив (OpenAI API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="contracts.json", help="Входной JSON из Этапа 1")
    parser.add_argument("--output", default="alternatives.json", help="Выходной JSON")
    parser.add_argument("--market-db", default="market_database.json", help="Файл-кэш базы рынка")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Модель OpenAI (например gpt-4o)")
    parser.add_argument("--mode", choices=["synthetic", "websearch"], default="synthetic",
                        help="Режим поиска рынка")
    parser.add_argument("--market-size", type=int, default=DEFAULT_MARKET_SIZE,
                        help="Сколько компаний генерировать на категорию")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help="Сколько лучших альтернатив оставлять в отчете")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Пауза между договорами в секундах")
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ОШИБКА: не задана переменная окружения OPENAI_API_KEY.\n", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ОШИБКА: файл не найден: {input_path} (сначала запустите extract_stage1.py)",
              file=sys.stderr)
        return 1
    with open(input_path, encoding="utf-8") as f:
        contracts = json.load(f)
    if not contracts:
        print("ОШИБКА: в contracts.json нет записей", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    db_path = Path(args.market_db)
    db = load_market_database(db_path)

    client = OpenAI(api_key=api_key)

    total = len(contracts)
    print(f"Модель: {args.model}. Режим: {args.mode}. Договоров: {total}. "
          f"База рынка: {db_path} ({len(db['categories'])} категорий)\n")

    results = []
    try:
        for idx, contract in enumerate(contracts, start=1):
            cid = contract.get("contract_number") or contract.get("source_file", "?")
            print(f"[{idx}/{total}] Договор {cid} — {contract.get('supplier') or '—'}, "
                  f"{int(contract.get('amount_kzt') or 0):,} тг", flush=True)
            try:
                rec = process_contract(client, args.model, contract, db, db_path, args.mode,
                                       args.market_size, args.top)
            except Exception as exc:
                err = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(f"    ✖ ошибка: {err}", flush=True)
                rec = make_result(contract, note=f"ошибка обработки: {err}")
            results.append(rec)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if idx < total and args.pause > 0:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nПрервано пользователем — сохраняю обработанные договоры.", flush=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    with_alts = sum(1 for r in results if r.get("alternatives"))
    total_savings = sum(r.get("potential_savings_kzt", 0) for r in results)
    failed = sum(1 for r in results if "ошибка обработки:" in r.get("note", ""))
    print(f"\nГотово: обработано {len(results)} из {total}; с альтернативами: {with_alts}; "
          f"суммарная потенциальная экономия: {total_savings:,} тг")
    print(f"Категорий в базе рынка: {len(db['categories'])} → {db_path}")
    print(f"Результат: {output_path}")
    
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
