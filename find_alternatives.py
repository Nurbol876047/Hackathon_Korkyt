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
  synthetic  (по умолчанию) — для каждой категории услуг Gemini один раз генерирует
                              15–20 вымышленных компаний-конкурентов с реалистичным
                              разбросом цен; база сохраняется в market_database.json.
  websearch                 — поиск реальных компаний в интернете. Поисковый запрос
                              формируется через Gemini, сам веб-поиск — TODO-заглушка
                              (см. search_market_alternatives). Если поиск ничего не
                              вернул, договор обрабатывается в режиме synthetic,
                              pipeline не блокируется.

Запуск:
    export GEMINI_API_KEY="ваш_ключ"
    python find_alternatives.py --input contracts.json --output alternatives.json --mode synthetic
"""

import argparse
import difflib
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# google-generativeai помечена Google как устаревшая, но работает — глушим предупреждение
warnings.filterwarnings("ignore", category=FutureWarning, message=r"(?s).*google\.generativeai")

import google.generativeai as genai              # noqa: E402
from google.generativeai import protos           # noqa: E402
from google.api_core import exceptions as gexc   # noqa: E402


# ---------------------------------------------------------------------------
# Настройки (те же, что в Этапе 1)
# ---------------------------------------------------------------------------

# gemini-2.0-flash отключена Google (API возвращает 404 и рекомендует gemini-3.6-flash).
DEFAULT_MODEL = "gemini-3.6-flash"

MAX_RETRIES = 2            # повторных попыток после первой неудачной (итого 3 запроса)
RETRY_BASE_DELAY = 5       # секунд; удваивается с каждой повторной попыткой
RATE_LIMIT_DELAY = 30      # секунд ожидания при ошибке 429 (превышена квота)
REQUEST_TIMEOUT = 180      # секунд на один запрос

DEFAULT_MARKET_SIZE = 18   # сколько компаний генерировать на категорию (ТЗ: 15–20)
DEFAULT_TOP_N = 5          # сколько альтернатив оставлять в выводе (ТЗ: 3–5)
MIN_ALTERNATIVES = 3       # если релевантных меньше — предупреждаем в консоли

SYSTEM_PROMPT = """Ты — аналитик рынка IT-услуг, оборудования и программного обеспечения
Республики Казахстан. Помогаешь аудитору государственных закупок сравнивать цену закупки
с рыночными альтернативами. Отвечай только JSON по заданной схеме, на русском языке.
Все суммы — в тенге (KZT), только числа."""


# ---------------------------------------------------------------------------
# Общий помощник: запрос к Gemini со structured output и повторными попытками
# ---------------------------------------------------------------------------

def ask_json(model: genai.GenerativeModel, prompt: str, schema: protos.Schema,
             temperature: float = 0.0):
    """
    Отправляет prompt в модель, требуя ответ строго по JSON-схеме (response_schema).
    При ошибке — MAX_RETRIES повторных попыток с паузой; если всё равно не удалось —
    исключение поднимается наверх (вызывающий код решает, что делать с договором).
    """
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
            print(f"    ⚠ попытка {attempt}/{total_attempts} не удалась: "
                  f"{type(exc).__name__}: {str(exc)[:150]} — повтор через {delay} с", flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# JSON-схемы ответов
# ---------------------------------------------------------------------------

S, T = protos.Schema, protos.Type


def schema_category() -> protos.Schema:
    """Ответ классификатора: категория услуги + типичный ценовой диапазон по рынку РК."""
    return S(
        type=T.OBJECT,
        properties={
            "category": S(type=T.STRING,
                          description="Короткое название категории услуги (2–6 слов, нижний регистр, "
                                      "именительный падеж), например 'разработка мобильного приложения'."),
            "matched_existing": S(type=T.BOOLEAN,
                                  description="true, если выбрана одна из уже существующих категорий."),
            "category_description": S(type=T.STRING,
                                      description="Что обычно входит в услугу этой категории (1–2 предложения)."),
            "typical_price_min_kzt": S(type=T.NUMBER,
                                       description="Нижняя граница типичной рыночной цены в Казахстане, тенге."),
            "typical_price_max_kzt": S(type=T.NUMBER,
                                       description="Верхняя граница типичной рыночной цены в Казахстане, тенге."),
        },
        required=["category", "matched_existing", "category_description",
                  "typical_price_min_kzt", "typical_price_max_kzt"],
    )


def schema_market() -> protos.Schema:
    """Ответ генератора рынка: массив вымышленных компаний с ценами."""
    return S(
        type=T.ARRAY,
        items=S(
            type=T.OBJECT,
            properties={
                "company": S(type=T.STRING,
                             description="Вымышленное название с ОПФ, например 'ТОО «Digital Dala»'."),
                "city": S(type=T.STRING, description="Город Казахстана."),
                "company_size": S(type=T.STRING, format_="enum", enum=["малая", "средняя", "крупная"]),
                "offer": S(type=T.STRING,
                           description="Что именно предлагает компания и в каком объёме (1 предложение)."),
                "tech_stack": S(type=T.STRING, description="Технологии/платформы через запятую."),
                "price_kzt": S(type=T.NUMBER, description="Цена предложения в тенге, число."),
            },
            required=["company", "city", "company_size", "offer", "tech_stack", "price_kzt"],
        ),
    )


def schema_relevance() -> protos.Schema:
    """Ответ фильтра релевантности: вердикт по каждому кандидату."""
    return S(
        type=T.ARRAY,
        items=S(
            type=T.OBJECT,
            properties={
                "candidate_id": S(type=T.INTEGER, description="Номер кандидата из списка."),
                "is_relevant": S(type=T.BOOLEAN,
                                 description="true, если кандидат реально может выполнить ту же услугу."),
                "relevance_reason": S(type=T.STRING,
                                      description="Краткое обоснование (1 предложение), почему подходит / не подходит."),
            },
            required=["candidate_id", "is_relevant", "relevance_reason"],
        ),
    )


def schema_search_query() -> protos.Schema:
    """Ответ генератора поискового запроса (режим websearch)."""
    return S(
        type=T.OBJECT,
        properties={
            "query": S(type=T.STRING, description="Короткий поисковый запрос (3–8 слов)."),
        },
        required=["query"],
    )


# ---------------------------------------------------------------------------
# База рынка (market_database.json)
# ---------------------------------------------------------------------------

def load_market_database(path: Path) -> dict:
    """Загружает базу рынка или создаёт пустую структуру."""
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
    """
    Ищет категорию среди уже существующих: сначала точное совпадение (без учёта
    регистра), затем близкое по написанию — чтобы 'разработка мобильного приложения'
    и 'разработка мобильных приложений' не плодили две базы.
    """
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

def classify_service_category(model: genai.GenerativeModel, service_text: str,
                              tech_stack: str, existing_categories: list) -> dict:
    """
    Определяет категорию услуги договора. Если подходящая категория уже есть в базе —
    модель обязана вернуть её ровно в том же написании (чтобы переиспользовать рынок).
    Цену закупки модели НЕ показываем: ценовой диапазон категории должен отражать
    рынок, а не подстраиваться под договор.
    """
    existing_block = ("\n".join(f"- {c}" for c in existing_categories)
                      if existing_categories else "(пока нет)")
    prompt = f"""Определи категорию услуги для закупки.

Описание предмета закупки: {service_text}
Упомянутые технологии: {tech_stack or "не указаны"}

Уже существующие категории:
{existing_block}

Правила:
- Если одна из существующих категорий подходит по сути (тот же тип результата и сопоставимый
  масштаб) — верни её РОВНО в том же написании и matched_existing = true.
- Иначе придумай новую короткую категорию (2–6 слов, нижний регистр) и matched_existing = false.
- typical_price_min_kzt / typical_price_max_kzt — реалистичный диапазон цен на рынке Казахстана
  за услугу такого типа и масштаба (по описанию), в тенге. Учитывай объём работ из описания."""
    result = ask_json(model, prompt, schema_category())
    result["category"] = str(result.get("category", "")).strip().lower() or "прочие услуги"
    return result


# ---------------------------------------------------------------------------
# Шаг 2 (режим synthetic). Генерация синтетического рынка для категории
# ---------------------------------------------------------------------------

def generate_synthetic_market(model: genai.GenerativeModel, service_category: str,
                              category_description: str, price_min: float, price_max: float,
                              n: int = DEFAULT_MARKET_SIZE) -> list:
    """
    Через Gemini генерирует n (15–20) правдоподобных вымышленных компаний-конкурентов
    для категории услуг. Цены разбросаны вокруг заданного рыночного диапазона:
    большинство внутри, часть ниже (небольшие региональные студии / ИП), часть выше
    (крупные интеграторы). Названия вымышленные — реальные компании не используются.
    """
    n = max(15, min(20, n))
    prompt = f"""Сгенерируй синтетический рынок для категории услуг: «{service_category}».
Что обычно входит в услугу: {category_description}

Нужно ровно {n} ВЫМЫШЛЕННЫХ компаний Казахстана (ТОО / ИП), которые предлагают такую услугу.

Требования к ценам (в тенге):
- Ориентир рыночного диапазона: от {int(price_min):,} до {int(price_max):,} тг.
- Примерно 65% предложений — внутри диапазона, ~15% — ниже (небольшие студии, ИП,
  региональные компании), ~20% — выше (крупные интеграторы, компании с большим штатом).
- Цены разные у всех компаний, округлённые как в реальных коммерческих предложениях
  (до тысяч тенге). Цена должна логично соответствовать размеру компании и объёму offer.

Требования к компаниям:
- Названия вымышленные и не совпадают с реальными компаниями.
- Города — разные города Казахстана (Астана, Алматы, Шымкент, Актобе, Караганда и др.).
- offer — конкретно что делает компания и в каком объёме, сопоставимо с категорией.
- tech_stack — реалистичные технологии/платформы для этой категории.""".replace(",", " ")
    # (replace убирает разделители тысяч из чисел в промпте, чтобы не путать модель)

    companies = ask_json(model, prompt, schema_market(), temperature=0.8)

    # Приводим к строгим типам, отбрасываем мусор
    cleaned = []
    seen = set()
    for c in companies:
        name = str(c.get("company", "")).strip()
        try:
            price = float(c.get("price_kzt", 0))
        except (TypeError, ValueError):
            continue
        if not name or price <= 0 or name.lower() in seen:
            continue
        seen.add(name.lower())
        cleaned.append({
            "company": name,
            "city": str(c.get("city", "")).strip(),
            "company_size": str(c.get("company_size", "")).strip(),
            "offer": str(c.get("offer", "")).strip(),
            "tech_stack": str(c.get("tech_stack", "")).strip(),
            "price_kzt": int(round(price)),
        })
    return cleaned


def get_market_for_category(model, db: dict, db_path: Path, category_info: dict,
                            market_size: int) -> tuple[str, list, bool]:
    """
    Возвращает (имя_категории, список_компаний, сгенерировано_сейчас).
    Если категория уже есть в базе — берём из кэша, иначе генерируем и сразу сохраняем.
    """
    existing = list(db["categories"].keys())
    category = find_existing_category(category_info["category"], existing) or category_info["category"]

    if category in db["categories"]:
        return category, db["categories"][category]["companies"], False

    companies = generate_synthetic_market(
        model, category, category_info["category_description"],
        category_info["typical_price_min_kzt"], category_info["typical_price_max_kzt"],
        n=market_size,
    )
    db["categories"][category] = {
        "description": category_info["category_description"],
        "price_range_kzt": {
            "min": int(category_info["typical_price_min_kzt"]),
            "max": int(category_info["typical_price_max_kzt"]),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "companies": companies,
    }
    save_market_database(db, db_path)
    return category, companies, True


# ---------------------------------------------------------------------------
# Шаг 2 (режим websearch). Поиск реальных альтернатив в интернете
# ---------------------------------------------------------------------------

def build_search_query(model: genai.GenerativeModel, service_text: str) -> str:
    """Через Gemini формирует короткий поисковый запрос из описания услуги."""
    prompt = f"""Сформируй короткий поисковый запрос (3–8 слов, на русском) для поиска компаний
в Казахстане, которые оказывают такую услугу: {service_text}
Запрос должен содержать суть услуги и слово «Казахстан» или город, без лишних деталей."""
    return str(ask_json(model, prompt, schema_search_query()).get("query", "")).strip()


def search_market_alternatives(service_text: str, query: str) -> list:
    """
    TODO: реальный веб-поиск компаний/продуктов по категории услуги.

    Интерфейс: принимает описание услуги и готовый поисковый запрос, возвращает список
    кандидатов в том же формате, что и синтетический рынок:
        [{"company": str, "city": str, "company_size": str, "offer": str,
          "tech_stack": str, "price_kzt": int, "source_url": str}, ...]
    Пустой список означает «поиск недоступен / ничего не найдено» — тогда pipeline
    автоматически откатывается на синтетический рынок.

    Примечание: Google Search grounding через устаревший SDK google-generativeai
    фактически не выполняется (grounding_metadata пустой) — для реального поиска
    нужен google-genai (tools=[{"google_search": {}}]) или внешний поисковый API
    (SerpAPI / Tavily) с последующей структуризацией результатов через Gemini.
    """
    _ = (service_text, query)
    return []


# ---------------------------------------------------------------------------
# Шаг 3. Фильтр релевантности кандидатов
# ---------------------------------------------------------------------------

def filter_relevant_matches(model: genai.GenerativeModel, service_text: str,
                            candidates: list) -> list:
    """
    Через Gemini отбирает кандидатов, которые реально делают ТО ЖЕ САМОЕ (тот же тип
    результата, сопоставимый объём), а не просто похожи по названию/категории.
    Цены кандидатов модели не показываем, чтобы решение о релевантности не зависело от цены.
    Возвращает копии релевантных кандидатов с полем relevance_reason.
    """
    if not candidates:
        return []
    listing = "\n".join(
        f"{i}. {c['company']} ({c.get('city', '—')}, {c.get('company_size', '—')} компания): "
        f"{c.get('offer', '')}. Технологии: {c.get('tech_stack', '') or '—'}"
        for i, c in enumerate(candidates, start=1)
    )
    prompt = f"""Предмет закупки по договору: {service_text}

Кандидаты-альтернативы:
{listing}

Для КАЖДОГО кандидата (по номеру) реши, может ли он выполнить ту же самую услугу:
тот же тип результата (например, разработка ≠ лицензия ≠ поставка оборудования),
сопоставимый объём и масштаб, подходящие технологии. Похожего названия недостаточно.
relevance_reason — одно предложение на русском: почему подходит или почему нет."""
    verdicts = ask_json(model, prompt, schema_relevance())

    relevant = []
    for v in verdicts:
        try:
            idx = int(v.get("candidate_id", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and bool(v.get("is_relevant")):
            item = dict(candidates[idx])
            item["relevance_reason"] = str(v.get("relevance_reason", "")).strip()
            relevant.append(item)
    return relevant


# ---------------------------------------------------------------------------
# Шаг 4. Сравнение цен и ранжирование
# ---------------------------------------------------------------------------

def compute_price_diffs(target_price: float, alternatives: list) -> list:
    """
    Добавляет каждому кандидату price_diff_pct — на сколько процентов альтернатива
    дешевле (+) или дороже (−) фактической цены закупки:
        price_diff_pct = (target_price - alt_price) / target_price * 100
    """
    result = []
    for alt in alternatives:
        item = dict(alt)
        item["price_diff_pct"] = round((target_price - alt["price_kzt"]) / target_price * 100, 1)
        result.append(item)
    return result


def rank_alternatives_by_price(alternatives: list) -> list:
    """Сортирует альтернативы от самой дешёвой к самой дорогой."""
    return sorted(alternatives, key=lambda a: a["price_kzt"])


def build_alternative_view(alt: dict) -> dict:
    """Формат альтернативы в выходном JSON (фиксированный порядок полей)."""
    return {
        "company": alt["company"],
        "city": alt.get("city", ""),
        "price_kzt": alt["price_kzt"],
        "price_diff_pct": alt["price_diff_pct"],
        "relevance_reason": alt.get("relevance_reason", ""),
        "offer": alt.get("offer", ""),
    }


def make_result(contract: dict, **extra) -> dict:
    """Каркас выходной записи по договору; недостающие поля — значения по умолчанию."""
    rec = {
        # contract_id — номер договора, а если его не удалось извлечь — имя файла.
        # source_file всегда сохраняем как надёжный ключ для стыковки с другими этапами.
        "contract_id": contract.get("contract_number") or contract.get("source_file", ""),
        "source_file": contract.get("source_file", ""),
        "current_supplier": contract.get("supplier", ""),
        "current_price_kzt": contract.get("amount_kzt", 0),
        "service_text": contract.get("service_text", ""),
        "service_category": "",
        "market_source": "",
        "candidates_checked": 0,
        "alternatives": [],
        "best_alternative": None,
        "potential_savings_kzt": 0,
        "potential_savings_pct": 0.0,
        "note": "",
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# Обработка одного договора
# ---------------------------------------------------------------------------

def process_contract(model, contract: dict, db: dict, db_path: Path, mode: str,
                     market_size: int, top_n: int) -> dict:
    service_text = (contract.get("service_text") or "").strip()
    try:
        target_price = float(contract.get("amount_kzt") or 0)
    except (TypeError, ValueError):
        target_price = 0.0

    # Договоры без суммы или описания сравнивать не с чем — оставляем запись с пометкой
    if target_price <= 0 or not service_text:
        print("    – пропуск: нет суммы или описания услуги", flush=True)
        return make_result(contract, note="нет данных для сравнения (сумма или описание услуги отсутствуют)")

    # 1. Категория услуги
    category_info = classify_service_category(
        model, service_text, contract.get("tech_stack_mentioned", ""), list(db["categories"].keys())
    )

    # 2. Кандидаты: веб-поиск (если доступен) или синтетический рынок
    candidates, market_source, category = [], "", category_info["category"]
    if mode == "websearch":
        query = build_search_query(model, service_text)
        print(f"    поисковый запрос: «{query}»", flush=True)
        candidates = search_market_alternatives(service_text, query)
        if candidates:
            market_source = "websearch"
        else:
            print("    веб-поиск недоступен/пуст — используем синтетический рынок", flush=True)
    if not candidates:
        category, candidates, generated = get_market_for_category(
            model, db, db_path, category_info, market_size
        )
        market_source = "synthetic"
        print(f"    категория: «{category}» — {len(candidates)} компаний "
              f"({'сгенерировано' if generated else 'из базы'})", flush=True)

    # 3. Релевантность → 4. цены → ранжирование → экономия
    relevant = filter_relevant_matches(model, service_text, candidates)
    ranked = rank_alternatives_by_price(compute_price_diffs(target_price, relevant))
    top = [build_alternative_view(a) for a in ranked[:top_n]]

    best = top[0] if top else None
    savings = int(round(max(0.0, target_price - best["price_kzt"]))) if best else 0
    savings_pct = round(savings / target_price * 100, 1) if best else 0.0

    note = ""
    if not top:
        note = "релевантных альтернатив не найдено"
    elif len(top) < MIN_ALTERNATIVES:
        note = f"найдено только {len(top)} релевантных альтернатив(ы)"

    if best:
        print(f"    релевантных: {len(relevant)} из {len(candidates)} | лучшая: {best['company']} — "
              f"{best['price_kzt']:,} тг ({best['price_diff_pct']:+.1f}%) | "
              f"экономия: {savings:,} тг", flush=True)
    else:
        print(f"    релевантных: 0 из {len(candidates)} — экономия не рассчитана", flush=True)

    return make_result(
        contract,
        service_category=category,
        market_source=market_source,
        candidates_checked=len(candidates),
        alternatives=top,
        best_alternative=best,
        potential_savings_kzt=savings,
        potential_savings_pct=savings_pct,
        note=note,
    )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 2: поиск альтернативных поставщиков и расчёт потенциальной экономии.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="contracts.json", help="contracts.json из Этапа 1")
    parser.add_argument("--output", default="alternatives.json", help="выходной JSON")
    parser.add_argument("--mode", choices=["synthetic", "websearch"], default="synthetic",
                        help="источник кандидатов")
    parser.add_argument("--market-db", default="market_database.json",
                        help="файл синтетической базы рынка (кэш между запусками)")
    parser.add_argument("--market-size", type=int, default=DEFAULT_MARKET_SIZE,
                        help="компаний на категорию при генерации (15–20)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help="сколько альтернатив оставлять в выводе")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели Gemini")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="пауза между договорами, сек (лимит запросов в минуту)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ОШИБКА: не задана переменная окружения GEMINI_API_KEY.\n"
              "Получите ключ на https://aistudio.google.com/apikey и выполните:\n"
              '    export GEMINI_API_KEY="ваш_ключ"', file=sys.stderr)
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

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model, system_instruction=SYSTEM_PROMPT)

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
                rec = process_contract(model, contract, db, db_path, args.mode,
                                       args.market_size, args.top)
            except Exception as exc:  # noqa: BLE001 — договор не должен выпасть из выборки
                err = f"{type(exc).__name__}: {str(exc)[:300]}"
                print(f"    ✖ ошибка: {err}", flush=True)
                rec = make_result(contract, note=f"ошибка обработки: {err}")
            results.append(rec)

            # Сохраняем после каждого договора — при обрыве прогресс не теряется
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if idx < total and args.pause > 0:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nПрервано пользователем — сохраняю обработанные договоры.", flush=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Итоги
    with_alts = sum(1 for r in results if r["alternatives"])
    total_savings = sum(r["potential_savings_kzt"] for r in results)
    print(f"\nГотово: обработано {len(results)} из {total}; с альтернативами: {with_alts}; "
          f"суммарная потенциальная экономия: {total_savings:,} тг")
    print(f"Категорий в базе рынка: {len(db['categories'])} → {db_path}")
    print(f"Результат: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
