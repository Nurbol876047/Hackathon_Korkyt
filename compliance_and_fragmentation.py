#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 4 из 5 — два модуля:
  (А) проверка соответствия закупленного продукта техническому заданию;
  (Б) детекция искусственного дробления закупок.

Вход:   contracts.json  — Этап 1 (extract_stage1.py)
Выход:  <output-dir>/tor_compliance.json          — по одной записи на договор
        <output-dir>/fragmentation_clusters.json  — список кластеров подозрительно
                                                    похожих договоров (2+ договора)

ВАЖНО: оба модуля выдают ИНДИКАТОР РИСКА для проверки человеком, а не вердикт.
В формулировках намеренно нет слов «нарушение» и «мошенничество».

Запуск:
    export GEMINI_API_KEY="ваш_ключ"
    python compliance_and_fragmentation.py --input contracts.json --output-dir ./results
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, message=r"(?s).*google\.generativeai")

import google.generativeai as genai              # noqa: E402
from google.generativeai import protos           # noqa: E402
from google.api_core import exceptions as gexc   # noqa: E402


# ---------------------------------------------------------------------------
# Настройки (пороги вынесены сюда, чтобы легко подстроить на демо)
# ---------------------------------------------------------------------------

# Модели. gemini-2.0-flash и text-embedding-004 отключены Google → актуальные замены
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_EMBED_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 50            # текстов в одном запросе embed_content

# --- Модуль А: соответствие ТЗ ---
COMPLIANCE_OK_SCORE = 80         # consistency_score >= этого — "соответствует"
COMPLIANCE_CHECK_SCORE = 50      # >= этого — "требует проверки", ниже — "высокий риск несоответствия"

COMPLIANCE_OK = "соответствует"
COMPLIANCE_CHECK = "требует проверки"
COMPLIANCE_RISK = "высокий риск несоответствия"
COMPLIANCE_NO_DATA = "недостаточно данных для проверки"

# --- Модуль Б: дробление закупок ---
SIMILARITY_THRESHOLD = 0.75      # cosine similarity текстов услуг, выше — «похожие»
MAX_DAYS_BETWEEN = 30            # максимальная разница дат между договорами пары, дней
NO_TENDER_THRESHOLD_KZT = 3_000_000   # условная граница закупки без конкурса (для демо)

# Веса компонентов suspicion_score (в сумме 100)
W_SIMILARITY = 40                # насколько похожи тексты услуг
W_DATES = 30                     # насколько близки даты
W_THRESHOLD = 30                 # обошли бы порог конкурса, если объединить

SUSPICION_HIGH = 70              # suspicion_score >= этого — "высокий"
SUSPICION_MEDIUM = 40            # >= этого — "средний", ниже — "низкий"

# Параметры запросов к API (как в Этапах 1–3)
MAX_RETRIES = 2
RETRY_BASE_DELAY = 5
RATE_LIMIT_DELAY = 30
REQUEST_TIMEOUT = 180

SYSTEM_PROMPT = """Ты — технический эксперт, помогающий аудитору государственных закупок
Республики Казахстан. Оцениваешь документы объективно и осторожно: указываешь только на то,
что реально есть в тексте, не домысливаешь. Твои оценки — индикатор для проверки человеком,
а не вердикт, поэтому не используй слова «нарушение», «мошенничество», «обман».
Отвечай только JSON по заданной схеме, на русском языке."""


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------

def with_retries(func, what: str):
    """Выполняет func() с повторными попытками; последняя ошибка поднимается наверх."""
    total_attempts = MAX_RETRIES + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt == total_attempts:
                raise
            delay = (RATE_LIMIT_DELAY if isinstance(exc, gexc.ResourceExhausted)
                     else RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            print(f"    ⚠ {what}: попытка {attempt}/{total_attempts} не удалась: "
                  f"{type(exc).__name__}: {str(exc)[:150]} — повтор через {delay} с", flush=True)
            time.sleep(delay)


def ask_json(model: genai.GenerativeModel, prompt: str, schema: protos.Schema,
             temperature: float = 0.0):
    """Запрос к Gemini со structured output (response_schema) и повторными попытками."""
    def _call():
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
    return with_retries(_call, "Gemini")


def fmt_kzt(value) -> str:
    """Сумма с пробелами между тысячами: 3000000 → '3 000 000'."""
    try:
        return f"{int(round(float(value or 0))):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "0"


def contract_id_of(contract: dict) -> str:
    """Идентификатор договора: номер договора, а если его нет — имя файла (как в Этапах 2–3)."""
    return contract.get("contract_number") or contract.get("source_file", "")


def parse_date(value: str) -> date | None:
    """contract_date из Этапа 1 в формате YYYY-MM-DD; иначе None."""
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return None


S, T = protos.Schema, protos.Type


# ===========================================================================
# МОДУЛЬ А. Проверка соответствия техническому заданию
# ===========================================================================

def schema_consistency() -> protos.Schema:
    return S(
        type=T.OBJECT,
        properties={
            "consistency_score": S(type=T.INTEGER,
                                   description="0–100: насколько описание результата соответствует "
                                               "заявленным технологиям (100 — полностью)."),
            "flagged_phrases": S(type=T.ARRAY, items=S(type=T.STRING),
                                 description="Дословные фразы из переданного текста, вызвавшие "
                                             "сомнение. Пусто, если сомнений нет."),
            "explanation": S(type=T.STRING,
                             description="2–3 предложения на русском: почему выставлен такой балл."),
        },
        required=["consistency_score", "flagged_phrases", "explanation"],
    )


def tor_consistency_check(model: genai.GenerativeModel, tech_stack_mentioned: str,
                          service_text: str) -> dict:
    """
    Через Gemini оценивает, есть ли противоречия или расплывчатые формулировки между
    заявленными технологиями (tech_stack_mentioned) и описанием результата (service_text).
    Примеры несоответствий: заявлено «нативное приложение iOS/Android», а в описании —
    общие фразы без конкретики о платформе; или маркеры веб-технологий там, где заявлена
    нативная разработка.
    """
    prompt = f"""Проверь согласованность заявленных технологий и описания результата закупки.

ЗАЯВЛЕННЫЕ ТЕХНОЛОГИИ / ПЛАТФОРМА (из текста договора):
{tech_stack_mentioned}

ОПИСАНИЕ ПРЕДМЕТА ЗАКУПКИ / РЕЗУЛЬТАТА:
{service_text}

Оцени:
1. Есть ли прямые противоречия (например, заявлена нативная разработка iOS/Android, а описан
   веб-сайт или «веб-обёртка»; заявлена одна платформа, а описана другая).
2. Есть ли расплывчатые формулировки, за которыми может скрываться несоответствие
   («современные технологии», «на базе передовых решений», «мобильное решение» без платформы).
3. Соответствует ли объём/тип результата заявленному стеку.

Правила:
- consistency_score: 90–100 — заявленное и описанное согласованы; 60–89 — есть неточности
  или расплывчатость; ниже 60 — есть признаки противоречия.
- flagged_phrases — ТОЛЬКО дословные фрагменты из переданного выше текста (копируй как есть).
  Если сомнений нет — пустой список.
- explanation — кратко и нейтрально; это индикатор для проверки человеком."""
    result = ask_json(model, prompt, schema_consistency())

    # Приводим к строгим типам
    try:
        score = int(result.get("consistency_score", 0))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    # Оставляем только фразы, которые реально есть в тексте — модель не должна цитировать
    # то, чего в документе нет
    haystack = _squash(f"{tech_stack_mentioned} {service_text}")
    phrases, dropped = [], 0
    for p in result.get("flagged_phrases", []) or []:
        p = str(p).strip().strip("«»\"'")
        if p and _squash(p) in haystack:
            phrases.append(p)
        elif p:
            dropped += 1
    if dropped:
        print(f"    (отброшено {dropped} фраз(ы), отсутствующих в тексте)", flush=True)

    return {
        "consistency_score": score,
        "flagged_phrases": phrases,
        "explanation": str(result.get("explanation", "")).strip(),
    }


def _squash(text: str) -> str:
    """Нижний регистр, одинаковые кавычки, один пробел — для проверки «фраза есть в тексте»."""
    text = re.sub(r"[«»\"„“”]", "", text.lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", text).strip()


def compliance_level(score: int) -> str:
    if score >= COMPLIANCE_OK_SCORE:
        return COMPLIANCE_OK
    if score >= COMPLIANCE_CHECK_SCORE:
        return COMPLIANCE_CHECK
    return COMPLIANCE_RISK


def run_tor_compliance(model: genai.GenerativeModel, contracts: list) -> list:
    """Модуль А по всем договорам. Договоры без tech_stack_mentioned не оцениваются."""
    results = []
    total = len(contracts)
    for i, contract in enumerate(contracts, start=1):
        cid = contract_id_of(contract)
        tech = (contract.get("tech_stack_mentioned") or "").strip()
        service_text = (contract.get("service_text") or "").strip()
        rec = {
            "source_file": contract.get("source_file", ""),
            "contract_id": cid,
            "supplier": contract.get("supplier", ""),
            "tech_stack_mentioned": tech,
            "service_text": service_text,
            "status": "",
            "consistency_score": None,
            "compliance_level": "",
            "flagged_phrases": [],
            "explanation": "",
        }
        print(f"  [{i}/{total}] {cid}", end=": ", flush=True)

        # Без явно заявленных технологий проверять нечего — не выдумываем результат
        if not tech or not service_text:
            rec["status"] = COMPLIANCE_NO_DATA
            rec["compliance_level"] = COMPLIANCE_NO_DATA
            rec["explanation"] = ("в документе не заявлены технологии/платформа"
                                  if not tech else "нет описания предмета закупки")
            print(COMPLIANCE_NO_DATA, flush=True)
            results.append(rec)
            continue

        try:
            check = tor_consistency_check(model, tech, service_text)
            rec.update(check)
            rec["status"] = "проверено"
            rec["compliance_level"] = compliance_level(check["consistency_score"])
            print(f"score {check['consistency_score']} — {rec['compliance_level']}; "
                  f"фраз: {len(check['flagged_phrases'])}", flush=True)
        except Exception as exc:  # noqa: BLE001 — договор остаётся в выводе с пометкой
            rec["status"] = "ошибка проверки"
            rec["compliance_level"] = COMPLIANCE_NO_DATA
            rec["explanation"] = f"ошибка API: {type(exc).__name__}: {str(exc)[:200]}"
            print(f"✖ {rec['explanation']}", flush=True)
        results.append(rec)
    return results


# ===========================================================================
# МОДУЛЬ Б. Детекция дробления закупок
# ===========================================================================

def embed_texts(texts: list, embed_model: str) -> np.ndarray:
    """
    Эмбеддинги текстов услуг через Gemini embed_content (пакетами по EMBED_BATCH_SIZE).
    Возвращает матрицу (n, dim) с нормированными строками — тогда cosine similarity
    всех пар считается одним матричным произведением.
    """
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = with_retries(
            lambda: genai.embed_content(model=embed_model, content=batch,
                                        task_type="SEMANTIC_SIMILARITY"),
            "embed_content",
        )
        vectors.extend(response["embedding"])
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(normalized: np.ndarray) -> np.ndarray:
    """Матрица cosine similarity для нормированных векторов (numpy, без sklearn)."""
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


# Полные и сокращённые организационно-правовые формы заказчиков (рус. и каз.):
# «Государственное учреждение "X"» и «ГУ "X"» должны давать один и тот же ключ
_ORG_FULL_RE = re.compile(
    r"(?:республиканское|коммунальное)?\s*государственное\s+(?:коммунальное\s+)?"
    r"(?:казенное\s+)?(?:учреждение|предприятие)"
    r"|(?:коммуналдық\s+)?мемлекеттік\s+(?:мекемесі|мекеме|кәсіпорны|кәсіпорын)",
    re.IGNORECASE)
_ORG_FORMS_RE = re.compile(
    r"\b(?:ГУ|РГУ|КГУ|ГККП|КГП|РГП|РГКП|ММ|КММ|РММ|ТОО|АО|ИП|ЖШС|АҚ)\b", re.IGNORECASE)


def normalize_customer(name: str) -> str:
    """Ключ сравнения заказчиков: без ОПФ, кавычек, регистра и лишних пробелов."""
    text = (name or "").lower().replace("ё", "е")
    text = _ORG_FULL_RE.sub(" ", text)
    text = _ORG_FORMS_RE.sub(" ", text)
    text = re.sub(r"[«»\"„“”'.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class UnionFind:
    """Объединение договоров в кластеры по парным связям (связные компоненты)."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_fragmentation_clusters(contracts: list, embed_model: str) -> tuple[list, dict]:
    """
    Ищет группы договоров с признаками одной раздробленной закупки. Пара договоров
    связывается, если ОДНОВРЕМЕННО:
      - cosine similarity текстов услуг >= SIMILARITY_THRESHOLD,
      - разница contract_date <= MAX_DAYS_BETWEEN дней,
      - один и тот же заказчик (после нормализации названия).
    Связанные пары объединяются в кластеры (связные компоненты); кластеры из 2+ договоров
    возвращаются вместе с метриками. Второй элемент — сводка для консоли.
    """
    # Договоры, пригодные для анализа: есть текст, дата и заказчик
    eligible, skipped = [], []
    for i, c in enumerate(contracts):
        text = (c.get("service_text_normalized") or c.get("service_text") or "").strip()
        d = parse_date(c.get("contract_date", ""))
        cust = normalize_customer(c.get("customer", ""))
        if text and d and cust:
            eligible.append((i, text, d, cust))
        else:
            skipped.append(contract_id_of(c))

    summary = {"eligible": len(eligible), "skipped": skipped, "pairs_similar": 0, "pairs_linked": 0}
    if len(eligible) < 2:
        return [], summary

    print(f"  эмбеддинги: {len(eligible)} текстов через {embed_model}...", flush=True)
    sim = cosine_similarity_matrix(embed_texts([e[1] for e in eligible], embed_model))

    uf = UnionFind(len(eligible))
    pair_info = {}
    for a, b in combinations(range(len(eligible)), 2):
        similarity = float(sim[a, b])
        if similarity < SIMILARITY_THRESHOLD:
            continue
        summary["pairs_similar"] += 1
        days = abs((eligible[a][2] - eligible[b][2]).days)
        if days > MAX_DAYS_BETWEEN or eligible[a][3] != eligible[b][3]:
            continue
        summary["pairs_linked"] += 1
        uf.union(a, b)
        pair_info[(a, b)] = (similarity, days)

    groups: dict = {}
    for k in range(len(eligible)):
        groups.setdefault(uf.find(k), []).append(k)

    clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda k: eligible[k][2])                      # по дате
        member_contracts = [contracts[eligible[k][0]] for k in members]
        pair_sims = [float(sim[a, b]) for a, b in combinations(members, 2)]
        pair_days = [abs((eligible[a][2] - eligible[b][2]).days) for a, b in combinations(members, 2)]
        clusters.append(build_cluster(member_contracts, pair_sims, pair_days))

    clusters.sort(key=lambda c: -c["suspicion_score"])
    for n, c in enumerate(clusters, start=1):
        c["cluster_id"] = f"F{n}"
    return clusters, summary


def build_cluster(member_contracts: list, pair_sims: list, pair_days: list) -> dict:
    """Метрики кластера, suspicion_score и объяснение (без обращения к модели)."""
    amounts = []
    for c in member_contracts:
        try:
            amounts.append(float(c.get("amount_kzt") or 0))
        except (TypeError, ValueError):
            amounts.append(0.0)
    total = sum(amounts)
    avg_sim = float(np.mean(pair_sims))
    avg_days = float(np.mean(pair_days))
    dates = [parse_date(c.get("contract_date", "")) for c in member_contracts]
    span_days = (max(dates) - min(dates)).days

    # --- suspicion_score ---
    # 1) похожесть текстов: порог → 0, полное совпадение → 1
    sim_component = (avg_sim - SIMILARITY_THRESHOLD) / (1.0 - SIMILARITY_THRESHOLD)
    # 2) близость дат: 0 дней → 1, MAX_DAYS_BETWEEN → 0
    date_component = 1.0 - min(avg_days, MAX_DAYS_BETWEEN) / MAX_DAYS_BETWEEN
    # 3) порог конкурса: каждый договор ниже порога, а сумма — выше → полный вес;
    #    сумма выше, но и отдельные договоры выше → половина; сумма ниже порога → 0
    each_below = all(a <= NO_TENDER_THRESHOLD_KZT for a in amounts)
    total_above = total > NO_TENDER_THRESHOLD_KZT
    if total_above and each_below:
        threshold_component = 1.0
    elif total_above:
        threshold_component = 0.5
    else:
        threshold_component = 0.0
    score = int(round(W_SIMILARITY * max(0.0, min(1.0, sim_component))
                      + W_DATES * date_component
                      + W_THRESHOLD * threshold_component))
    level = ("высокий" if score >= SUSPICION_HIGH else
             "средний" if score >= SUSPICION_MEDIUM else "низкий")

    # --- объяснение: только факты и осторожные формулировки ---
    customer = member_contracts[0].get("customer", "") or "—"
    parts = [
        f"{len(member_contracts)} договора(ов) одного заказчика «{customer}» с похожим предметом закупки "
        f"(средняя семантическая похожесть {avg_sim:.2f}) заключены в интервале {span_days} дн. "
        f"({min(dates).isoformat()} — {max(dates).isoformat()}); суммарно {fmt_kzt(total)} ₸."
    ]
    if total_above and each_below:
        parts.append(f"Каждый договор по отдельности не превышает условный порог {fmt_kzt(NO_TENDER_THRESHOLD_KZT)} ₸ "
                     f"для закупки без конкурса, а их сумма превышает его — признаки возможного дробления, "
                     f"требует дополнительной проверки.")
    elif total_above:
        parts.append(f"Суммарно договоры превышают условный порог {fmt_kzt(NO_TENDER_THRESHOLD_KZT)} ₸; "
                     f"часть договоров превышает его и по отдельности — рекомендуется проверить, "
                     f"не является ли это одной закупкой.")
    else:
        parts.append(f"Суммарная стоимость ниже условного порога {fmt_kzt(NO_TENDER_THRESHOLD_KZT)} ₸ — "
                     f"группа отмечена как контекст, риск низкий.")

    return {
        "cluster_id": "",
        "customer": customer,
        "contract_ids": [contract_id_of(c) for c in member_contracts],
        "source_files": [c.get("source_file", "") for c in member_contracts],
        "contracts": [{
            "source_file": c.get("source_file", ""),
            "contract_id": contract_id_of(c),
            "contract_date": c.get("contract_date", ""),
            "supplier": c.get("supplier", ""),
            "amount_kzt": c.get("amount_kzt", 0),
            "service_text": c.get("service_text", ""),
        } for c in member_contracts],
        "contracts_count": len(member_contracts),
        "total_amount_kzt": int(round(total)),
        "avg_similarity": round(avg_sim, 3),
        "date_span_days": span_days,
        "each_below_threshold": each_below,
        "threshold_kzt": NO_TENDER_THRESHOLD_KZT,
        "suspicion_score": score,
        "suspicion_level": level,
        "explanation": " ".join(parts),
        "llm_check": None,
    }


def schema_same_subject() -> protos.Schema:
    return S(
        type=T.OBJECT,
        properties={
            "same_subject": S(type=T.BOOLEAN,
                              description="true, если договоры — один и тот же предмет закупки "
                                          "или части одной закупки."),
            "reason": S(type=T.STRING, description="1–2 предложения на русском, нейтрально."),
        },
        required=["same_subject", "reason"],
    )


def verify_cluster_with_llm(model: genai.GenerativeModel, cluster: dict) -> dict:
    """
    Защита от ложных срабатываний: эмбеддинги могут сблизить «поставка бумаги» и
    «поставка картриджей». Модель подтверждает, что предметы закупок действительно
    одинаковые/взаимодополняющие, а не просто похожие по форме.
    """
    listing = "\n".join(f"{i}. {c['contract_date']}, {fmt_kzt(c['amount_kzt'])} ₸: {c['service_text']}"
                        for i, c in enumerate(cluster["contracts"], start=1))
    prompt = f"""Заказчик «{cluster['customer']}» заключил несколько договоров:
{listing}

Это один и тот же предмет закупки (или части одной работы/поставки, которые логично было бы
закупить одним договором)? Или это явно разные предметы, похожие лишь по форме описания?
Если предметы разные — same_subject = false. Формулируй нейтрально."""
    result = ask_json(model, prompt, schema_same_subject())
    return {"same_subject": bool(result.get("same_subject")),
            "reason": str(result.get("reason", "")).strip()}


# ===========================================================================
# Точка входа
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 4: соответствие ТЗ (А) и детекция дробления закупок (Б).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="contracts.json", help="contracts.json из Этапа 1")
    parser.add_argument("--output-dir", default="./results", help="папка для результатов")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="модель Gemini для текстовой оценки")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="модель эмбеддингов")
    parser.add_argument("--no-llm-check", action="store_true",
                        help="не подтверждать кластеры через Gemini (только эмбеддинги/даты/заказчик)")
    parser.add_argument("--skip-compliance", action="store_true", help="пропустить модуль А")
    parser.add_argument("--skip-fragmentation", action="store_true", help="пропустить модуль Б")
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model, system_instruction=SYSTEM_PROMPT)
    print(f"Модель: {args.model}; эмбеддинги: {args.embed_model}. Договоров: {len(contracts)}\n")

    # --- Модуль А ---
    if not args.skip_compliance:
        print("МОДУЛЬ А: проверка соответствия ТЗ", flush=True)
        compliance = run_tor_compliance(model, contracts)
        path_a = out_dir / "tor_compliance.json"
        with open(path_a, "w", encoding="utf-8") as f:
            json.dump(compliance, f, ensure_ascii=False, indent=2)
        checked = [r for r in compliance if r["status"] == "проверено"]
        levels = {lvl: sum(1 for r in checked if r["compliance_level"] == lvl)
                  for lvl in (COMPLIANCE_OK, COMPLIANCE_CHECK, COMPLIANCE_RISK)}
        print(f"  итог: проверено {len(checked)} из {len(compliance)}; "
              + ", ".join(f"{k}: {v}" for k, v in levels.items()) + f" → {path_a}\n")

    # --- Модуль Б ---
    if not args.skip_fragmentation:
        frag_contracts = list(contracts)
        if len(frag_contracts) == 1:
            base = frag_contracts[0]
            from datetime import timedelta
            base_date = parse_date(base.get("contract_date", "")) or date.today()
            
            mock1 = dict(base)
            mock1["source_file"] = "history_archive_01.pdf"
            mock1["contract_number"] = "MOCK-001"
            mock1["contract_date"] = (base_date - timedelta(days=12)).isoformat()
            mock1["supplier"] = "ТОО «Строй Альянс» (mock)"
            mock1["amount_kzt"] = 2800000
            
            mock2 = dict(base)
            mock2["source_file"] = "history_archive_02.pdf"
            mock2["contract_number"] = "MOCK-002"
            mock2["contract_date"] = (base_date - timedelta(days=25)).isoformat()
            mock2["supplier"] = "ИП «Trade Group» (mock)"
            mock2["amount_kzt"] = 2950000
            
            frag_contracts.extend([mock1, mock2])
            print("  [DEMO] Добавлены 2 моковых исторических договора для демонстрации дробления.", flush=True)

        print("МОДУЛЬ Б: детекция дробления закупок", flush=True)
        print(f"  пороги: similarity ≥ {SIMILARITY_THRESHOLD}, ≤ {MAX_DAYS_BETWEEN} дн., "
              f"один заказчик; условный порог конкурса {fmt_kzt(NO_TENDER_THRESHOLD_KZT)} ₸")
        try:
            clusters, summary = find_fragmentation_clusters(frag_contracts, args.embed_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✖ эмбеддинги недоступны: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            clusters, summary = [], {"eligible": 0, "skipped": [], "pairs_similar": 0, "pairs_linked": 0}
        if summary["skipped"]:
            print(f"  не анализируются (нет даты/заказчика/текста): {', '.join(summary['skipped'])}")
        print(f"  пригодных договоров: {summary['eligible']}; похожих пар: {summary['pairs_similar']}; "
              f"связанных пар (даты + заказчик): {summary['pairs_linked']}; кластеров: {len(clusters)}")

        # Подтверждение кластеров моделью — защита от ложных срабатываний
        confirmed = []
        for cluster in clusters:
            label = f"{cluster['cluster_id']} [{', '.join(cluster['contract_ids'])}]"
            if not args.no_llm_check:
                try:
                    cluster["llm_check"] = verify_cluster_with_llm(model, cluster)
                except Exception as exc:  # noqa: BLE001 — при сбое проверки кластер оставляем
                    cluster["llm_check"] = {"same_subject": True,
                                            "reason": f"проверка недоступна: {type(exc).__name__}"}
                if not cluster["llm_check"]["same_subject"]:
                    print(f"  – {label}: отклонён (разные предметы закупки) — {cluster['llm_check']['reason']}")
                    continue
                cluster["explanation"] += " " + cluster["llm_check"]["reason"]
            confirmed.append(cluster)
            print(f"  • {label}: score {cluster['suspicion_score']} ({cluster['suspicion_level']}), "
                  f"сумма {fmt_kzt(cluster['total_amount_kzt'])} ₸, похожесть {cluster['avg_similarity']}, "
                  f"интервал {cluster['date_span_days']} дн.")
        for n, c in enumerate(confirmed, start=1):
            c["cluster_id"] = f"F{n}"

        path_b = out_dir / "fragmentation_clusters.json"
        with open(path_b, "w", encoding="utf-8") as f:
            json.dump(confirmed, f, ensure_ascii=False, indent=2)
        print(f"  итог: кластеров с признаками дробления: {len(confirmed)} → {path_b}")

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
