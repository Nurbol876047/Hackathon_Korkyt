#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 1 из 5 — извлечение структурированных данных из PDF-файлов договоров
и актов выполненных работ через OpenAI API (GPT-4o/gpt-4o-mini).

PDF конвертируется в изображения с помощью pdf2image (требует poppler)
и отправляется в GPT-4o Vision. Ответ модели ограничен Pydantic-схемой
(structured output / response_format), поэтому парсить текст руками не нужно.

Запуск:
    export OPENAI_API_KEY="ваш_ключ"
    python extract_stage1.py --input ./contracts --output contracts.json
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import base64
import io
from pathlib import Path
from typing import Literal

from openai import OpenAI
from openai import RateLimitError, APIConnectionError, APIError
from pydantic import BaseModel, Field
import pdf2image

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini" # or gpt-4o

MAX_RETRIES = 2
RETRY_BASE_DELAY = 5
RATE_LIMIT_DELAY = 30
REQUEST_TIMEOUT = 300

CONFIDENCE_VALUES = ("high", "medium", "low")

STRING_FIELDS = (
    "supplier",
    "customer",
    "contract_date",
    "lot_number",
    "contract_number",
    "service_text",
    "tech_stack_mentioned",
    "district",
    "ai_assessment",
)

CSV_FIELDS = [
    "source_file",
    "supplier",
    "customer",
    "amount_kzt",
    "contract_date",
    "lot_number",
    "contract_number",
    "service_text",
    "service_text_normalized",
    "tech_stack_mentioned",
    "district",
    "confidence",
    "ai_assessment",
    "error",
]

# ---------------------------------------------------------------------------
# Промпт и JSON-схема ответа (Pydantic)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — ассистент аудитора государственных закупок Республики Казахстан.
Тебе передают изображения страниц PDF-файла: договор о государственных закупках, техническую спецификацию
или акт выполненных работ. Документ может быть на русском и/или казахском языке
и может быть сканом невысокого качества.

Твоя задача — извлечь из документа данные строго по заданной JSON-схеме.

ПРАВИЛА:
1. Заполняй поля ТОЛЬКО тем, что явно присутствует в документе. Ничего не додумывай,
   не восстанавливай по косвенным признакам и не подставляй типичные значения.
2. Если значение поля в документе не найдено — верни пустую строку "" (для amount_kzt — 0).
3. amount_kzt — общая сумма договора в тенге, как указана в документе (итоговая сумма
   с НДС, если она есть). Только число: без пробелов, без "тг" и "тенге". Если сумма
   в документе приведена в тысячах тенге — пересчитай в тенге.
4. contract_date — дата заключения (подписания) договора в формате YYYY-MM-DD.
   Если дата не читается полностью (нет дня, месяца или года) — верни "".
5. supplier — поставщик / подрядчик / исполнитель; customer — заказчик (государственный
   орган, учреждение, акимат, отдел, школа и т.п.). Наименования указывай как в документе,
   вместе с организационно-правовой формой (ТОО, ИП, ГУ, КГУ, ГККП, ЖШС и т.д.).
6. lot_number — номер лота или номер закупки/объявления (например, с портала goszakup.gov.kz).
7. contract_number — номер договора. Пустая строка, если не найден.
8. service_text — своими словами, 1–3 предложения на русском языке: что именно закупается
   (услуга / работа / товар), для чего и в каком объёме, если объём указан.
9. tech_stack_mentioned — через запятую перечисли упомянутые технологии, платформы,
   программные продукты, оборудование (например: "1С:Предприятие, PostgreSQL, Cisco").
   Если ничего не упомянуто — "".
10. district — район, город или область (например: "Актюбинская область, г. Актобе").
11. confidence:
   - "high"   — документ хорошо читается, все ключевые поля найдены однозначно;
   - "medium" — часть полей не найдена или есть небольшие сомнения в прочтении;
   - "low"    — скан плохого качества, текст читается с трудом, данные противоречивы
                или документ не похож на договор госзакупки.
12. ai_assessment — твои мысли и оценка как эксперта: какова реальная рыночная ценность
    этой закупки, есть ли схожие темы или аналоги на рынке, насколько адекватно описан
    предмет (1-4 предложения на русском языке).
Отвечай только JSON по схеме, без пояснений."""

USER_PROMPT = "Извлеки данные из приложенного документа (файл: {name}) согласно схеме."

class ContractData(BaseModel):
    supplier: str = Field(description="Наименование поставщика/подрядчика/исполнителя. Пустая строка, если не найдено.")
    customer: str = Field(description="Наименование заказчика (государственный орган/учреждение). Пустая строка, если не найдено.")
    amount_kzt: float = Field(description="Общая сумма договора в тенге, только число. 0, если не найдена.")
    contract_date: str = Field(description="Дата заключения договора в формате YYYY-MM-DD. Пустая строка, если не найдена.")
    lot_number: str = Field(description="Номер лота или номер закупки/объявления. Пустая строка, если нет.")
    contract_number: str = Field(description="Номер договора. Пустая строка, если не найден.")
    service_text: str = Field(description="Описание предмета закупки своими словами, 1–3 предложения на русском языке. Пустая строка, если предмет не определяется.")
    tech_stack_mentioned: str = Field(description="Упомянутые технологии, платформы, ПО, оборудование через запятую. Пустая строка, если не упомянуты.")
    district: str = Field(description="Район / город / область, если указаны. Пустая строка, если нет.")
    confidence: Literal["high", "medium", "low"] = Field(description="Уверенность в извлечённых данных: high / medium / low.")
    ai_assessment: str = Field(description="Оценка ИИ (мысли о ценности, аналоги на рынке, адекватность описания). Пустая строка, если нет данных.")

# ---------------------------------------------------------------------------
# Нормализация текста услуги
# ---------------------------------------------------------------------------

ORG_FORMS = [
    "ТОО", "ИП", "АО", "ОАО", "ЗАО", "НАО", "ООО", "ПАО",
    "ГУ", "РГУ", "КГУ", "ГККП", "КГП", "РГП", "РГКП", "ГП", "ОО", "ОФ", "ЧУ",
    "ЖШС", "ЖК", "АҚ", "ММ", "КММ", "РММ", "МКК", "ШЖҚ", "РМК", "КМҚК", "ҚМҚК",
    "LLP", "LLC", "JSC", "LTD",
]
_ORG_FORMS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in sorted(ORG_FORMS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
_QUOTES_RE = re.compile(r"[«»\"„“”‘’']")
_WHITESPACE_RE = re.compile(r"\s+")

def normalize_service_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    text = _ORG_FORMS_RE.sub(" ", text)
    text = _QUOTES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Работа с OpenAI API
# ---------------------------------------------------------------------------

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_once(client: OpenAI, model_name: str, pdf_path: Path) -> dict:
    images = pdf2image.convert_from_path(str(pdf_path), dpi=150, first_page=1, last_page=20)
    
    content_list = [{"type": "text", "text": USER_PROMPT.format(name=pdf_path.name)}]
    for img in images:
        base64_img = encode_image(img)
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}",
                "detail": "auto"
            }
        })
        
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_list}
        ],
        response_format=ContractData,
        temperature=0.0,
        timeout=REQUEST_TIMEOUT
    )
    
    return response.choices[0].message.parsed.model_dump()

def coerce_record(raw: dict) -> dict:
    rec = {}
    for field in STRING_FIELDS:
        value = raw.get(field)
        rec[field] = str(value).strip() if value is not None else ""

    amount = raw.get("amount_kzt", 0)
    if isinstance(amount, str):
        amount = amount.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 0.0
    if amount != amount or amount < 0:
        amount = 0.0
    rec["amount_kzt"] = int(amount) if amount.is_integer() else round(amount, 2)

    confidence = str(raw.get("confidence", "")).strip().lower()
    rec["confidence"] = confidence if confidence in CONFIDENCE_VALUES else "low"
    return rec

def extract_with_retry(client: OpenAI, model_name: str, pdf_path: Path) -> dict:
    total_attempts = MAX_RETRIES + 1
    last_error = ""

    for attempt in range(1, total_attempts + 1):
        try:
            rec = coerce_record(extract_once(client, model_name, pdf_path))
            rec["error"] = ""
            return rec
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                if isinstance(exc, RateLimitError):
                    delay = RATE_LIMIT_DELAY
                else:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"    ⚠ попытка {attempt}/{total_attempts} не удалась: "
                    f"{last_error[:200]} — повтор через {delay} с",
                    flush=True,
                )
                time.sleep(delay)

    rec = coerce_record({})
    rec["confidence"] = "low"
    rec["error"] = last_error
    return rec

# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_json(records: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def save_csv(records: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 1: извлечение данных из PDF договоров госзакупок через OpenAI API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="папка с PDF-файлами договоров")
    parser.add_argument("--output", default="contracts.json", help="путь к выходному JSON")
    parser.add_argument(
        "--csv", default=None,
        help="путь к выходному CSV (по умолчанию — рядом с JSON, с расширением .csv)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели OpenAI (например gpt-4o)")
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="пауза между файлами в секундах",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "ОШИБКА: не задана переменная окружения OPENAI_API_KEY.\n",
            file=sys.stderr,
        )
        return 1

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ОШИБКА: папка не найдена: {input_dir}", file=sys.stderr)
        return 1

    pdf_files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        print(f"ОШИБКА: в папке {input_dir} нет PDF-файлов", file=sys.stderr)
        return 1

    output_json = Path(args.output)
    output_csv = Path(args.csv) if args.csv else output_json.with_suffix(".csv")

    client = OpenAI(api_key=api_key)

    total = len(pdf_files)
    print(f"Модель: {args.model}. Найдено PDF-файлов: {total}. Папка: {input_dir}\n")

    records: list = []
    try:
        for idx, pdf_path in enumerate(pdf_files, start=1):
            print(f"[{idx}/{total}] Обработка: {pdf_path.name}", flush=True)

            rec = extract_with_retry(client, args.model, pdf_path)
            rec["source_file"] = str(pdf_path.relative_to(input_dir))
            rec["service_text_normalized"] = normalize_service_text(rec["service_text"])
            rec = {key: rec.get(key, "") for key in CSV_FIELDS}
            records.append(rec)

            if rec["error"]:
                print(f"    ✖ не удалось извлечь: {rec['error'][:200]}", flush=True)
            else:
                print(
                    f"    ✔ {rec['supplier'] or '—'} | {rec['amount_kzt']:,} тг | "
                    f"{rec['contract_date'] or '—'} | confidence={rec['confidence']}",
                    flush=True,
                )

            save_json(records, output_json)
            save_csv(records, output_csv)

            if idx < total and args.pause > 0:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nПрервано пользователем — сохраняю уже обработанные файлы.", flush=True)
        save_json(records, output_json)
        save_csv(records, output_csv)

    by_conf = {c: sum(1 for r in records if r["confidence"] == c) for c in CONFIDENCE_VALUES}
    failed = sum(1 for r in records if r["error"])
    print(
        f"\nГотово: обработано {len(records)} из {total}. "
        f"confidence: high={by_conf['high']}, medium={by_conf['medium']}, low={by_conf['low']}; "
        f"ошибок API: {failed}"
    )
    print(f"JSON: {output_json}\nCSV:  {output_csv}")
    
    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
