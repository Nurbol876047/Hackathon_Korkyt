#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 1 из 5 — извлечение структурированных данных из PDF-файлов договоров
и актов выполненных работ через Gemini API.

PDF отправляется в модель напрямую (без OCR): Gemini читает PDF нативно,
включая отсканированные документы. Ответ модели ограничен JSON-схемой
(structured output / response_schema), поэтому парсить текст руками не нужно —
API гарантирует валидный JSON заданной структуры.

Запуск:
    export GEMINI_API_KEY="ваш_ключ"
    python extract_stage1.py --input ./contracts --output contracts.json

Результат:
    contracts.json — полный список записей (по одной на каждый PDF)
    contracts.csv  — та же таблица в CSV (utf-8-sig — корректно открывается
                     в Excel с кириллицей)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

# Google пометила google-generativeai как устаревшую (рекомендует google-genai),
# но библиотека продолжает работать. Глушим предупреждение, чтобы не засорять вывод.
warnings.filterwarnings("ignore", category=FutureWarning, message=r"(?s).*google\.generativeai")

import google.generativeai as genai              # noqa: E402
from google.generativeai import protos           # noqa: E402
from google.api_core import exceptions as gexc   # noqa: E402


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

# gemini-2.0-flash отключена Google (API возвращает 404 и рекомендует gemini-3.6-flash).
# Другую модель можно указать флагом --model.
DEFAULT_MODEL = "gemini-3.6-flash"

MAX_RETRIES = 2            # число ПОВТОРНЫХ попыток после первой неудачной (итого 3 запроса)
RETRY_BASE_DELAY = 5       # секунд; пауза удваивается с каждой повторной попыткой
RATE_LIMIT_DELAY = 30      # секунд ожидания, если API вернул 429 (превышена квота запросов)
REQUEST_TIMEOUT = 300      # секунд на один запрос к API (большие сканы читаются долго)

# Inline-передача байтов ограничена ~20 МБ на запрос. PDF крупнее этого порога
# загружаем через File API, а после запроса удаляем с серверов Google.
INLINE_LIMIT_BYTES = 18 * 1024 * 1024

CONFIDENCE_VALUES = ("high", "medium", "low")

# Строковые поля схемы (заполняются моделью). amount_kzt и confidence обрабатываются отдельно.
STRING_FIELDS = (
    "supplier",
    "customer",
    "contract_date",
    "lot_number",
    "contract_number",
    "service_text",
    "tech_stack_mentioned",
    "district",
)

# Порядок колонок в CSV и ключей в JSON. К полям схемы добавлены служебные:
#   source_file             — имя исходного PDF (чтобы запись всегда можно было сопоставить с файлом)
#   service_text_normalized — нормализованный текст услуги для сравнения похожести на Этапе 2
#   error                   — текст ошибки API, если извлечение не удалось (иначе "")
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
    "error",
]


# ---------------------------------------------------------------------------
# Промпт и JSON-схема ответа
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты — ассистент аудитора государственных закупок Республики Казахстан.
Тебе передают PDF-файл: договор о государственных закупках, техническую спецификацию
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
7. service_text — своими словами, 1–3 предложения на русском языке: что именно закупается
   (услуга / работа / товар), для чего и в каком объёме, если объём указан.
8. tech_stack_mentioned — через запятую перечисли упомянутые технологии, платформы,
   программные продукты, оборудование (например: "1С:Предприятие, PostgreSQL, Cisco").
   Если ничего не упомянуто — "".
9. district — район, город или область (например: "Актюбинская область, г. Актобе").
10. confidence:
   - "high"   — документ хорошо читается, все ключевые поля найдены однозначно;
   - "medium" — часть полей не найдена или есть небольшие сомнения в прочтении;
   - "low"    — скан плохого качества, текст читается с трудом, данные противоречивы
                или документ не похож на договор госзакупки.
Отвечай только JSON по схеме, без пояснений."""

USER_PROMPT = "Извлеки данные из приложенного документа (файл: {name}) согласно схеме."


def build_response_schema() -> protos.Schema:
    """
    JSON-схема ответа модели для structured output.

    Передаётся в GenerationConfig(response_schema=...). Gemini гарантирует, что ответ
    будет валидным JSON именно такой структуры: все поля присутствуют, типы соблюдены,
    confidence — только одно из перечисленных значений.
    """
    S, T = protos.Schema, protos.Type
    return S(
        type=T.OBJECT,
        properties={
            "supplier": S(
                type=T.STRING,
                description="Наименование поставщика/подрядчика/исполнителя. "
                            "Пустая строка, если не найдено.",
            ),
            "customer": S(
                type=T.STRING,
                description="Наименование заказчика (государственный орган/учреждение). "
                            "Пустая строка, если не найдено.",
            ),
            "amount_kzt": S(
                type=T.NUMBER,
                description="Общая сумма договора в тенге, только число. 0, если не найдена.",
            ),
            "contract_date": S(
                type=T.STRING,
                description="Дата заключения договора в формате YYYY-MM-DD. "
                            "Пустая строка, если не найдена.",
            ),
            "lot_number": S(
                type=T.STRING,
                description="Номер лота или номер закупки/объявления. Пустая строка, если нет.",
            ),
            "contract_number": S(
                type=T.STRING,
                description="Номер договора. Пустая строка, если не найден.",
            ),
            "service_text": S(
                type=T.STRING,
                description="Описание предмета закупки своими словами, 1–3 предложения "
                            "на русском языке. Пустая строка, если предмет не определяется.",
            ),
            "tech_stack_mentioned": S(
                type=T.STRING,
                description="Упомянутые технологии, платформы, ПО, оборудование через запятую. "
                            "Пустая строка, если не упомянуты.",
            ),
            "district": S(
                type=T.STRING,
                description="Район / город / область, если указаны. Пустая строка, если нет.",
            ),
            "confidence": S(
                type=T.STRING,
                format_="enum",
                enum=list(CONFIDENCE_VALUES),
                description="Уверенность в извлечённых данных: high / medium / low.",
            ),
        },
        required=[*STRING_FIELDS, "amount_kzt", "confidence"],
    )


# ---------------------------------------------------------------------------
# Нормализация текста услуги (для сравнения похожести на Этапе 2)
# ---------------------------------------------------------------------------

# Организационно-правовые формы (рус. и каз.), которые нужно убрать из текста услуги.
# Сюда намеренно НЕ включены "ПК" (может означать "персональный компьютер")
# и другие сокращения, совпадающие с ИТ-терминами.
ORG_FORMS = [
    # русские
    "ТОО", "ИП", "АО", "ОАО", "ЗАО", "НАО", "ООО", "ПАО",
    "ГУ", "РГУ", "КГУ", "ГККП", "КГП", "РГП", "РГКП", "ГП", "ОО", "ОФ", "ЧУ",
    # казахские
    "ЖШС", "ЖК", "АҚ", "ММ", "КММ", "РММ", "МКК", "ШЖҚ", "РМК", "КМҚК", "ҚМҚК",
    # латиница (иногда встречается в договорах)
    "LLP", "LLC", "JSC", "LTD",
]

# Регулярное выражение: любая из форм как отдельное слово (без учёта регистра).
# Формы отсортированы по убыванию длины, чтобы "ГККП" не срезалось до "ГП".
_ORG_FORMS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(f) for f in sorted(ORG_FORMS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Кавычки всех видов — после удаления ОПФ остаются "висячие" кавычки от названий компаний.
_QUOTES_RE = re.compile(r"[«»\"„“”‘’']")

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_service_text(text: str) -> str:
    """
    Приводит описание услуги к виду, удобному для сравнения похожести:
      - нижний регистр, "ё" -> "е";
      - удалены организационно-правовые формы (ТОО / ИП / АО / ЖШС / ...);
      - удалены кавычки;
      - переносы строк и повторные пробелы схлопнуты в один пробел.

    >>> normalize_service_text('Услуги ТОО «Alpha»  по\\nсопровождению  ПК')
    'услуги alpha по сопровождению пк'
    """
    if not text:
        return ""
    text = text.lower().replace("ё", "е")
    text = _ORG_FORMS_RE.sub(" ", text)
    text = _QUOTES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Работа с Gemini API
# ---------------------------------------------------------------------------

def _make_pdf_part(pdf_path: Path):
    """
    Готовит PDF для передачи в модель.

    Возвращает кортеж (part, uploaded_file):
      - для небольших файлов part — inline-байты, uploaded_file = None;
      - для крупных — файл загружается через File API, и uploaded_file нужно удалить
        после запроса (это делает вызывающий код).
    """
    if pdf_path.stat().st_size <= INLINE_LIMIT_BYTES:
        return {"mime_type": "application/pdf", "data": pdf_path.read_bytes()}, None

    uploaded = genai.upload_file(
        str(pdf_path), mime_type="application/pdf", display_name=pdf_path.name
    )
    # Ждём, пока файл обработается на стороне Google
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = genai.get_file(uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"File API: файл в состоянии {uploaded.state.name}")
    return uploaded, uploaded


def extract_once(model: genai.GenerativeModel, pdf_path: Path) -> dict:
    """
    Один запрос к Gemini: отправляет PDF и возвращает распарсенный JSON-ответ.
    Любая проблема (сеть, квота, блокировка фильтрами, обрыв генерации) — исключение,
    которое обрабатывает extract_with_retry().
    """
    part, uploaded = _make_pdf_part(pdf_path)
    try:
        response = model.generate_content(
            [USER_PROMPT.format(name=pdf_path.name), part],
            request_options={"timeout": REQUEST_TIMEOUT},
        )
    finally:
        # Загруженный через File API документ удаляем сразу — он больше не нужен
        if uploaded is not None:
            try:
                genai.delete_file(uploaded.name)
            except Exception:
                pass

    # Ответ мог быть заблокирован фильтрами безопасности — тогда кандидатов нет
    if not response.candidates:
        reason = getattr(response.prompt_feedback, "block_reason", None)
        raise RuntimeError(f"пустой ответ модели (block_reason={reason})")

    finish = response.candidates[0].finish_reason
    if finish.name != "STOP":
        # Например, MAX_TOKENS — JSON обрезан и невалиден
        raise RuntimeError(f"генерация прервана: finish_reason={finish.name}")

    return json.loads(response.text)


def coerce_record(raw: dict) -> dict:
    """
    Приводит сырой ответ модели к строгим типам схемы.
    Недостающие поля получают значения по умолчанию ("" / 0 / "low"),
    чтобы структура записи была одинаковой для всех файлов.
    """
    rec = {}
    for field in STRING_FIELDS:
        value = raw.get(field)
        rec[field] = str(value).strip() if value is not None else ""

    # Сумма: модель обязана вернуть число, но на всякий случай чистим строку вида "1 500 000,00"
    amount = raw.get("amount_kzt", 0)
    if isinstance(amount, str):
        amount = amount.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        amount = 0.0
    if amount != amount or amount < 0:   # NaN или отрицательная сумма — считаем ненайденной
        amount = 0.0
    rec["amount_kzt"] = int(amount) if amount.is_integer() else round(amount, 2)

    confidence = str(raw.get("confidence", "")).strip().lower()
    rec["confidence"] = confidence if confidence in CONFIDENCE_VALUES else "low"
    return rec


def extract_with_retry(model: genai.GenerativeModel, pdf_path: Path) -> dict:
    """
    Извлекает данные из PDF с повторными попытками.

    При неудаче всех попыток файл НЕ теряется: возвращается запись с пустыми полями,
    confidence = "low" и текстом последней ошибки в поле error.
    """
    total_attempts = MAX_RETRIES + 1
    last_error = ""

    for attempt in range(1, total_attempts + 1):
        try:
            rec = coerce_record(extract_once(model, pdf_path))
            rec["error"] = ""
            return rec
        except Exception as exc:  # noqa: BLE001 — намеренно ловим всё: файл должен остаться в выборке
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < total_attempts:
                # При превышении квоты (429) ждём дольше, иначе — экспоненциальная пауза
                if isinstance(exc, gexc.ResourceExhausted):
                    delay = RATE_LIMIT_DELAY
                else:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"    ⚠ попытка {attempt}/{total_attempts} не удалась: "
                    f"{last_error[:200]} — повтор через {delay} с",
                    flush=True,
                )
                time.sleep(delay)

    # Все попытки исчерпаны
    rec = coerce_record({})
    rec["confidence"] = "low"
    rec["error"] = last_error
    return rec


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_json(records: list, path: Path) -> None:
    """Сохраняет записи в JSON (ensure_ascii=False — кириллица как есть, а не \\uXXXX)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_csv(records: list, path: Path) -> None:
    """Сохраняет записи в CSV. utf-8-sig добавляет BOM — Excel корректно определяет кодировку."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Этап 1: извлечение данных из PDF договоров госзакупок через Gemini API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="папка с PDF-файлами договоров")
    parser.add_argument("--output", default="contracts.json", help="путь к выходному JSON")
    parser.add_argument(
        "--csv", default=None,
        help="путь к выходному CSV (по умолчанию — рядом с JSON, с расширением .csv)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели Gemini")
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="пауза между файлами в секундах (бережём лимит запросов в минуту)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Ключ API — только из переменной окружения
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print(
            "ОШИБКА: не задана переменная окружения GEMINI_API_KEY.\n"
            "Получите ключ на https://aistudio.google.com/apikey и выполните:\n"
            '    export GEMINI_API_KEY="ваш_ключ"',
            file=sys.stderr,
        )
        return 1

    # 2. Собираем список PDF (включая вложенные папки, расширение без учёта регистра)
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

    # 3. Настраиваем модель: температура 0 (детерминированность), structured output по схеме
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=args.model,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=build_response_schema(),
        ),
    )

    total = len(pdf_files)
    print(f"Модель: {args.model}. Найдено PDF-файлов: {total}. Папка: {input_dir}\n")

    # 4. Пакетная обработка
    records: list = []
    try:
        for idx, pdf_path in enumerate(pdf_files, start=1):
            print(f"[{idx}/{total}] Обработка: {pdf_path.name}", flush=True)

            rec = extract_with_retry(model, pdf_path)
            rec["source_file"] = str(pdf_path.relative_to(input_dir))
            rec["service_text_normalized"] = normalize_service_text(rec["service_text"])
            # Фиксируем порядок ключей, как в CSV_FIELDS
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

            # Сохраняем после каждого файла: при обрыве прогресс не потеряется
            save_json(records, output_json)
            save_csv(records, output_csv)

            if idx < total and args.pause > 0:
                time.sleep(args.pause)
    except KeyboardInterrupt:
        print("\nПрервано пользователем — сохраняю уже обработанные файлы.", flush=True)
        save_json(records, output_json)
        save_csv(records, output_csv)

    # 5. Итоги
    by_conf = {c: sum(1 for r in records if r["confidence"] == c) for c in CONFIDENCE_VALUES}
    failed = sum(1 for r in records if r["error"])
    print(
        f"\nГотово: обработано {len(records)} из {total}. "
        f"confidence: high={by_conf['high']}, medium={by_conf['medium']}, low={by_conf['low']}; "
        f"ошибок API: {failed}"
    )
    print(f"JSON: {output_json}\nCSV:  {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
