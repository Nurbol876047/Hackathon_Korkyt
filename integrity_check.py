#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 5 из 5, модуль А — проверка целостности PDF-файлов договоров.

Чисто техническая проверка БЕЗ использования ИИ: по метаданным и структуре страниц
ищутся признаки того, что документ мог быть отредактирован после первоначального
подписания/скана. Результат — ИНДИКАТОР для проверки оригинала, а не вывод о подделке.

Вход:   папка с PDF (та же, что на Этапе 1)
Выход:  integrity_check.json — по одной записи на файл (ключ стыковки: source_file)

Флаги:
    modified_after_creation — /ModDate позже /CreationDate более чем на MOD_DELTA_DAYS
    editor_producer         — /Producer или /Creator указывает на текстовый редактор,
                              а не на сканер / типографское ПО
    text_layer_mismatch     — на странице одновременно и растровое изображение (скан),
                              и текстовый слой; эвристика — OCR-слой тоже даёт такую картину
    incremental_update      — в файле несколько секций %%EOF (инкрементальное сохранение —
                              так сохраняют правки поверх уже готового PDF)

Запуск:
    python integrity_check.py --input ./contracts --output integrity_check.json
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Настройки (пороги вынесены сюда, чтобы легко подстроить)
# ---------------------------------------------------------------------------

MOD_DELTA_DAYS = 1                 # ModDate позже CreationDate более чем на столько дней — флаг

# Подстроки (без учёта регистра) в /Producer или /Creator, типичные для редакторов.
# Список легко дополнить: сканеры и типографский софт сюда не входят.
EDITOR_PRODUCER_MARKERS = (
    "microsoft word", "microsoft® word", "microsoft office", "word для",
    "libreoffice", "openoffice", "writer",
    "acrobat pro", "acrobat dc", "adobe acrobat 2", "acrobat pdfmaker",
    "wps office", "google docs", "pages", "foxit phantom", "foxit pdf editor",
    "nitro pro", "pdf-xchange editor", "ilovepdf", "smallpdf", "sejda", "pdfescape",
    "pdfsam", "master pdf editor", "wondershare", "pdfelement",
)
# Признаки сканера/типографского ПО — если совпало, флаг editor_producer не ставится,
# даже если строка содержит что-то из списка выше (например «Acrobat Pro» у Adobe Scan)
SCANNER_PRODUCER_MARKERS = (
    "scan", "canon", "epson", "xerox", "kyocera", "ricoh", "brother", "hp ", "hewlett",
    "konica", "fujitsu", "scansnap", "abbyy", "finereader", "naps2", "paperport",
    "kofax", "readiris", "ghostscript", "cairo", "reportlab", "wkhtmltopdf", "chrome",
    "chromium", "skia", "latex", "pdftex", "xetex", "quartz",
)

# Веса флагов: integrity_score = 100 − сумма сработавших весов (не ниже 0)
FLAG_WEIGHTS = {
    "modified_after_creation": 35,
    "editor_producer": 25,
    "incremental_update": 25,
    "text_layer_mismatch": 15,
}
INTEGRITY_OK_SCORE = 80      # >= — "признаков не обнаружено"
INTEGRITY_CHECK_SCORE = 50   # >= — "требует проверки", ниже — "высокий риск"

LEVEL_OK = "признаков не обнаружено"
LEVEL_CHECK = "требует проверки оригинала"
LEVEL_RISK = "высокий риск: требует проверки оригинала"
LEVEL_ERROR = "файл не удалось прочитать"

NOTE_FLAGGED = "Обнаружены технические признаки, требующие проверки оригинала документа."
NOTE_CLEAN = "Технических признаков редактирования после создания не обнаружено."

# ---------------------------------------------------------------------------
# Чтение PDF: pikepdf → pypdf (что установлено)
# ---------------------------------------------------------------------------

try:
    import pikepdf
    HAVE_PIKEPDF = True
except ImportError:            # pragma: no cover
    HAVE_PIKEPDF = False
try:
    from pypdf import PdfReader
    HAVE_PYPDF = True
except ImportError:            # pragma: no cover
    HAVE_PYPDF = False

if not HAVE_PIKEPDF and not HAVE_PYPDF:
    print("ОШИБКА: нужна библиотека pikepdf или pypdf (pip install pikepdf pypdf)", file=sys.stderr)
    sys.exit(1)


_PDF_DATE_RE = re.compile(
    r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?\s*([+\-Zz])?(\d{2})?'?(\d{2})?'?"
)


def parse_pdf_date(raw) -> datetime | None:
    """Разбирает строку вида D:20240115123000+06'00' в datetime (UTC). Неполные даты допустимы."""
    if not raw:
        return None
    m = _PDF_DATE_RE.match(str(raw).strip())
    if not m:
        return None
    y, mo, d, h, mi, s, sign, tzh, tzm = m.groups()
    try:
        dt = datetime(int(y), int(mo or 1), int(d or 1), int(h or 0), int(mi or 0), int(s or 0),
                      tzinfo=timezone.utc)
    except ValueError:
        return None
    if sign in ("+", "-") and tzh:
        offset = timedelta(hours=int(tzh), minutes=int(tzm or 0))
        dt = dt - offset if sign == "+" else dt + offset
    return dt


def read_metadata(pdf_path: Path) -> dict:
    """Метаданные /Info + число страниц + по каждой странице: есть ли текст и есть ли изображения."""
    info = {"creation_date": "", "mod_date": "", "producer": "", "creator": "", "author": "",
            "pages": 0, "pages_with_text": 0, "pages_with_images": 0, "pages_text_and_image": 0}

    if HAVE_PIKEPDF:
        with pikepdf.open(pdf_path) as pdf:
            di = pdf.docinfo
            get = lambda k: str(di.get(k, "")) if k in di else ""
            info.update(creation_date=get("/CreationDate"), mod_date=get("/ModDate"),
                        producer=get("/Producer"), creator=get("/Creator"), author=get("/Author"))
            info["pages"] = len(pdf.pages)
            for page in pdf.pages:
                has_img = _pikepdf_page_has_image(page)
                has_txt = _pikepdf_page_has_text(page)
                info["pages_with_images"] += has_img
                info["pages_with_text"] += has_txt
                info["pages_text_and_image"] += has_img and has_txt
        return info

    reader = PdfReader(str(pdf_path))
    md = reader.metadata or {}
    get = lambda k: str(md.get(k, "") or "")
    info.update(creation_date=get("/CreationDate"), mod_date=get("/ModDate"),
                producer=get("/Producer"), creator=get("/Creator"), author=get("/Author"))
    info["pages"] = len(reader.pages)
    for page in reader.pages:
        try:
            has_img = len(page.images) > 0
        except Exception:  # noqa: BLE001 — битые XObject не должны ронять проверку
            has_img = False
        try:
            has_txt = bool((page.extract_text() or "").strip())
        except Exception:  # noqa: BLE001
            has_txt = False
        info["pages_with_images"] += has_img
        info["pages_with_text"] += has_txt
        info["pages_text_and_image"] += has_img and has_txt
    return info


def _pikepdf_page_has_image(page) -> bool:
    try:
        xobjs = page.Resources.get("/XObject", {})
        return any(str(x.get("/Subtype", "")) == "/Image" for x in xobjs.values())
    except Exception:  # noqa: BLE001
        return False


def _pikepdf_page_has_text(page) -> bool:
    """Есть ли операторы вывода текста (Tj/TJ) в контенте страницы."""
    try:
        for op in pikepdf.parse_content_stream(page):
            if str(op.operator) in ("Tj", "TJ", "'", '"'):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def count_eof_markers(pdf_path: Path) -> int:
    """Число %%EOF: >1 означает инкрементальные сохранения поверх исходного файла."""
    try:
        return pdf_path.read_bytes().count(b"%%EOF")
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Флаги и оценка
# ---------------------------------------------------------------------------

def integrity_flags(pdf_path: Path) -> dict:
    """Возвращает запись с метаданными, списком флагов, пояснениями и integrity_score."""
    rec = {
        "source_file": pdf_path.name,
        "creation_date": "", "mod_date": "", "producer": "", "creator": "", "author": "",
        "pages": 0, "eof_markers": 0,
        "flags": [], "flag_details": {},
        "integrity_score": None, "integrity_level": "", "note": "", "error": "",
    }
    try:
        meta = read_metadata(pdf_path)
    except Exception as exc:  # noqa: BLE001 — повреждённый/зашифрованный файл
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["integrity_level"] = LEVEL_ERROR
        rec["note"] = "Файл не удалось разобрать — проверьте оригинал вручную."
        return rec

    created, modified = parse_pdf_date(meta["creation_date"]), parse_pdf_date(meta["mod_date"])
    rec.update(creation_date=created.isoformat() if created else meta["creation_date"],
               mod_date=modified.isoformat() if modified else meta["mod_date"],
               producer=meta["producer"], creator=meta["creator"], author=meta["author"],
               pages=meta["pages"])
    flags, details = [], {}

    # 1. Дата изменения заметно позже даты создания
    if created and modified and (modified - created) > timedelta(days=MOD_DELTA_DAYS):
        flags.append("modified_after_creation")
        details["modified_after_creation"] = (
            f"дата изменения ({modified:%Y-%m-%d}) позже даты создания ({created:%Y-%m-%d}) "
            f"на {(modified - created).days} дн. при пороге {MOD_DELTA_DAYS} дн.")

    # 2. Producer/Creator — редактор, а не сканер
    software = f"{meta['producer']} | {meta['creator']}".lower()
    is_scanner = any(m in software for m in SCANNER_PRODUCER_MARKERS)
    editor_hit = next((m for m in EDITOR_PRODUCER_MARKERS if m in software), None)
    if editor_hit and not is_scanner:
        flags.append("editor_producer")
        details["editor_producer"] = (
            f"ПО создания/обработки — «{meta['producer'] or meta['creator']}» (маркер «{editor_hit}»): "
            "документ сформирован или пересохранён в редакторе, а не получен со сканера.")

    # 3. Инкрементальные сохранения
    eof = count_eof_markers(pdf_path)
    rec["eof_markers"] = eof
    if eof > 1:
        flags.append("incremental_update")
        details["incremental_update"] = (
            f"в файле {eof} секции %%EOF — файл сохранялся поверх исходной версии {eof - 1} раз(а) "
            "(так выглядят правки, аннотации или добавленные страницы поверх готового PDF).")

    # 4. Текстовый слой поверх скана
    if meta["pages"] and meta["pages_text_and_image"] > 0:
        flags.append("text_layer_mismatch")
        details["text_layer_mismatch"] = (
            f"на {meta['pages_text_and_image']} из {meta['pages']} стр. одновременно есть растровое "
            "изображение и текстовый слой. Это может быть обычный OCR-слой, но может и наложенный текст — "
            "эвристика, не доказательство.")

    score = max(0, 100 - sum(FLAG_WEIGHTS[f] for f in flags))
    rec.update(flags=flags, flag_details=details, integrity_score=score,
               integrity_level=(LEVEL_OK if score >= INTEGRITY_OK_SCORE else
                                LEVEL_CHECK if score >= INTEGRITY_CHECK_SCORE else LEVEL_RISK),
               note=NOTE_FLAGGED if flags else NOTE_CLEAN)
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Этап 5А: проверка целостности PDF-договоров (без ИИ)")
    parser.add_argument("--input", default="./contracts", help="папка с PDF (рекурсивно)")
    parser.add_argument("--output", default="integrity_check.json", help="куда сохранить результат")
    args = parser.parse_args()

    input_dir = Path(args.input)
    pdfs = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"ОШИБКА: в {input_dir} нет PDF-файлов", file=sys.stderr)
        return 1
    print(f"Библиотека: {'pikepdf' if HAVE_PIKEPDF else 'pypdf'}. Файлов: {len(pdfs)}\n")

    records = []
    for i, pdf in enumerate(pdfs, start=1):
        rec = integrity_flags(pdf)
        # source_file — путь относительно входной папки, как на Этапе 1
        rec["source_file"] = str(pdf.relative_to(input_dir))
        records.append(rec)
        status = rec["error"][:80] if rec["error"] else \
            f"score {rec['integrity_score']} — {rec['integrity_level']}" + \
            (f" [{', '.join(rec['flags'])}]" if rec["flags"] else "")
        print(f"  [{i}/{len(pdfs)}] {rec['source_file']}: {status}", flush=True)

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    levels = {}
    for r in records:
        levels[r["integrity_level"]] = levels.get(r["integrity_level"], 0) + 1
    print("\nИтог: " + ", ".join(f"{k}: {v}" for k, v in levels.items()))
    print(f"Результат: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
