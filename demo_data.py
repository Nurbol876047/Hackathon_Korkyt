#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Генератор ДЕМО-данных для дашборда (Этап 5), когда результаты Этапов 1–4 ещё не
получены (нет PDF или ключа Gemini).

Создаёт в указанной папке файлы в том же формате, что и реальные этапы:
    contracts.json, alternatives.json, price_benchmark.json, category_stats.json,
    tor_compliance.json, fragmentation_clusters.json
и папку --pdf-dir с демо-PDF (пустые страницы с разными метаданными), по которым
integrity_check.py (Этап 5А) строит integrity_check.json уже по-настоящему.

ВАЖНО: все компании, договоры и суммы вымышлены и сгенерированы для демонстрации.

Запуск:
    python demo_data.py --output-dir ./demo_results --pdf-dir ./demo_contracts
"""

import argparse
import json
import random
import statistics
from datetime import date, timedelta
from pathlib import Path

try:
    import pikepdf
except ImportError:  # демо-PDF просто не создаются
    pikepdf = None

SEED = 42

# (source_file, supplier, customer, amount, date, service_text, tech_stack, district)
CONTRACTS = [
    ("dogovor_001.pdf", "ТОО «Цифровой Меридиан»", "ГУ «Аппарат акима Сарыаркинского района»",
     3_600_000, "2026-03-12", "Разработка веб-портала для приёма обращений граждан с личным кабинетом.",
     "PHP, Laravel, PostgreSQL", "Сарыаркинский район"),
    ("dogovor_002.pdf", "ТОО «Аймақ Софт»", "ГУ «Отдел образования Есильского района»",
     2_950_000, "2026-02-03", "Разработка сайта отдела образования с новостной лентой и разделом документов.",
     "WordPress, PHP, MySQL", "Есильский район"),
    ("dogovor_003.pdf", "ТОО «КазНет Сервис»", "ГУ «Аппарат акима Сарыаркинского района»",
     6_200_000, "2026-04-20", "Техническая поддержка и сопровождение информационных систем акимата на 12 месяцев.",
     "1С, Windows Server", "Сарыаркинский район"),
    ("dogovor_004.pdf", "ТОО «Дала Технолоджис»", "ГУ «Отдел культуры Алматинского района»",
     1_350_000, "2026-01-28", "Разработка лендинга для анонса культурных мероприятий района.",
     "HTML, CSS, JavaScript", "Алматинский район"),
    ("dogovor_005.pdf", "ИП Сериков Б.Т.", "ГУ «Отдел образования Есильского района»",
     2_850_000, "2026-02-17", "Создание интернет-сайта отдела образования: разделы новостей и документы.",
     "WordPress, PHP", "Есильский район"),
    ("dogovor_006.pdf", "ТОО «Smart Qala»", "ГУ «Управление цифровизации г. Астаны»",
     11_500_000, "2026-03-30", "Разработка мобильного приложения для записи в очередь в ЦОН (iOS и Android).",
     "Flutter, Firebase, REST API", "г. Астана"),
    ("dogovor_007.pdf", "ТОО «Аймақ Софт»", "ГУ «Отдел образования Есильского района»",
     2_900_000, "2026-03-02", "Доработка сайта отдела образования: добавление раздела документов и новостей.",
     "WordPress, PHP", "Есильский район"),
    ("dogovor_008.pdf", "ТОО «БайтЛаб»", "ГУ «Аппарат акима Байконурского района»",
     3_400_000, "2026-05-11", "Внедрение системы электронного документооборота на 40 рабочих мест.",
     "Documentolog, PostgreSQL", "Байконурский район"),
    ("dogovor_009.pdf", "ТОО «Цифровой Меридиан»", "ГУ «Аппарат акима Сарыаркинского района»",
     7_900_000, "2026-06-05", "Разработка веб-портала для приёма обращений граждан (второй этап, модуль аналитики).",
     "PHP, Laravel, React", "Сарыаркинский район"),
    ("dogovor_010.pdf", "ТОО «Орда Диджитал»", "ГУ «Отдел ЖКХ Алматинского района»",
     980_000, "2026-04-14", "Создание информационного сайта отдела ЖКХ с формой обратной связи.",
     "Tilda", "Алматинский район"),
    ("dogovor_011.pdf", "ТОО «КазНет Сервис»", "ГУ «Отдел культуры Алматинского района»",
     2_100_000, "2026-05-22", "Техническое сопровождение сайта и хостинг на 12 месяцев.",
     "Nginx, Ubuntu", "Алматинский район"),
    ("dogovor_012.pdf", "ТОО «Smart Qala»", "ГУ «Управление цифровизации г. Астаны»",
     9_800_000, "2026-06-18", "Разработка чат-бота для консультаций граждан по госуслугам в Telegram и WhatsApp.",
     "Python, Telegram Bot API, Dialogflow", "г. Астана"),
]

CATEGORIES = {
    "разработка веб-сайтов и порталов": {
        "desc": "Создание сайтов, лендингов, порталов и их доработка",
        "files": ["dogovor_001.pdf", "dogovor_002.pdf", "dogovor_004.pdf", "dogovor_005.pdf",
                  "dogovor_007.pdf", "dogovor_009.pdf", "dogovor_010.pdf"],
        "market": (900_000, 5_500_000),
    },
    "мобильные приложения и чат-боты": {
        "desc": "Мобильные приложения, боты, интеграции с мессенджерами",
        "files": ["dogovor_006.pdf", "dogovor_012.pdf"],
        "market": (5_000_000, 12_000_000),
    },
    "сопровождение и техподдержка ИС": {
        "desc": "Техническая поддержка, хостинг, сопровождение систем",
        "files": ["dogovor_003.pdf", "dogovor_011.pdf"],
        "market": (1_500_000, 4_500_000),
    },
    "внедрение информационных систем": {
        "desc": "Внедрение СЭД, 1С и других готовых систем",
        "files": ["dogovor_008.pdf"],
        "market": (2_500_000, 4_500_000),
    },
}

COMPANY_NAMES = ["ТОО «Алаш Код»", "ТОО «Тұран Софт»", "ИП Жанибеков А.", "ТОО «Kokshe IT»",
                 "ТОО «Bilim Digital»", "ТОО «Jetysu Web»", "ТОО «Sary-Arka Tech»", "ИП Оспанова Д.",
                 "ТОО «Nomad Labs»", "ТОО «Qazaq Cloud»", "ТОО «Astana Dev»", "ТОО «Ulytau Systems»",
                 "ТОО «Aqmola IT»", "ИП Каримов Р.", "ТОО «Beibitshilik Soft»", "ТОО «Turkistan Web»",
                 "ТОО «Irtysh Digital»", "ТОО «Zhaiyq Apps»"]
CITIES = ["Астана", "Алматы", "Караганда", "Шымкент", "Кокшетау", "Павлодар", "Актобе"]

RISK_CHECK_PCT, RISK_HIGH_PCT = 15.0, 40.0


def build(out_dir: Path) -> None:
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    contracts = []
    for i, (sf, sup, cust, amt, d, svc, tech, dist) in enumerate(CONTRACTS, start=1):
        contracts.append({
            "source_file": sf, "supplier": sup, "customer": cust, "amount_kzt": amt,
            "contract_date": d, "lot_number": f"LOT-2026-{i:03d}", "contract_number": f"№ {i * 7}-Д/2026",
            "service_text": svc, "service_text_normalized": svc.lower(), "tech_stack_mentioned": tech,
            "district": dist, "confidence": rng.choice(["high", "high", "medium"]), "error": "",
        })
    by_file = {c["source_file"]: c for c in contracts}

    # --- рынок и альтернативы (Этап 2) ---
    market = {"is_synthetic": True, "note": "Синтетическая база для демо: все компании вымышлены.",
              "created_at": "2026-09-15T12:00:00", "categories": {}}
    alternatives, benchmark, cat_stats = [], [], {}
    for cat, meta in CATEGORIES.items():
        lo, hi = meta["market"]
        companies = []
        for name in rng.sample(COMPANY_NAMES, 12):
            companies.append({"company": name, "city": rng.choice(CITIES),
                              "company_size": rng.choice(["малая", "средняя", "крупная"]),
                              "offer": f"{meta['desc'].split(',')[0]} под ключ", "tech_stack": "",
                              "price_kzt": int(rng.uniform(lo, hi) // 10_000 * 10_000)})
        market["categories"][cat] = {"description": meta["desc"], "price_range_kzt": {"min": lo, "max": hi},
                                     "generated_at": "2026-09-15T12:00:00", "companies": companies}
        market_prices = [c["price_kzt"] for c in companies]
        contract_prices = [by_file[f]["amount_kzt"] for f in meta["files"]]
        all_prices = contract_prices + market_prices
        median = statistics.median(all_prices)
        cat_stats[cat] = {
            "description": meta["desc"], "median_price": round(median),
            "mean_price": round(statistics.mean(all_prices)), "std_price": round(statistics.stdev(all_prices)),
            "min_price": min(all_prices), "max_price": max(all_prices),
            "contracts_count": len(meta["files"]), "contracts_with_price": len(meta["files"]),
            "market_prices_count": len(market_prices), "sample_size": len(all_prices),
            "market_categories": [cat],
            "risk_counts": {"норма": 0, "требует проверки": 0, "высокий риск": 0, "недостаточно данных": 0},
        }
        for f in meta["files"]:
            c = by_file[f]
            price = c["amount_kzt"]
            cheaper = sorted([x for x in companies if x["price_kzt"] < price], key=lambda x: x["price_kzt"])[:5]
            alts = [{"company": x["company"], "city": x["city"], "price_kzt": x["price_kzt"],
                     "price_diff_pct": round((price - x["price_kzt"]) / price * 100, 1),
                     "relevance_reason": "аналогичный объём работ и стек", "offer": x["offer"]} for x in cheaper]
            best = alts[0] if alts else None
            alternatives.append({
                "contract_id": c["contract_number"], "source_file": f, "current_supplier": c["supplier"],
                "current_price_kzt": price, "service_text": c["service_text"], "service_category": cat,
                "market_source": "synthetic", "candidates_checked": len(companies), "alternatives": alts,
                "best_alternative": best,
                "potential_savings_kzt": price - best["price_kzt"] if best else 0,
                "potential_savings_pct": best["price_diff_pct"] if best else 0.0,
                "note": "" if alts else "более дешёвых релевантных предложений не найдено",
            })
            dev = round((price - median) / median * 100, 1)
            risk = ("высокий риск" if dev > RISK_HIGH_PCT else
                    "требует проверки" if dev > RISK_CHECK_PCT else "норма")
            cat_stats[cat]["risk_counts"][risk] += 1
            benchmark.append({
                "source_file": f, "contract_id": c["contract_number"], "supplier": c["supplier"],
                "amount_kzt": price, "category": cat, "category_median_price": round(median),
                "price_deviation_pct": dev, "price_ratio": round(price / median, 2), "risk_level": risk,
                "sample_size": len(all_prices), "district": c["district"], "district_key": c["district"].lower(),
                "district_context": "", "benchmark_note": "цена заметно ниже медианы" if dev < -40 else "",
            })

    # контекст по районам
    region_avg = statistics.mean(r["price_deviation_pct"] for r in benchmark)
    by_dist = {}
    for r in benchmark:
        by_dist.setdefault(r["district_key"], []).append(r["price_deviation_pct"])
    for r in benchmark:
        devs = by_dist[r["district_key"]]
        if len(devs) < 2:
            continue
        avg = statistics.mean(devs)
        base = (f"Район «{r['district_key']}»: среднее отклонение от медианы {avg:+.1f}% "
                f"при {region_avg:+.1f}% по региону в целом ({len(devs)} договоров)")
        r["district_context"] = base + (" — цены систематически выше" if avg - region_avg > 10
                                        else " — без систематического завышения")

    # --- соответствие ТЗ (Этап 4, модуль А) ---
    scores = {"dogovor_001.pdf": 92, "dogovor_002.pdf": 88, "dogovor_003.pdf": 45, "dogovor_004.pdf": 95,
              "dogovor_005.pdf": 84, "dogovor_006.pdf": 90, "dogovor_007.pdf": 78, "dogovor_008.pdf": 67,
              "dogovor_009.pdf": 58, "dogovor_010.pdf": 30, "dogovor_011.pdf": 89, "dogovor_012.pdf": 93}
    notes = {
        "dogovor_003.pdf": ("В ТЗ заявлена поддержка 1С и Windows Server, однако в актах выполненных работ "
                            "упоминается только обслуживание сайта на Linux.", ["обслуживание сайта на Linux"]),
        "dogovor_009.pdf": ("Заявлен React, но в описании результата — только серверные шаблоны Blade.",
                            ["серверные шаблоны Blade"]),
        "dogovor_010.pdf": ("Указан конструктор Tilda; при этом в спецификации описана «разработка на PHP-фреймворке» "
                            "и «собственная CMS» — формулировки не согласуются.", ["собственная CMS", "PHP-фреймворк"]),
        "dogovor_008.pdf": ("Заявлен Documentolog, но количество лицензий в акте (25) не совпадает с ТЗ (40).",
                            ["25 лицензий"]),
    }
    compliance = []
    for c in contracts:
        s = scores[c["source_file"]]
        lvl = "соответствует" if s >= 80 else "требует проверки" if s >= 50 else "высокий риск несоответствия"
        expl, phrases = notes.get(c["source_file"], ("Описание продукта согласуется с заявленным стеком.", []))
        compliance.append({"source_file": c["source_file"], "contract_id": c["contract_number"],
                           "supplier": c["supplier"], "tech_stack_mentioned": c["tech_stack_mentioned"],
                           "service_text": c["service_text"], "status": "проверено", "consistency_score": s,
                           "compliance_level": lvl, "flagged_phrases": phrases, "explanation": expl})

    # --- дробление (Этап 4, модуль Б) ---
    def cluster(cid, files, sim, score, reason):
        members = [by_file[f] for f in files]
        dates = sorted(date.fromisoformat(m["contract_date"]) for m in members)
        total = sum(m["amount_kzt"] for m in members)
        each_below = all(m["amount_kzt"] < 3_000_000 for m in members)
        level = "высокий" if score >= 70 else "средний" if score >= 40 else "низкий"
        return {
            "cluster_id": cid, "customer": members[0]["customer"],
            "contract_ids": [m["contract_number"] for m in members], "source_files": files,
            "contracts": [{"source_file": m["source_file"], "contract_id": m["contract_number"],
                           "contract_date": m["contract_date"], "supplier": m["supplier"],
                           "amount_kzt": m["amount_kzt"], "service_text": m["service_text"]} for m in members],
            "contracts_count": len(members), "total_amount_kzt": total, "avg_similarity": sim,
            "date_span_days": (dates[-1] - dates[0]).days, "each_below_threshold": each_below,
            "threshold_kzt": 3_000_000, "suspicion_score": score, "suspicion_level": level,
            "explanation": (f"{len(members)} договора одного заказчика с похожим предметом "
                            f"(похожесть {sim}), интервал {(dates[-1] - dates[0]).days} дн.; "
                            + ("каждый ниже порога" if each_below else "не все ниже порога")
                            + f" {3_000_000:,} ₸, суммарно {total:,} ₸. {reason}").replace(",", " "),
            "llm_check": {"same_subject": True, "reason": reason},
        }

    clusters = [
        cluster("F1", ["dogovor_002.pdf", "dogovor_005.pdf", "dogovor_007.pdf"], 0.91, 86,
                "Все три договора описывают создание/доработку одного и того же сайта отдела образования."),
        cluster("F2", ["dogovor_001.pdf", "dogovor_009.pdf"], 0.83, 48,
                "Второй договор оформлен как «второй этап» того же портала; суммы выше порога, "
                "поэтому признак дробления слабый."),
    ]

    dump = lambda name, obj: (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    dump("contracts.json", contracts)
    dump("market_database.json", market)
    dump("alternatives.json", alternatives)
    dump("price_benchmark.json", benchmark)
    dump("category_stats.json", cat_stats)
    dump("tor_compliance.json", compliance)
    dump("fragmentation_clusters.json", clusters)
    print(f"Демо-данные записаны в {out_dir}/ ({len(contracts)} договоров)")


# ---------------------------------------------------------------------------
# Демо-PDF для Этапа 5А: (source_file, producer, creator, created, modified, kind)
# kind: "scan" — страница-картинка без текста; "ocr" — картинка + текстовый слой;
#       "text" — только текст; "incremental" — text + инкрементальное сохранение
# ---------------------------------------------------------------------------

PDF_SPECS = {
    "dogovor_001.pdf": ("Canon iR-ADV C5535 PDF", "Canon Scan Utility", "20260312", "20260312", "scan"),
    "dogovor_002.pdf": ("Microsoft® Word 2019", "Microsoft® Word 2019", "20260203", "20260203", "text"),
    "dogovor_003.pdf": ("Adobe Acrobat Pro DC 23.1", "Adobe Acrobat Pro DC", "20260420", "20260702", "incremental"),
    "dogovor_004.pdf": ("ABBYY FineReader 15", "ScanSnap Manager", "20260128", "20260128", "ocr"),
    "dogovor_005.pdf": ("Xerox WorkCentre 7845", "Xerox Scan", "20260217", "20260217", "scan"),
    "dogovor_006.pdf": ("LibreOffice 7.6", "Writer", "20260330", "20260330", "text"),
    "dogovor_007.pdf": ("Microsoft® Word для Microsoft 365", "Microsoft® Word", "20260302", "20260302", "text"),
    "dogovor_008.pdf": ("Kyocera TASKalfa 3253ci", "Kyocera Scan", "20260511", "20260511", "scan"),
    "dogovor_009.pdf": ("Adobe Acrobat Pro DC 23.1", "Adobe Acrobat Pro DC", "20260605", "20260605", "text"),
    "dogovor_010.pdf": ("Nitro Pro 14", "Nitro Pro", "20260414", "20260527", "incremental"),
    "dogovor_011.pdf": ("Epson Scan 2", "Epson Scan 2", "20260522", "20260522", "scan"),
    "dogovor_012.pdf": ("Google Chrome / Skia", "Skia/PDF m124", "20260618", "20260618", "text"),
}


def _demo_page(pdf, kind: str):
    """Страница A4 с картинкой (серый прямоугольник как «скан») и/или ASCII-текстом."""
    from pikepdf import Dictionary, Name, Stream
    resources, content = Dictionary(), b""
    if kind in ("scan", "ocr"):
        img = Stream(pdf, bytes([200]) * (40 * 56), Type=Name.XObject, Subtype=Name.Image,
                     Width=40, Height=56, ColorSpace=Name.DeviceGray, BitsPerComponent=8)
        resources.XObject = Dictionary(Im0=img)
        content += b"q 495 0 0 742 50 50 cm /Im0 Do Q\n"
    if kind in ("ocr", "text", "incremental"):
        resources.Font = Dictionary(F1=Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))
        content += b"BT /F1 12 Tf 60 780 Td (DEMO CONTRACT - synthetic document for dashboard) Tj ET\n"
    return Dictionary(Type=Name.Page, MediaBox=[0, 0, 595, 842], Resources=resources,
                      Contents=Stream(pdf, content))


def build_pdfs(pdf_dir: Path) -> None:
    """Создаёт демо-PDF с заданными метаданными. Без pikepdf — тихо пропускается."""
    if pikepdf is None:
        print("pikepdf не установлен — демо-PDF не созданы (Этап 5А будет без данных)")
        return
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for name, (producer, creator, created, modified, kind) in PDF_SPECS.items():
        pdf = pikepdf.new()
        pdf.pages.append(pikepdf.Page(_demo_page(pdf, kind)))
        pdf.docinfo["/Producer"] = producer
        pdf.docinfo["/Creator"] = creator
        pdf.docinfo["/Author"] = "demo"
        pdf.docinfo["/CreationDate"] = f"D:{created}100000+06'00'"
        pdf.docinfo["/ModDate"] = f"D:{modified}153000+06'00'"
        path = pdf_dir / name
        pdf.save(path, fix_metadata_version=False)
        if kind == "incremental":
            # правка поверх готового файла: pikepdf дописывает новую секцию xref + %%EOF
            with pikepdf.open(path, allow_overwriting_input=True) as p2:
                p2.docinfo["/Subject"] = "amended"
                p2.save(path, fix_metadata_version=False)
            # эмулируем инкрементальное сохранение: второй %%EOF в хвосте файла
            with open(path, "ab") as f:
                f.write(b"\n%% incremental update marker\n%%EOF\n")
    print(f"Демо-PDF записаны в {pdf_dir}/ ({len(PDF_SPECS)} файлов)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Генерация демо-данных для дашборда")
    parser.add_argument("--output-dir", default="./demo_results")
    parser.add_argument("--pdf-dir", default="./demo_contracts", help="куда положить демо-PDF для Этапа 5А")
    args = parser.parse_args()
    build(Path(args.output_dir))
    build_pdfs(Path(args.pdf_dir))
    print("Теперь: python integrity_check.py --input", args.pdf_dir,
          "--output", str(Path(args.output_dir) / "integrity_check.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
