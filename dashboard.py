#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аудит государственных закупок Казахстана.
Этап 5 из 5 — Streamlit-дашборд: сводка результатов Этапов 1–4 для аудитора.

Вход (папка --results, по умолчанию ./results; если пусто — ./demo_results):
    contracts.json                — Этап 1
    alternatives.json             — Этап 2
    price_benchmark.json,
    category_stats.json           — Этап 3
    tor_compliance.json,
    fragmentation_clusters.json   — Этап 4
Отсутствующие файлы не блокируют дашборд: соответствующий раздел показывает подсказку.

Ключ стыковки — source_file. Всё, что показано, — индикаторы для проверки человеком.

Запуск:
    streamlit run dashboard.py                       # ./results, иначе ./demo_results
    streamlit run dashboard.py -- --results ./out    # своя папка
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Порог «сводного» риска и веса компонентов (в сумме 100). Каждый компонент — вклад
# соответствующего этапа; отсутствие данных по этапу даёт 0, а не штраф.
W_PRICE, W_COMPLIANCE, W_FRAGMENTATION = 45, 30, 25
OVERALL_HIGH = 60
OVERALL_CHECK = 30

PRICE_RISK_SCORE = {"высокий риск": 1.0, "требует проверки": 0.5, "норма": 0.0, "недостаточно данных": 0.0}
COMPLIANCE_SCORE = {"высокий риск несоответствия": 1.0, "требует проверки": 0.5, "соответствует": 0.0,
                    "недостаточно данных для проверки": 0.0}
FRAG_SCORE = {"высокий": 1.0, "средний": 0.5, "низкий": 0.2}

RISK_COLORS = {"высокий риск": "#c0392b", "требует проверки": "#e67e22", "норма": "#27ae60",
               "недостаточно данных": "#7f8c8d"}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def parse_results_dir() -> Path:
    """`streamlit run dashboard.py -- --results DIR`; без аргумента — ./results или ./demo_results."""
    argv = sys.argv[1:]
    if "--results" in argv:
        return Path(argv[argv.index("--results") + 1])
    for candidate in (Path("./results"), Path("./demo_results")):
        if (candidate / "contracts.json").exists():
            return candidate
    return Path("./results")


@st.cache_data(show_spinner=False)
def load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_all(results_dir: Path) -> dict:
    names = ["contracts", "alternatives", "price_benchmark", "category_stats",
             "tor_compliance", "fragmentation_clusters"]
    return {n: load_json(str(results_dir / f"{n}.json")) for n in names}


# ---------------------------------------------------------------------------
# Сводная таблица по договорам (стыковка по source_file)
# ---------------------------------------------------------------------------

def fmt_kzt(v) -> str:
    try:
        return f"{int(round(float(v))):,}".replace(",", " ") + " ₸"
    except (TypeError, ValueError):
        return "—"


def fmt_mln(v) -> str:
    """Компактный формат для метрик (единица «млн ₸» — в подписи)."""
    try:
        return f"{float(v) / 1_000_000:.1f}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def build_overview(data: dict) -> pd.DataFrame:
    contracts = data["contracts"] or []
    alts = {r["source_file"]: r for r in (data["alternatives"] or [])}
    bench = {r["source_file"]: r for r in (data["price_benchmark"] or [])}
    comp = {r["source_file"]: r for r in (data["tor_compliance"] or [])}
    frag = {}
    for cl in (data["fragmentation_clusters"] or []):
        for sf in cl["source_files"]:
            # договор может входить в несколько кластеров — берём самый подозрительный
            if sf not in frag or cl["suspicion_score"] > frag[sf]["suspicion_score"]:
                frag[sf] = cl

    rows = []
    for c in contracts:
        sf = c["source_file"]
        b, a, k, f = bench.get(sf), alts.get(sf), comp.get(sf), frag.get(sf)
        price_risk = b["risk_level"] if b else "недостаточно данных"
        comp_level = k["compliance_level"] if k else "недостаточно данных для проверки"
        frag_level = f["suspicion_level"] if f else ""
        score = (W_PRICE * PRICE_RISK_SCORE.get(price_risk, 0)
                 + W_COMPLIANCE * COMPLIANCE_SCORE.get(comp_level, 0)
                 + W_FRAGMENTATION * FRAG_SCORE.get(frag_level, 0))
        overall = ("высокий риск" if score >= OVERALL_HIGH else
                   "требует проверки" if score >= OVERALL_CHECK else "норма")
        rows.append({
            "source_file": sf,
            "Договор": c.get("contract_number") or sf,
            "Заказчик": c.get("customer", ""),
            "Поставщик": c.get("supplier", ""),
            "Район": c.get("district", ""),
            "Дата": c.get("contract_date", ""),
            "Сумма, ₸": c.get("amount_kzt", 0) or 0,
            "Категория": b["category"] if b else (a["service_category"] if a else ""),
            "Откл. от медианы, %": b["price_deviation_pct"] if b else None,
            "Ценовой риск": price_risk,
            "Экономия, ₸": a["potential_savings_kzt"] if a else 0,
            "Экономия, %": a["potential_savings_pct"] if a else 0.0,
            "Соответствие ТЗ": comp_level,
            "Балл ТЗ": k["consistency_score"] if k else None,
            "Дробление": f"{f['cluster_id']} ({frag_level})" if f else "",
            "Сводный балл": round(score),
            "Сводный риск": overall,
            "Уверенность извлечения": c.get("confidence", ""),
            "Ошибка извлечения": c.get("error", ""),
        })
    return pd.DataFrame(rows)


def color_risk(val: str) -> str:
    base = {"высокий": "#f8d7da", "требует": "#ffe5cc", "норма": "#d4edda", "соответствует": "#d4edda",
            "средний": "#ffe5cc", "низкий": "#eef2f5"}
    for key, color in base.items():
        if isinstance(val, str) and val.startswith(key):
            return f"background-color: {color}; color: #1f2933"
    return ""


# ---------------------------------------------------------------------------
# Разделы
# ---------------------------------------------------------------------------

def section_summary(df: pd.DataFrame, data: dict, results_dir: Path):
    st.subheader("Сводка")
    total_amount = df["Сумма, ₸"].sum()
    savings = df["Экономия, ₸"].sum()
    n_high = int((df["Сводный риск"] == "высокий риск").sum())
    n_check = int((df["Сводный риск"] == "требует проверки").sum())
    n_frag = len(data["fragmentation_clusters"] or [])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Договоров", len(df))
    c2.metric("Сумма, млн ₸", fmt_mln(total_amount))
    c3.metric("Экономия, млн ₸", fmt_mln(savings),
              f"{savings / total_amount * 100:.1f}% от суммы" if total_amount else None, delta_color="off")
    c4.metric("Риск: высокий / проверка", f"{n_high} / {n_check}")
    c5.metric("Дробление: кластеров", n_frag)

    left, right = st.columns([1, 1])
    with left:
        st.caption("Распределение сводного риска")
        counts = df["Сводный риск"].value_counts().reindex(
            ["высокий риск", "требует проверки", "норма"], fill_value=0)
        st.bar_chart(counts, color="#c0392b", horizontal=True)
    with right:
        st.caption("Сумма договоров по категориям, ₸")
        by_cat = df[df["Категория"] != ""].groupby("Категория")["Сумма, ₸"].sum().sort_values()
        if not by_cat.empty:
            st.bar_chart(by_cat, color="#2c7fb8", horizontal=True)
        else:
            st.info("Категории появятся после Этапа 3 (price_benchmark.py).")

    missing = [n for n, v in data.items() if v is None]
    if missing:
        st.warning("Не найдены файлы: " + ", ".join(f"`{m}.json`" for m in missing)
                   + f" в `{results_dir}/`. Соответствующие разделы показаны частично.")


def section_contracts(df: pd.DataFrame, data: dict):
    st.subheader("Договоры")
    f1, f2, f3 = st.columns(3)
    risk_filter = f1.multiselect("Сводный риск", ["высокий риск", "требует проверки", "норма"],
                                 default=["высокий риск", "требует проверки", "норма"])
    cat_filter = f2.multiselect("Категория", sorted(x for x in df["Категория"].unique() if x))
    dist_filter = f3.multiselect("Район", sorted(x for x in df["Район"].unique() if x))

    view = df[df["Сводный риск"].isin(risk_filter)]
    if cat_filter:
        view = view[view["Категория"].isin(cat_filter)]
    if dist_filter:
        view = view[view["Район"].isin(dist_filter)]
    view = view.sort_values("Сводный балл", ascending=False)

    cols = ["Договор", "Заказчик", "Поставщик", "Сумма, ₸", "Категория", "Откл. от медианы, %",
            "Ценовой риск", "Экономия, ₸", "Соответствие ТЗ", "Дробление", "Сводный балл", "Сводный риск"]
    st.dataframe(
        view[cols].style.map(color_risk, subset=["Ценовой риск", "Соответствие ТЗ", "Сводный риск"])
        .format({"Сумма, ₸": "{:,.0f}", "Экономия, ₸": "{:,.0f}", "Откл. от медианы, %": "{:+.1f}"}),
        width="stretch", hide_index=True, height=min(60 + 36 * len(view), 600),
    )
    st.caption(f"Сводный балл = {W_PRICE}·цена + {W_COMPLIANCE}·ТЗ + {W_FRAGMENTATION}·дробление; "
               f"≥{OVERALL_HIGH} — высокий риск, ≥{OVERALL_CHECK} — требует проверки.")

    # --- карточка договора ---
    st.markdown("#### Карточка договора")
    if view.empty:
        st.info("Нет договоров под выбранные фильтры.")
        return
    options = view["source_file"].tolist()
    labels = {sf: f"{row['Договор']} — {row['Поставщик']} — {fmt_kzt(row['Сумма, ₸'])}"
              for sf, row in zip(view["source_file"], view.to_dict("records"))}
    chosen = st.selectbox("Выберите договор", options, format_func=lambda sf: labels[sf])
    contract_card(chosen, data)


def contract_card(sf: str, data: dict):
    contract = next((c for c in (data["contracts"] or []) if c["source_file"] == sf), None)
    bench = next((r for r in (data["price_benchmark"] or []) if r["source_file"] == sf), None)
    alt = next((r for r in (data["alternatives"] or []) if r["source_file"] == sf), None)
    comp = next((r for r in (data["tor_compliance"] or []) if r["source_file"] == sf), None)
    clusters = [cl for cl in (data["fragmentation_clusters"] or []) if sf in cl["source_files"]]
    if not contract:
        return

    a, b = st.columns([1, 1])
    with a:
        st.markdown(f"**Файл:** `{sf}`  \n**Договор:** {contract.get('contract_number') or '—'}  \n"
                    f"**Лот:** {contract.get('lot_number') or '—'}  \n**Дата:** {contract.get('contract_date') or '—'}  \n"
                    f"**Заказчик:** {contract.get('customer') or '—'}  \n**Поставщик:** {contract.get('supplier') or '—'}  \n"
                    f"**Район:** {contract.get('district') or '—'}  \n**Сумма:** {fmt_kzt(contract.get('amount_kzt'))}  \n"
                    f"**Уверенность извлечения:** {contract.get('confidence') or '—'}")
        st.markdown(f"**Предмет закупки:** {contract.get('service_text') or '—'}")
        if contract.get("tech_stack_mentioned"):
            st.markdown(f"**Технологии:** {contract['tech_stack_mentioned']}")
        if contract.get("error"):
            st.error(f"Ошибка извлечения: {contract['error']}")
    with b:
        if bench:
            color = RISK_COLORS.get(bench["risk_level"], "#7f8c8d")
            st.markdown(f"**Ценовой бенчмаркинг** — "
                        f"<span style='color:{color};font-weight:600'>{bench['risk_level']}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"Категория: *{bench['category'] or '—'}*  \n"
                        f"Медиана категории: {fmt_kzt(bench['category_median_price'])} "
                        f"(выборка {bench['sample_size']})  \n"
                        f"Отклонение: **{bench['price_deviation_pct']:+.1f}%** "
                        f"(×{bench['price_ratio']})")
            if bench.get("benchmark_note"):
                st.caption(bench["benchmark_note"])
            if bench.get("district_context"):
                st.caption(bench["district_context"])
        else:
            st.info("Нет данных Этапа 3 по этому договору.")

        if comp:
            color = {"соответствует": "#27ae60", "требует проверки": "#e67e22",
                     "высокий риск несоответствия": "#c0392b"}.get(comp["compliance_level"], "#7f8c8d")
            score = comp["consistency_score"]
            st.markdown(f"**Соответствие ТЗ** — <span style='color:{color};font-weight:600'>"
                        f"{comp['compliance_level']}</span>"
                        + (f" ({score}/100)" if score is not None else ""), unsafe_allow_html=True)
            st.markdown(comp.get("explanation") or "")
            if comp.get("flagged_phrases"):
                st.markdown("Спорные формулировки: " + ", ".join(f"«{p}»" for p in comp["flagged_phrases"]))
        else:
            st.info("Нет данных Этапа 4 (модуль А) по этому договору.")

    if alt:
        st.markdown(f"**Альтернативные предложения** (рынок: {alt.get('market_source') or '—'}, "
                    f"проверено кандидатов: {alt.get('candidates_checked', 0)})")
        if alt["alternatives"]:
            st.markdown(f"Потенциальная экономия: **{fmt_kzt(alt['potential_savings_kzt'])} "
                        f"({alt['potential_savings_pct']:.1f}%)** — лучшая альтернатива "
                        f"{alt['best_alternative']['company']}")
            adf = pd.DataFrame(alt["alternatives"]).rename(columns={
                "company": "Компания", "city": "Город", "price_kzt": "Цена, ₸", "price_diff_pct": "Дешевле на, %",
                "relevance_reason": "Почему релевантно", "offer": "Предложение"})
            st.dataframe(adf.style.format({"Цена, ₸": "{:,.0f}", "Дешевле на, %": "{:.1f}"}),
                         width="stretch", hide_index=True)
        else:
            st.caption(alt.get("note") or "Более дешёвых альтернатив не найдено.")
        if alt.get("market_source") == "synthetic":
            st.caption("⚠ Рынок синтетический: компании вымышлены и сгенерированы для демо.")

    for cl in clusters:
        st.markdown(f"**Кластер дробления {cl['cluster_id']}** — уровень *{cl['suspicion_level']}* "
                    f"(балл {cl['suspicion_score']}): {cl['explanation']}")


def section_benchmark(df: pd.DataFrame, data: dict):
    st.subheader("Ценовой бенчмаркинг по категориям")
    stats = data["category_stats"]
    bench = data["price_benchmark"]
    if not stats or not bench:
        st.info("Запустите Этап 3: `price_benchmark.py` → `price_benchmark.json`, `category_stats.json`.")
        return

    rows = [{"Категория": name, "Описание": s["description"], "Договоров": s["contracts_count"],
             "Медиана, ₸": s["median_price"], "Мин, ₸": s["min_price"], "Макс, ₸": s["max_price"],
             "Выборка": s["sample_size"], "Высокий риск": s["risk_counts"].get("высокий риск", 0),
             "Требует проверки": s["risk_counts"].get("требует проверки", 0),
             "Норма": s["risk_counts"].get("норма", 0)} for name, s in stats.items()]
    st.dataframe(pd.DataFrame(rows).style.format({"Медиана, ₸": "{:,.0f}", "Мин, ₸": "{:,.0f}",
                                                    "Макс, ₸": "{:,.0f}"}),
                 width="stretch", hide_index=True)

    st.caption("Отклонение цены каждого договора от медианы своей категории, %")
    bdf = pd.DataFrame(bench)
    bdf = bdf[bdf["risk_level"] != "недостаточно данных"].copy()
    if bdf.empty:
        st.info("Нет оценённых договоров.")
        return
    bdf["label"] = bdf["contract_id"].astype(str) + " · " + bdf["supplier"].astype(str).str[:28]
    chart = bdf.set_index("label")[["price_deviation_pct"]].rename(
        columns={"price_deviation_pct": "Отклонение, %"}).sort_values("Отклонение, %")
    st.bar_chart(chart, horizontal=True, color="#e67e22")

    districts = bdf[bdf["district_context"] != ""].drop_duplicates("district_key")["district_context"].tolist()
    if districts:
        st.markdown("**Контекст по районам**")
        for line in districts:
            st.markdown(f"- {line}")


def section_fragmentation(data: dict):
    st.subheader("Признаки дробления закупок")
    clusters = data["fragmentation_clusters"]
    if clusters is None:
        st.info("Запустите Этап 4: `compliance_and_fragmentation.py` → `fragmentation_clusters.json`.")
        return
    if not clusters:
        st.success("Кластеров с признаками дробления не найдено.")
        return
    for cl in sorted(clusters, key=lambda c: -c["suspicion_score"]):
        color = {"высокий": "#c0392b", "средний": "#e67e22", "низкий": "#7f8c8d"}[cl["suspicion_level"]]
        with st.expander(f"{cl['cluster_id']} · {cl['customer']} · {cl['contracts_count']} договора на "
                         f"{fmt_kzt(cl['total_amount_kzt'])} · балл {cl['suspicion_score']} ({cl['suspicion_level']})",
                         expanded=cl["suspicion_level"] == "высокий"):
            st.markdown(f"<span style='color:{color};font-weight:600'>Уровень: {cl['suspicion_level']}</span>  \n"
                        f"Похожесть текстов: {cl['avg_similarity']} · интервал дат: {cl['date_span_days']} дн. · "
                        f"каждый ниже порога {fmt_kzt(cl['threshold_kzt'])}: "
                        f"{'да' if cl['each_below_threshold'] else 'нет'}", unsafe_allow_html=True)
            st.markdown(cl["explanation"])
            cdf = pd.DataFrame(cl["contracts"]).rename(columns={
                "contract_id": "Договор", "contract_date": "Дата", "supplier": "Поставщик",
                "amount_kzt": "Сумма, ₸", "service_text": "Предмет", "source_file": "Файл"})
            st.dataframe(cdf[["Договор", "Дата", "Поставщик", "Сумма, ₸", "Предмет", "Файл"]]
                         .style.format({"Сумма, ₸": "{:,.0f}"}), width="stretch", hide_index=True)


def section_compliance(data: dict):
    st.subheader("Соответствие продукта техническому заданию")
    comp = data["tor_compliance"]
    if comp is None:
        st.info("Запустите Этап 4: `compliance_and_fragmentation.py` → `tor_compliance.json`.")
        return
    cdf = pd.DataFrame(comp)
    counts = cdf["compliance_level"].value_counts()
    st.caption("  ·  ".join(f"{k}: {v}" for k, v in counts.items()))
    view = cdf.rename(columns={"contract_id": "Договор", "supplier": "Поставщик",
                               "tech_stack_mentioned": "Заявлено в ТЗ", "consistency_score": "Балл",
                               "compliance_level": "Уровень", "explanation": "Пояснение"})
    view = view.sort_values("Балл", na_position="last")
    st.dataframe(view[["Договор", "Поставщик", "Заявлено в ТЗ", "Балл", "Уровень", "Пояснение"]]
                 .style.map(color_risk, subset=["Уровень"]), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Аудит госзакупок РК", page_icon="🔎", layout="wide")
    results_dir = parse_results_dir()
    data = load_all(results_dir)

    st.title("AI-аудит государственных закупок")
    st.caption("Индикаторы риска для проверки аудитором — не юридический вердикт. "
               f"Источник данных: `{results_dir}/`"
               + ("  ·  **ДЕМО-ДАННЫЕ (вымышленные)**" if "demo" in str(results_dir) else ""))

    if not data["contracts"]:
        st.error(f"Не найден `{results_dir}/contracts.json`. Запустите Этап 1 (`extract_stage1.py`) "
                 "или сгенерируйте демо: `python demo_data.py --output-dir ./demo_results`.")
        st.stop()

    if st.sidebar.button("Обновить данные"):
        load_json.clear()
        st.rerun()
    st.sidebar.markdown("**Этапы pipeline**")
    for n, label in [("contracts", "1. Извлечение из PDF"), ("alternatives", "2. Альтернативы"),
                     ("price_benchmark", "3. Бенчмаркинг"), ("tor_compliance", "4А. Соответствие ТЗ"),
                     ("fragmentation_clusters", "4Б. Дробление")]:
        st.sidebar.markdown(("✅ " if data[n] is not None else "⬜ ") + label)

    df = build_overview(data)
    section_summary(df, data, results_dir)
    tabs = st.tabs(["Договоры", "Бенчмаркинг", "Дробление", "Соответствие ТЗ"])
    with tabs[0]:
        section_contracts(df, data)
    with tabs[1]:
        section_benchmark(df, data)
    with tabs[2]:
        section_fragmentation(data)
    with tabs[3]:
        section_compliance(data)


main()
