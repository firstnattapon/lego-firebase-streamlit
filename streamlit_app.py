"""streamlit_app.py — Dashboard (read-only) อ่านจาก Firebase RTDB + Rebalancing 101

2 แท็บ:
  📊 Live Dashboard   — อ่าน committed rows จาก Firebase (เหมือนเดิม), กรอง
                        committed==True, เลือก chain, ตาราง 17 คอลัมน์, integrity E1–E8
  🎓 Rebalancing 101  — playground เชิงสอน "gated demo (DNA gate + บัญชีแบบแช่แข็ง)"
                        ไม่ต้องมี Firebase: สุ่มราคา -> เดิน ledger gated_theoretical_v2
                        เห็นว่ารอบ "จำนวนสั่ง = 0" (pass/แช่แข็ง) ถูกบันทึกอย่างไร

deploy บน streamlit.app: ใส่ service account JSON + DB URL ใน st.secrets (เฉพาะแท็บ Live)
"""
from __future__ import annotations

import json

import firebase_admin
import pandas as pd
import streamlit as st
from firebase_admin import credentials, db

from dna_engine import DNAError
from lego_dash_core import (DEFAULT_GATED_DNA, MAX_REBALANCING_STEPS, MONEY_COLS,
                            build_gate_actions, count_ledger_corrections,
                            default_chain_index, filter_audit_rows,
                            gated_rebalancing_cashflow_from_prices,
                            integrity_report, order_columns,
                            rebalancing_cashflow_from_prices,
                            recompute_gated_ledger, rows_to_df,
                            simulate_rebalancing_prices)

ROWS_PATH = "webull_lego_rows"
STATE_PATH = "webull_lego_state"
AUDIT_PATH = "webull_lego_order_audit"


def _has_secret(key: str) -> bool:
    try:                      # ไม่มีไฟล์ secrets เลย -> st.secrets จะ raise ไม่ใช่คืน False
        return key in st.secrets
    except Exception:
        return False


def _secrets_ready() -> bool:
    return _has_secret("FIREBASE_SA_JSON") and _has_secret("FIREBASE_DB_URL")


@st.cache_resource
def _init():
    if not firebase_admin._apps:
        cred = credentials.Certificate(json.loads(st.secrets["FIREBASE_SA_JSON"]))
        firebase_admin.initialize_app(cred, {"databaseURL": st.secrets["FIREBASE_DB_URL"]})
    return True


def load_rows() -> pd.DataFrame:
    return rows_to_df(db.reference(ROWS_PATH).get())


def render_live_dashboard() -> None:
    # แท็บนี้ต้องมี Firebase — ไม่มี secrets ก็แค่แจ้ง แล้วไม่ล้มทั้งหน้า (แท็บ 101 ยังใช้ได้)
    if not _secrets_ready():
        st.info("ตั้ง st.secrets: FIREBASE_SA_JSON และ FIREBASE_DB_URL ก่อน จึงจะเห็น "
                "ข้อมูลสด — ระหว่างนี้เปิดแท็บ 🎓 Rebalancing 101 เพื่อลอง gated demo ได้เลย")
        return
    _init()

    state = db.reference(STATE_PATH).get() or {}
    if state:
        st.subheader("State pointer (anchor ปัจจุบัน)")
        st.dataframe(pd.DataFrame(state).T, width="stretch")

    df = load_rows()
    if df.empty:
        st.info("ยังไม่มีแถว committed — รอ Cloud Function รอบแรก")
        return

    # เลือก chain ก่อน — กัน step ซ้ำข้าม chain ปนกราฟ
    chains = sorted(df["chain_key"].dropna().unique()) if "chain_key" in df else []
    selected = chains[0] if chains else None
    if len(chains) > 1:
        # default = chain ที่ active ล่าสุดตาม state.updated_at (ไม่ใช่เรียงอักษร)
        selected = st.selectbox("Chain", chains,
                                index=default_chain_index(chains, state))
        df = df[df["chain_key"] == selected]
    elif chains:
        st.caption(f"Chain: {chains[0]}")

    # P₀ จาก state pointer (chain ตัดหน้า) — genesis จะ fallback = ราคาแถวแรกใน recompute
    p0_hint = None
    if selected and isinstance(state, dict):
        raw = (state.get(selected) or {}).get("p0")
        p0_hint = float(raw) if raw is not None else None

    # dashboard เป็น read-only: ไม่เชื่อค่า recurrence ที่ engine เขียน (อาจมี PASS ที่ยัง
    # act ผิด) → derive ledger ใหม่ตามหลักการแช่แข็ง จากราคา+การตัดสินใจ ก่อนแสดงเสมอ
    df_fixed = recompute_gated_ledger(df, p0=p0_hint)
    n_corr = count_ledger_corrections(df, df_fixed)
    if n_corr:
        st.warning(
            f"⚠️ engine เขียน ledger ผิด {n_corr} แถว (แถว PASS ที่ยัง act — ΔAₙ ≠ 0) "
            "— ตาราง/กราฟด้านล่างถูก **คำนวณใหม่ตามหลักการแช่แข็ง** (ทุก PASS ทั้ง "
            "PASS_DNA_ZERO และ PASS_THRESHOLD → ΔAₙ = 0, Aₙ ค้าง) ควรแก้ engine ต้นทางด้วย"
        )
    df = df_fixed

    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("สถานะล่าสุด", latest.get("สถานะ", "—"))
    c2.metric("DNA step", int(latest.get("DNA step", 0)))
    c3.metric("Pₙ (USD)", f"{float(latest.get('ราคา Pₙ (USD)', 0)):.2f}")
    c4.metric("Eₙ สะสม (USD)", f"{float(latest.get('Eₙ ส่วนเกินสะสม (USD)', 0)):.2f}")

    st.subheader("Recurrence: Rₙ (อ้างอิง) vs Aₙ (gated ledger) vs Eₙ (smooth)")
    st.caption(
        "gated_theoretical_v2: แถว act (เทรดจริง READY_BUY/SELL) ΔAₙ = FIX_C×(Pₙ/P_acted−1) "
        "เทียบราคาแถว act ล่าสุด · แถว pass (ไม่เทรด — PASS_DNA_ZERO หรือ PASS_THRESHOLD, "
        "จำนวนสั่ง = 0) ΔAₙ = 0, Aₙ ค้าง, Eₙ ค้างค่า act ล่าสุด (smooth) — Eₙ ไม่ลดและ ≥ 0 เสมอ"
    )
    chart_df = df.set_index("DNA step")[[
        "Rₙ อ้างอิง (USD)", "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"]].astype(float)
    st.line_chart(chart_df)

    st.subheader("ตาราง 17 คอลัมน์ (round 2dp)")
    show = order_columns(df.copy())
    for col in MONEY_COLS:
        if col in show:
            show[col] = show[col].astype(float).round(2)
    st.dataframe(show, width="stretch")

    # integrity: ตรวจจากค่า full precision ที่ "แสดงจริง" (recomputed) — ต้องผ่านเสมอ
    with st.expander("🔎 Integrity check — สมการ LEGO (E1–E8)", expanded=False):
        report, ok = integrity_report(df, p0_hint=p0_hint)
        st.dataframe(report, width="stretch")
        if ok:
            st.success("ทุกสมการผ่าน — ค่าที่แสดง (คำนวณใหม่) สอดคล้อง LEGO invariant"
                       + (f" · แก้จาก engine {n_corr} แถว" if n_corr else ""))
        else:
            st.error("พบแถวที่ไม่สอดคล้องสมการ — ตรวจ chain/engine ก่อนเชื่อกราฟ")

    # audit เฉพาะ chain ที่เลือก — ผูกด้วย run_id กัน order ข้าม chain ปนตาราง
    audit = db.reference(AUDIT_PATH).get() or {}
    if audit:
        run_ids = df["run_id"] if "run_id" in df.columns else []
        adf = filter_audit_rows(audit, run_ids)
        if not adf.empty:
            st.subheader("Order audit (redacted)")
            st.dataframe(adf, width="stretch")


def render_rebalancing_101() -> None:
    st.subheader("🧬 Gated demo — DNA gate + บัญชีแบบแช่แข็ง (gated_theoretical_v2)")
    st.caption(
        "ตรงกับ ledger จริงของ lego-firebase — ledger คีย์ที่ **การเทรดจริง (การตัดสินใจ)** "
        "ไม่ใช่ DNA signal ดิบ: **ทุกแถวรวมแถวจำนวนสั่ง = 0** commit ภายใต้ semantics "
        "gated_theoretical_v2 (“บัญชีแบบแช่แข็ง”) ไม่มีแถวไหนถูกทิ้ง"
    )
    st.markdown(
        "- **act (READY_BUY/READY_SELL, จำนวนสั่ง > 0 — เทรดจริง):** ΔAₙ = Fix_c × "
        "(Pₙ/P_acted − 1), เลื่อน P_acted = Pₙ, Eₙ = Aₙ − Rₙ\n"
        "- **pass (ไม่เทรด — จำนวนสั่ง = 0):** ΔAₙ = 0, holdings แช่แข็งตั้งแต่ act ล่าสุด, "
        "Aₙ ค้าง, P_acted แช่แข็ง, Eₙ smooth (ค้างค่า act ล่าสุด) — **รวมทั้ง PASS_DNA_ZERO "
        "(signal = 0) และ PASS_THRESHOLD (signal = 1 แต่ |gap| ≤ DIFF)**: ทั้งสองไม่ยิง order "
        "จึงแช่แข็ง ledger เหมือนกัน (ΔAₙ = 0) — Eₙ ไม่ลดและ ≥ 0 เสมอ (x − 1 ≥ ln x)"
    )

    pc = st.columns(2)
    with pc[0]:
        st.markdown("#### 1) เส้นอ้างอิงทางทฤษฎี")
        st.code("Rₙ = Fix_c × ln(Pₙ / P₀)", language=None)
        st.caption("กระแสเงินสดอ้างอิงของการรักษามูลค่าคงที่แบบต่อเนื่อง (act ทุกรอบ)")
    with pc[1]:
        st.markdown("#### 2) เส้น gated จริง (แช่แข็งช่วง pass)")
        st.code("act:  ΔAₙ = Fix_c × (Pₙ/P_acted − 1),  Eₙ = Aₙ − Rₙ\n"
                "pass: ΔAₙ = 0 (แช่แข็ง),  Eₙ = Aₙ − Fix_c × ln(P_acted/P₀)",
                language=None)
        st.caption("P_acted = ราคารอบ act ล่าสุด — ค้างไว้ตลอดช่วง pass")

    st.markdown("#### Testing Lab — สุ่มราคา แล้วเดิน ledger ตาม DNA gate")
    if "dash_gated_seed" not in st.session_state:
        st.session_state.dash_gated_seed = 101

    r1 = st.columns(3)
    fix_c = r1[0].number_input("Fix_c (มูลค่าเป้าหมาย)", min_value=0.01,
                               value=1500.0, step=100.0, format="%.2f")
    p0 = r1[1].number_input("ราคาเริ่มต้น P₀", min_value=0.01, value=100.0,
                            format="%.5f")
    vol = r1[2].number_input("ความผันผวน/รอบ (%)", min_value=0.0, max_value=40.0,
                             value=4.0, step=0.1)
    r2 = st.columns(3)
    drift = r2[0].number_input("แนวโน้ม/รอบ (%)", min_value=-10.0, max_value=10.0,
                               value=0.0, step=0.01)
    steps = r2[1].number_input("จำนวนรอบ", min_value=2,
                               max_value=MAX_REBALANCING_STEPS, value=100, step=1)
    seed = r2[2].number_input("Seed", min_value=0, step=1, key="dash_gated_seed")
    dna_code = st.text_input(
        "DNA code (gate 0/1 ต่อรอบ — decode เดียวกับ engine: bypass:N / [1,N] / stream)",
        value=DEFAULT_GATED_DNA, key="dash_gated_dna")

    try:
        prices = simulate_rebalancing_prices(float(p0), float(vol) / 100.0,
                                             float(drift) / 100.0, int(steps),
                                             int(seed))
        actions = build_gate_actions(dna_code, len(prices))
        gated_rows = gated_rebalancing_cashflow_from_prices(
            prices, float(fix_c), float(p0), actions)
        pass_rows = rebalancing_cashflow_from_prices(prices, float(fix_c), float(p0))
        g_final, p_final = gated_rows[-1], pass_rows[-1]

        n_pass = len(actions) - sum(actions)
        m = st.columns(4)
        # เดโมนี้ไม่มีชั้น threshold — DNA gate จึงเท่ากับการตัดสินใจเทรด (act=เทรด, pass=แช่แข็ง)
        m[0].metric("act (เทรด) / pass (แช่แข็ง)", f"{sum(actions)} / {n_pass}")
        m[1].metric("Gated Aₙ", f"{g_final['actual_cumulative']:+,.2f}")
        m[2].metric("Gated Eₙ (smooth)", f"{g_final['excess']:+,.2f}")
        m[3].metric("Δ เทียบ pass-all Aₙ",
                    f"{g_final['actual_cumulative'] - p_final['actual_cumulative']:+,.2f}")

        st.markdown("#### กราฟ — Rₙ (อ้างอิง) vs Aₙ (gated) vs Eₙ (smooth)")
        chart_df = pd.DataFrame(gated_rows).set_index("step")[[
            "ln_reference", "actual_cumulative", "excess"]]
        chart_df.columns = ["Rₙ อ้างอิง", "Aₙ (gated)", "Eₙ (smooth)"]
        st.line_chart(chart_df)

        st.markdown("#### ตาราง ledger (gated)")
        gated_frame = pd.DataFrame(gated_rows)
        st.dataframe(
            gated_frame.rename(columns={
                "step": "รอบ", "action": "gate", "price": "ราคา Pᵢ",
                "acted_price": "P_acted (แช่แข็ง)", "delta_actual": "ΔA รอบนี้",
                "actual_cumulative": "Aₙ", "ln_reference": "Rₙ",
                "excess": "Eₙ (smooth)",
            }),
            width="stretch", hide_index=True,
        )
        st.download_button("ดาวน์โหลด CSV (gated)",
                           data=gated_frame.to_csv(index=False),
                           file_name="rebalancing_gated_demo.csv",
                           mime="text/csv")
        st.info(
            "ข้อสังเกต: gated ไม่ได้ดีกว่า pass-all เสมอ — ผลต่างมาจาก cross-terms "
            "ของช่วงที่ข้าม (ตลาด trend การข้ามชนะ, ตลาด mean-revert การ act ทุกรอบชนะ) "
            "DNA gate จึงต้องมาจากการคัดเลือกด้วย backtest/GA ไม่ใช่ข้ามมั่ว"
        )
        st.caption("แบบจำลองเพื่อการเรียนรู้ · ยังไม่รวม spread, slippage และภาษี")
    except (DNAError, ValueError) as exc:
        st.error(f"คำนวณไม่ได้: {exc}")


def main():
    st.set_page_config(page_title="LEGO Shannon Demon", layout="wide")
    st.title("🧬 LEGO Shannon Demon")

    tab_live, tab_101 = st.tabs(["📊 Live Dashboard", "🎓 Rebalancing 101"])
    with tab_live:
        render_live_dashboard()
    with tab_101:
        render_rebalancing_101()


if __name__ == "__main__":
    main()
