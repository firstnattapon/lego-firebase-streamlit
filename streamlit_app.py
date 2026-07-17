"""streamlit_app.py — Dashboard (read-only) อ่านจาก Firebase RTDB

deploy บน streamlit.app: ใส่ service account JSON + DB URL ใน st.secrets
กรอง committed==True เท่านั้น (orphan/pending ไม่แสดง) + เลือก chain ก่อน plot
"""
from __future__ import annotations

import json

import firebase_admin
import pandas as pd
import streamlit as st
from firebase_admin import credentials, db

ROWS_PATH = "webull_lego_rows"
STATE_PATH = "webull_lego_state"
AUDIT_PATH = "webull_lego_order_audit"

MONEY_COLS = ["ราคา Pₙ (USD)", "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)",
              "Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)",
              "Eₙ ส่วนเกินสะสม (USD)"]


@st.cache_resource
def _init():
    if not firebase_admin._apps:
        cred = credentials.Certificate(json.loads(st.secrets["FIREBASE_SA_JSON"]))
        firebase_admin.initialize_app(cred, {"databaseURL": st.secrets["FIREBASE_DB_URL"]})
    return True


def load_rows() -> pd.DataFrame:
    data = db.reference(ROWS_PATH).get() or {}
    df = pd.DataFrame(list(data.values()))
    if df.empty:
        return df
    if "committed" in df:
        df = df[df["committed"] == True]          # noqa: E712 — กรอง orphan/pending
    if "version" in df:
        df = df.sort_values("version")
    return df


def main():
    st.set_page_config(page_title="LEGO Shannon Demon", layout="wide")
    st.title("🧬 LEGO Shannon Demon — Live Dashboard")
    _init()

    state = db.reference(STATE_PATH).get() or {}
    if state:
        st.subheader("State pointer (anchor ปัจจุบัน)")
        st.dataframe(pd.DataFrame(state).T, use_container_width=True)

    df = load_rows()
    if df.empty:
        st.info("ยังไม่มีแถว committed — รอ Cloud Function รอบแรก")
        return

    # เลือก chain ก่อน — กัน step ซ้ำข้าม chain ปนกราฟ
    chains = sorted(df["chain_key"].dropna().unique()) if "chain_key" in df else []
    if len(chains) > 1:
        selected = st.selectbox("Chain", chains, index=len(chains) - 1)
        df = df[df["chain_key"] == selected]
    elif chains:
        st.caption(f"Chain: {chains[0]}")

    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("สถานะล่าสุด", latest.get("สถานะ", "—"))
    c2.metric("DNA step", int(latest.get("DNA step", 0)))
    c3.metric("Pₙ (USD)", f"{float(latest.get('ราคา Pₙ (USD)', 0)):.2f}")
    c4.metric("Eₙ สะสม (USD)", f"{float(latest.get('Eₙ ส่วนเกินสะสม (USD)', 0)):.2f}")

    st.subheader("Recurrence: Rₙ (อ้างอิง) vs Aₙ (จริง) vs Eₙ (ส่วนเกิน)")
    chart_df = df.set_index("DNA step")[[
        "Rₙ อ้างอิง (USD)", "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"]].astype(float)
    st.line_chart(chart_df)

    st.subheader("ตาราง 17 คอลัมน์ (round 2dp)")
    show = df.copy()
    for col in MONEY_COLS:
        if col in show:
            show[col] = show[col].astype(float).round(2)
    st.dataframe(show, use_container_width=True)

    audit = db.reference(AUDIT_PATH).get() or {}
    if audit:
        st.subheader("Order audit (redacted)")
        st.dataframe(pd.DataFrame(list(audit.values())), use_container_width=True)


if __name__ == "__main__":
    main()
