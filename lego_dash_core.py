"""lego_dash_core.py — pure logic ของ dashboard (ไม่มี streamlit/firebase I/O)

แยกจาก streamlit_app.py เพื่อให้ test ได้ตรง ๆ:
  - rows_to_df       : RTDB dict -> DataFrame เฉพาะ committed==True เรียง version (fail closed)
  - order_columns    : บังคับลำดับสัญญา 17 คอลัมน์ (RTDB ไม่การันตีลำดับ key) + meta ต่อท้าย
  - integrity_report : ตรวจสมการ LEGO ต่อแถวจากค่า full precision (E1–E8)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# mirror ของ lego_one_row.COLUMN_ORDER (สัญญา 17 คอลัมน์ ลำดับตายตัว)
COLUMN_ORDER = [
    "เวลา (UTC)",             # 1
    "สินทรัพย์",              # 2
    "สถานะ",                  # 3
    "DNA step",              # 4
    "DNA signal",            # 5
    "ราคา Pₙ (USD)",          # 6
    "จำนวนถือครอง (หุ้น)",    # 7
    "คำสั่ง",                 # 8
    "ฝั่ง",                   # 9
    "เหตุผล",                 # 10
    "จำนวนสั่ง (หุ้น)",        # 11
    "มูลค่าพอร์ต (USD)",      # 12
    "ส่วนต่างเป้าหมาย (USD)",  # 13
    "Rₙ อ้างอิง (USD)",       # 14
    "ΔAₙ ต่อสเต็ป (USD)",     # 15
    "Aₙ สะสม (USD)",          # 16
    "Eₙ ส่วนเกินสะสม (USD)",  # 17
]
META_COLS = ["run_id", "chain_key", "version", "committed", "semantics"]

# mirror ของ lego_state.CASHFLOW_SEMANTICS — ledger ทฤษฎีแบบ gated (ตาม gated demo):
#   act (signal=1):  ΔAₙ = FIX_C×(Pₙ/P_acted − 1) โดย P_acted = ราคาแถว act ล่าสุด
#   pass (signal=0): ΔAₙ = 0, Aₙ ค้าง, P_acted แช่แข็ง, Eₙ smooth (ค้างค่า act ล่าสุด)
GATED_SEMANTICS = "gated_theoretical_v2"
# semantics เก่า (ก่อน v2): ΔAₙ = กำไร realized จากรอบ Buy↔Sell ที่จับคู่ปิด
LEGACY_REALIZED_SEMANTICS = "cycle_realized_v1"

# คอลัมน์เงิน 7 ตัว (6, 12–17) — round 2dp เฉพาะตอนแสดง (ตรง columns_presented ฝั่ง engine)
MONEY_COLS = ["ราคา Pₙ (USD)", "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)",
              "Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)",
              "Eₙ ส่วนเกินสะสม (USD)"]

PASS_DNA_ZERO = "PASS_DNA_ZERO"
PASS_THRESHOLD = "PASS_THRESHOLD"
READY_BUY = "READY_BUY"
READY_SELL = "READY_SELL"


def rows_to_df(data) -> pd.DataFrame:
    """RTDB payload -> DataFrame เฉพาะแถว committed==True เรียงตาม version

    fail closed: ไม่มีข้อมูล / ไม่มี flag committed -> DataFrame ว่าง
    (orphan/pending ต้องไม่โผล่บน dashboard ตามสัญญา Step 18)
    """
    if not data:
        return pd.DataFrame()
    items = list(data.values()) if isinstance(data, dict) else [x for x in data if x]
    df = pd.DataFrame(items)
    if df.empty or "committed" not in df.columns:
        return pd.DataFrame()
    df = df[df["committed"] == True]  # noqa: E712 — กรอง orphan/pending
    if df.empty:
        return pd.DataFrame()
    if "version" in df.columns:
        df = df.sort_values("version")
    return df.reset_index(drop=True)


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """RTDB คืน key ไม่การันตีลำดับ -> บังคับ 17 คอลัมน์ตามสัญญา แล้วต่อ meta/อื่น ๆ"""
    cols = [c for c in COLUMN_ORDER if c in df.columns]
    cols += [c for c in META_COLS if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def default_chain_index(chains: list, state: dict | None) -> int:
    """chain ที่ active ล่าสุดตาม state[ck].updated_at (ISO string เทียบอักษร = ลำดับเวลา)
    ไม่มีข้อมูล state/updated_at -> ตัวสุดท้ายของ list (พฤติกรรมเดิม)"""
    if not chains:
        return 0
    if not isinstance(state, dict) or not state:
        return len(chains) - 1
    stamps = {ck: str((state.get(ck) or {}).get("updated_at", "")) for ck in chains}
    latest = max(chains, key=lambda ck: stamps[ck])
    return chains.index(latest) if stamps[latest] else len(chains) - 1


def filter_audit_rows(audit: dict | None, run_ids) -> pd.DataFrame:
    """audit เฉพาะ order ของ chain ที่เลือก — ผูกด้วย run_id (audit 1 รายการ/แถว)
    payload เก่าที่ไม่มี run_id เลย -> คืนทั้งหมด (ไม่ตัดข้อมูลที่กรองไม่ได้)"""
    if not audit:
        return pd.DataFrame()
    adf = pd.DataFrame([v for v in audit.values() if isinstance(v, dict)])
    if adf.empty or "run_id" not in adf.columns:
        return adf
    return adf[adf["run_id"].isin(set(run_ids))].reset_index(drop=True)


def _max_abs(s: pd.Series) -> float:
    s = s.dropna()
    return 0.0 if s.empty else float(s.abs().max())


def integrity_report(df: pd.DataFrame, p0_hint: float | None = None,
                     tol: float = 1e-6) -> tuple[pd.DataFrame, bool]:
    """ตรวจสมการ LEGO กับแถว committed ของ chain เดียว (เรียง version แล้ว)

    ใช้ค่า full precision จาก RTDB (ห้ามใช้ค่า round 2dp)
    residual ผ่านเมื่อ <= tol × max(1, FIX_C)

      E1  FIX_C คงที่:   Vₙ + gapₙ = FIX_C ทุกแถว   (นิยาม gap = FIX_C − Vₙ)
      E2  มูลค่าพอร์ต:    Vₙ = holdingsₙ × Pₙ
      E3  อ้างอิง:        Rₙ = FIX_C × ln(Pₙ / P₀)
      E4  ต่อสเต็ป (gated): act (signal=1) -> ΔAₙ = FIX_C × (Pₙ / P_acted − 1)
                         โดย P_acted = ราคาแถว act ล่าสุด (reconstruct จาก signal series)
                         pass (signal=0) -> ΔAₙ = 0
                         แถว semantics เก่า (legacy/cycle_realized_v1) -> ข้ามพร้อมหมายเหตุ
      E5  สะสม:          Aₙ = Aₙ₋₁ + ΔAₙ — ข้ามเฉพาะรอยต่อเปลี่ยน semantics
                         (baseline Aₙ รีเซ็ตเป็น 0)
      E6  ส่วนเกิน (smooth): act -> Eₙ = Aₙ − Rₙ ;
                         pass -> Eₙ = Aₙ − FIX_C × ln(P_acted / P₀)
      E7  โครงสร้าง:      step +1 ทุกแถว, version +1 ทุกแถว, signal ∈ {0,1}
      E8  decision:      signal=0 -> PASS_DNA_ZERO ; READY_BUY -> gap>0, qty>0 ;
                         READY_SELL -> gap<0, qty>0 ; PASS_* -> qty=0
    """
    p = df["ราคา Pₙ (USD)"].astype(float)
    h = df["จำนวนถือครอง (หุ้น)"].astype(float)
    v = df["มูลค่าพอร์ต (USD)"].astype(float)
    gap = df["ส่วนต่างเป้าหมาย (USD)"].astype(float)
    R = df["Rₙ อ้างอิง (USD)"].astype(float)
    dA = df["ΔAₙ ต่อสเต็ป (USD)"].astype(float)
    A = df["Aₙ สะสม (USD)"].astype(float)
    E = df["Eₙ ส่วนเกินสะสม (USD)"].astype(float)
    step = df["DNA step"].astype(int)
    sig = df["DNA signal"].astype(int)
    qty = df["จำนวนสั่ง (หุ้น)"].astype(float)
    status = df["สถานะ"].astype(str)

    fixc_series = v + gap
    fix_c = float(fixc_series.iloc[0])
    scale = tol * max(1.0, abs(fix_c))
    genesis = bool(step.iloc[0] == 0)

    p0 = p0_hint
    if p0 is None and genesis:
        p0 = float(p.iloc[0])   # แถว genesis: P₀ = P ของแถวแรก

    checks: list[tuple[str, str, float | None, bool, str]] = []

    def add(cid: str, eq: str, residual: float | None, ok: bool, note: str = ""):
        checks.append((cid, eq, residual, ok, note))

    r1 = _max_abs(fixc_series - fix_c)
    add("E1", "Vₙ + gapₙ = FIX_C (คงที่)", r1, r1 <= scale, f"FIX_C ≈ {fix_c:.2f}")

    r2 = _max_abs(v - h * p)
    add("E2", "Vₙ = holdingsₙ × Pₙ", r2, r2 <= scale)

    if p0 is not None and p0 > 0:
        r3 = _max_abs(R - fix_c * np.log(p / p0))
        add("E3", "Rₙ = FIX_C × ln(Pₙ/P₀)", r3, r3 <= scale, f"P₀ = {p0}")
    else:
        add("E3", "Rₙ = FIX_C × ln(Pₙ/P₀)", None, True, "ข้าม — ไม่รู้ P₀ (ไม่มีแถว genesis/state)")

    # แถว semantics เก่า (legacy price-formula / cycle_realized_v1): ตรวจสมการ gated ไม่ได้
    if "semantics" in df.columns:
        gated = (df["semantics"].astype(str) == GATED_SEMANTICS).tolist()
    else:
        gated = [False] * len(df)

    # reconstruct P_acted (ราคาแถว act ล่าสุด) จาก signal series:
    # act (signal=1) -> เลื่อนเป็น Pₙ ; pass (signal=0) -> แช่แข็ง
    # prev_acted[i] = P_acted "ก่อน" แถว i (None = ยังไม่รู้ เช่นตัดหน้า chain มา)
    n = len(df)
    prev_acted: list[float | None] = [None] * n
    acted: float | None = None
    for i in range(n):
        prev_acted[i] = acted
        if i == 0 and genesis:
            acted = float(p.iloc[0])            # genesis: P_acted เริ่ม = P₀
        elif not gated[i]:
            acted = float(p.iloc[i])            # semantics เก่า: pointer เลื่อนทุกแถว
        elif int(sig.iloc[i]) == 1:
            acted = float(p.iloc[i])

    resid4, skip4 = [], 0
    resid6, skip6 = [], 0
    for i in range(n):
        if i == 0 and genesis:
            continue                             # แถว genesis เช็คใน E6 (ทุกค่า = 0)
        if not gated[i]:
            skip4 += 1
            skip6 += 1
            continue
        pa = prev_acted[i]
        if int(sig.iloc[i]) == 1:
            if pa is None:
                skip4 += 1                       # ไม่รู้ P_acted ก่อนหน้า (ตัดหน้า chain)
            else:
                resid4.append(abs(float(dA.iloc[i]) - fix_c * (float(p.iloc[i]) / pa - 1.0)))
            resid6.append(abs(float(E.iloc[i]) - (float(A.iloc[i]) - float(R.iloc[i]))))
        else:
            resid4.append(abs(float(dA.iloc[i])))          # pass -> ΔAₙ = 0
            if pa is None or p0 is None or p0 <= 0:
                skip6 += 1                       # smooth E ต้องรู้ P_acted และ P₀
            else:
                resid6.append(abs(float(E.iloc[i])
                                  - (float(A.iloc[i]) - fix_c * math.log(pa / p0))))

    r4 = max(resid4) if resid4 else None
    note4 = "" if skip4 == 0 else f"ข้าม {skip4} แถว (semantics เก่า/ไม่รู้ P_acted)"
    add("E4", "gated: act ΔAₙ = FIX_C × (Pₙ/P_acted − 1) ; pass ΔAₙ = 0",
        r4, (r4 is None) or r4 <= scale, note4 or ("ไม่มีแถวตรวจได้" if r4 is None else ""))

    if n > 1:
        gs = pd.Series(gated, index=df.index)
        boundary = gs.ne(gs.shift(1))                   # รอยต่อเปลี่ยน semantics (สองทิศ)
        boundary.iloc[0] = False
        resid5 = A - (A.shift(1) + dA)
        r5 = _max_abs(resid5[~boundary])
        note5 = ("" if int(boundary.sum()) == 0 else
                 "ข้ามรอยต่อเปลี่ยน semantics (baseline Aₙ รีเซ็ตเป็น 0)")
        add("E5", "Aₙ = Aₙ₋₁ + ΔAₙ", r5, r5 <= scale, note5)
    else:
        add("E5", "Aₙ = Aₙ₋₁ + ΔAₙ", None, True, "แถวเดียว — ไม่มีคู่เทียบ")

    r6 = max(resid6) if resid6 else 0.0
    if genesis:
        r6 = max(r6, abs(R.iloc[0]), abs(dA.iloc[0]), abs(A.iloc[0]), abs(E.iloc[0]))
        note6 = "รวมเช็คแถวแรก R₀=ΔA₀=A₀=E₀=0"
    else:
        note6 = "" if skip6 == 0 else f"ข้าม {skip6} แถว (semantics เก่า/ไม่รู้ P_acted/P₀)"
    add("E6", "smooth: act Eₙ = Aₙ − Rₙ ; pass Eₙ = Aₙ − FIX_C × ln(P_acted/P₀)",
        r6, r6 <= scale, note6)

    ok_step = bool((step.diff().iloc[1:] == 1).all()) if len(df) > 1 else True
    ok_ver = True
    if "version" in df.columns and len(df) > 1:
        ok_ver = bool((df["version"].astype(int).diff().iloc[1:] == 1).all())
    ok_sig = bool(sig.isin([0, 1]).all())
    add("E7", "step +1 / version +1 / signal ∈ {0,1}", None,
        ok_step and ok_ver and ok_sig,
        "" if (ok_step and ok_ver and ok_sig) else "ลำดับ step/version ขาด หรือ signal นอก {0,1}")

    bad = 0
    bad += int(((sig == 0) & ((status != PASS_DNA_ZERO) | (qty != 0))).sum())
    bad += int(((status == READY_BUY) & ~((gap > 0) & (qty > 0))).sum())
    bad += int(((status == READY_SELL) & ~((gap < 0) & (qty > 0))).sum())
    bad += int((status.isin([PASS_DNA_ZERO, PASS_THRESHOLD]) & (qty != 0)).sum())
    add("E8", "gate/decision สอดคล้องสถานะ", None, bad == 0,
        "" if bad == 0 else f"ผิด {bad} แถว")

    report = pd.DataFrame(checks, columns=["ข้อ", "สมการ/กฎ", "residual สูงสุด", "ผ่าน", "หมายเหตุ"])
    return report, bool(report["ผ่าน"].all())
