"""lego_dash_core.py — pure logic ของ dashboard (ไม่มี streamlit/firebase I/O)

แยกจาก streamlit_app.py เพื่อให้ test ได้ตรง ๆ:
  - rows_to_df       : RTDB dict -> DataFrame เฉพาะ committed==True เรียง version (fail closed)
  - order_columns    : บังคับลำดับสัญญา 17 คอลัมน์ (RTDB ไม่การันตีลำดับ key) + meta ต่อท้าย
  - integrity_report : ตรวจสมการ LEGO ต่อแถวจากค่า full precision (E1–E8)
  - recompute_gated_ledger : derive recurrence ใหม่ตามหลักการแช่แข็ง ก่อนแสดงผลเสมอ

นโยบายเดียวทั้งไฟล์ (gated_theoretical_v2):
  - act เมื่อ **signal = 1 และ สถานะ READY_BUY/READY_SELL และ จำนวนสั่ง > 0** เท่านั้น
  - ทุกกรณีอื่นแช่แข็ง: ΔAₙ = 0, Aₙ ค้าง, P_acted ค้าง
  - ไม่แช่แข็ง Rₙ เด็ดขาด — Rₙ = FIX_C × ln(Pₙ/P₀) ตามราคาตลาดทุกแถว
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from dna_engine import decode_dna

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
META_COLS = ["run_id", "chain_key", "version", "committed", "semantics",
             "market_slot_id", "market_ordinal", "clock_mode"]

# mirror ของ lego_state.CASHFLOW_SEMANTICS — ledger ทฤษฎีแบบ gated (ตาม gated demo):
#   ledger คีย์ที่ "เทรดจริงหรือไม่" (การตัดสินใจ) ไม่ใช่ DNA signal ดิบ:
#   act (เทรดจริง READY_BUY/READY_SELL, signal = 1, จำนวนสั่ง > 0):
#       ΔAₙ = FIX_C×(Pₙ/P_acted − 1) โดย P_acted = ราคาแถว act ล่าสุด
#   pass (ไม่เทรด): ΔAₙ = 0, Aₙ ค้าง, P_acted แช่แข็ง, Eₙ smooth
#       *รวม PASS_DNA_ZERO (signal=0) และ PASS_THRESHOLD (signal=1 แต่ |gap| ≤ DIFF):
#        ทั้งคู่ไม่ยิง order จึงต้องแช่แข็ง ledger เหมือนกัน (ΔAₙ = 0)
GATED_SEMANTICS = "gated_theoretical_v2"
# semantics เก่า (ก่อน v2): ΔAₙ = กำไร realized จากรอบ Buy↔Sell ที่จับคู่ปิด
LEGACY_REALIZED_SEMANTICS = "cycle_realized_v1"

# คอลัมน์เงิน 7 ตัว (6, 12–17) — round 2dp เฉพาะตอนแสดง (ตรง columns_presented ฝั่ง engine)
MONEY_COLS = ["ราคา Pₙ (USD)", "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)",
              "Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)",
              "Eₙ ส่วนเกินสะสม (USD)"]
LEDGER_COLS = ["Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)",
               "Eₙ ส่วนเกินสะสม (USD)"]
# คอลัมน์ที่ recompute ต้องมีครบ มิฉะนั้นไม่แตะ (fail safe)
RECOMPUTE_REQUIRED = ["ราคา Pₙ (USD)", "สถานะ", "DNA step", "มูลค่าพอร์ต (USD)",
                      "ส่วนต่างเป้าหมาย (USD)"] + LEDGER_COLS
# คอลัมน์ตัดสินใจที่ทำให้ตัดสิน act/pass ได้เข้มตามนโยบาย
POLICY_COLS = ["สถานะ", "DNA signal", "จำนวนสั่ง (หุ้น)"]

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


def has_policy_columns(df: pd.DataFrame) -> bool:
    """ตัดสิน act/pass แบบเข้มได้ก็ต่อเมื่อมีคอลัมน์ตัดสินใจครบทั้งสาม"""
    return all(col in df.columns for col in POLICY_COLS)


def _traded_flags(df: pd.DataFrame) -> list[bool]:
    """act = เทรดจริงเท่านั้น: signal = 1 **และ** READY_BUY/READY_SELL **และ** qty > 0

    signal 0, PASS ทุกชนิด (รวม PASS_THRESHOLD ที่ signal=1 แต่ |gap| ≤ DIFF), สถานะแปลก
    และแถว READY ที่ผิดรูป (qty = 0 หรือ signal = 0) ล้วนเป็น pass ต้องแช่แข็ง ledger
    frame เก่าที่ไม่มีคอลัมน์ตัดสินใจครบ -> ถอยไปดูสถานะอย่างเดียว (พฤติกรรมรุ่นก่อน)
    """
    ready = df["สถานะ"].astype(str).isin([READY_BUY, READY_SELL])
    if not has_policy_columns(df):
        return ready.tolist()
    return (ready
            & df["DNA signal"].astype(int).eq(1)
            & df["จำนวนสั่ง (หุ้น)"].astype(float).gt(0)).tolist()


def _gated_flags(df: pd.DataFrame) -> list[bool]:
    """แถวไหนอยู่ใต้ semantics gated_theoretical_v2 (แถวเก่าห้ามคำนวณด้วยสมการนี้)"""
    if "semantics" not in df.columns:
        return [True] * len(df)
    return df["semantics"].astype(str).eq(GATED_SEMANTICS).tolist()


def _reference_context(df: pd.DataFrame, p0: float | None) -> tuple[float, float] | None:
    """หา (P₀, FIX_C) สำหรับเส้นอ้างอิง โดยไม่พึ่งสถานะการเทรดเลย"""
    required = {"ราคา Pₙ (USD)", "มูลค่าพอร์ต (USD)", "ส่วนต่างเป้าหมาย (USD)"}
    if df.empty or not required.issubset(df.columns):
        return None

    resolved_p0 = p0
    if resolved_p0 is None and "DNA step" in df.columns:
        try:
            if int(df["DNA step"].iloc[0]) == 0:
                resolved_p0 = float(df["ราคา Pₙ (USD)"].iloc[0])
        except (TypeError, ValueError):
            return None
    if resolved_p0 is None:
        return None

    resolved_p0 = float(resolved_p0)
    fix_c = float(df["มูลค่าพอร์ต (USD)"].astype(float).iloc[0]
                  + df["ส่วนต่างเป้าหมาย (USD)"].astype(float).iloc[0])
    if not np.isfinite(resolved_p0) or resolved_p0 <= 0:
        return None
    if not np.isfinite(fix_c):
        return None
    return resolved_p0, fix_c


def _apply_reference_column(out: pd.DataFrame, source: pd.DataFrame,
                            p0: float | None) -> pd.DataFrame:
    """Rₙ คือเส้นอ้างอิงตลาด จึงต้องขยับทั้งแถว act และแถว pass ที่แช่แข็ง"""
    if "Rₙ อ้างอิง (USD)" not in out.columns:
        return out
    context = _reference_context(source, p0)
    if context is None:
        return out

    resolved_p0, fix_c = context
    prices = source["ราคา Pₙ (USD)"].astype(float)
    if bool((~np.isfinite(prices) | prices.le(0)).any()):
        return out

    reference = fix_c * np.log(prices / resolved_p0)
    gated = pd.Series(_gated_flags(source), index=source.index)
    out.loc[gated.to_numpy(), "Rₙ อ้างอิง (USD)"] = reference.loc[gated].to_numpy()
    return out


def integrity_report(df: pd.DataFrame, p0_hint: float | None = None,
                     tol: float = 1e-6) -> tuple[pd.DataFrame, bool]:
    """ตรวจสมการ LEGO กับแถว committed ของ chain เดียว (เรียง version แล้ว)

    ใช้ค่า full precision จาก RTDB (ห้ามใช้ค่า round 2dp)
    residual ผ่านเมื่อ <= tol × max(1, FIX_C)

      E1  FIX_C คงที่:   Vₙ + gapₙ = FIX_C ทุกแถว   (นิยาม gap = FIX_C − Vₙ)
      E2  มูลค่าพอร์ต:    Vₙ = holdingsₙ × Pₙ
      E3  อ้างอิง:        Rₙ = FIX_C × ln(Pₙ / P₀)
      E4  ต่อสเต็ป (gated): act (signal=1 + READY + qty>0) ->
                         ΔAₙ = FIX_C × (Pₙ / P_acted − 1)
                         โดย P_acted = ราคาแถว act ล่าสุด (reconstruct จาก decision series)
                         pass (ทุกกรณีอื่น) -> ΔAₙ = 0 (แช่แข็ง)
                         แถว semantics เก่า (legacy/cycle_realized_v1) -> ข้ามพร้อมหมายเหตุ
      E5  สะสม:          Aₙ = Aₙ₋₁ + ΔAₙ — ข้ามเฉพาะรอยต่อเปลี่ยน semantics
                         (baseline Aₙ รีเซ็ตเป็น 0)
      E6  ส่วนเกิน (smooth): act -> Eₙ = Aₙ − Rₙ ;
                         pass -> Eₙ = Aₙ − FIX_C × ln(P_acted / P₀)
      E7  โครงสร้าง:      step เพิ่มตาม market slot (มี market_ordinal -> Δstep = Δordinal ≥ 1;
                         ไม่มี -> step +1 แบบเดิม), version +1 ทุกแถว, signal ∈ {0,1}
      E8  decision:      signal=1 + READY + qty>0 เท่านั้นที่เทรด; ทุกกรณีอื่นแช่แข็ง
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
    traded = _traded_flags(df)

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
    gated = _gated_flags(df) if "semantics" in df.columns else [False] * len(df)

    # reconstruct P_acted (ราคาแถว act ล่าสุด) จาก decision series:
    # act -> เลื่อนเป็น Pₙ ; pass -> แช่แข็ง
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
        elif traded[i]:
            acted = float(p.iloc[i])            # act เฉพาะแถวที่เทรดจริง

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
        if traded[i]:
            if pa is None:
                skip4 += 1                       # ไม่รู้ P_acted ก่อนหน้า (ตัดหน้า chain)
            else:
                resid4.append(abs(float(dA.iloc[i]) - fix_c * (float(p.iloc[i]) / pa - 1.0)))
            resid6.append(abs(float(E.iloc[i]) - (float(A.iloc[i]) - float(R.iloc[i]))))
        else:
            resid4.append(abs(float(dA.iloc[i])))          # pass (ไม่เทรด) -> ΔAₙ = 0 แช่แข็ง
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

    # DNA เดินตาม market slot: ปกติ scheduler ไม่พลาด -> step +1 เหมือนเดิมทุกประการ
    # แต่ถ้าพลาด slot step ต้องกระโดดเท่ากับ market_ordinal ที่ข้ามไป (ห้ามย้อน/ซ้ำ)
    ok_step, note_step = True, ""
    if len(df) > 1:
        step_delta = step.diff().iloc[1:]
        if "market_ordinal" in df.columns and df["market_ordinal"].notna().all():
            slot_delta = df["market_ordinal"].astype(int).diff().iloc[1:]
            ok_step = bool((step_delta == slot_delta).all() and (slot_delta >= 1).all())
            skipped = int((slot_delta > 1).sum())
            if ok_step and skipped:
                note_step = f"ข้าม {skipped} ช่วง (scheduler พลาด slot — DNA เดินตามเวลาตลาด)"
        else:
            ok_step = bool((step_delta == 1).all())   # แถวเก่าที่ไม่มี slot provenance
    ok_ver = True
    if "version" in df.columns and len(df) > 1:
        ok_ver = bool((df["version"].astype(int).diff().iloc[1:] == 1).all())
    ok_sig = bool(sig.isin([0, 1]).all())
    ok7 = ok_step and ok_ver and ok_sig
    add("E7", "step ตาม market slot / version +1 / signal ∈ {0,1}", None, ok7,
        note_step if ok7 else "ลำดับ step/version ขาด หรือ signal นอก {0,1}")

    # E8 ใช้กฎเดียวกับ ledger: READY ที่ signal=0 หรือ qty=0 คือแถวผิด ไม่ใช่แถว act
    ready_buy = status.eq(READY_BUY)
    ready_sell = status.eq(READY_SELL)
    ready = ready_buy | ready_sell
    valid_trade = ((ready_buy & sig.eq(1) & gap.gt(0) & qty.gt(0))
                   | (ready_sell & sig.eq(1) & gap.lt(0) & qty.gt(0)))
    valid_frozen = ~ready & qty.eq(0)
    bad = int((~(valid_trade | valid_frozen)).sum())
    add("E8", "signal=1 + READY + qty>0 เท่านั้นที่เทรด; ทุกกรณีอื่นแช่แข็ง", None, bad == 0,
        "" if bad == 0 else f"ผิด {bad} แถว")

    report = pd.DataFrame(checks, columns=["ข้อ", "สมการ/กฎ", "residual สูงสุด", "ผ่าน", "หมายเหตุ"])
    return report, bool(report["ผ่าน"].all())


def recompute_gated_ledger(df: pd.DataFrame, p0: float | None = None) -> pd.DataFrame:
    """คำนวณคอลัมน์ recurrence (Rₙ, ΔAₙ, Aₙ, Eₙ) ใหม่จาก input ที่เชื่อถือได้
    (ราคา Pₙ + การตัดสินใจ + P₀) ตามหลักการ "บัญชีแบบแช่แข็ง"

    ใช้เพราะ dashboard เป็น read-only: ถ้า engine เขียน recurrence ผิด (เช่นแถว PASS
    ที่ยัง act, ΔAₙ ≠ 0) เราจะ "ไม่เชื่อ" ค่าที่เก็บมา แต่ derive ใหม่จากราคา+การตัดสินใจ
    เพื่อให้ตาราง/กราฟที่ "แสดง" ถูกต้องตามหลักการเสมอ

    act (signal=1 + READY_BUY/READY_SELL + qty>0): ΔAₙ = FIX_C×(Pₙ/P_acted−1),
        เลื่อน P_acted = Pₙ, Eₙ = Aₙ − Rₙ
    pass (ทุกกรณีอื่น): ΔAₙ = 0, Aₙ ค้าง, P_acted แช่แข็ง, Eₙ = Aₙ − FIX_C×ln(P_acted/P₀)
    แถว semantics เก่า (legacy/cycle_realized_v1) -> คงค่าเดิม (ไม่แตะ)

    Rₙ ไม่เคยแช่แข็ง: recompute จากราคาตลาดทุกแถว gated ไม่ว่า act หรือ pass
    เป็น no-op กับ chain ที่ค่าถูกอยู่แล้ว (residual 0) จึงปลอดภัยจะเรียกก่อนแสดงผลเสมอ
    คอลัมน์ตัดสินใจไม่ถูกเขียนทับ — dashboard ห้ามแก้หลักฐานที่ commit ไปแล้ว
    """
    if df.empty or any(c not in df.columns for c in RECOMPUTE_REQUIRED):
        # fail safe: ไม่ครบคอลัมน์ -> ไม่คำนวณ ledger เลย
        # คืน copy เสมอ: dashboard เป็น read-only จึงห้ามเขียนทับ frame ของผู้เรียก
        return _apply_reference_column(df.copy(), df, p0)

    source = df.reset_index(drop=True)
    out = source.copy()
    p = out["ราคา Pₙ (USD)"].astype(float)
    fix_c = float((out["มูลค่าพอร์ต (USD)"].astype(float)
                   + out["ส่วนต่างเป้าหมาย (USD)"].astype(float)).iloc[0])
    step = out["DNA step"].astype(int)
    genesis = bool(step.iloc[0] == 0)
    if p0 is None and genesis:
        p0 = float(p.iloc[0])                          # แถว genesis: P₀ = ราคาแถวแรก
    can_ref = p0 is not None and p0 > 0
    gated = _gated_flags(out)
    traded = _traded_flags(out)

    R = out["Rₙ อ้างอิง (USD)"].astype(float).tolist()
    dA = out["ΔAₙ ต่อสเต็ป (USD)"].astype(float).tolist()
    A = out["Aₙ สะสม (USD)"].astype(float).tolist()
    E = out["Eₙ ส่วนเกินสะสม (USD)"].astype(float).tolist()

    acted: float | None = None
    A_prev = 0.0
    for i in range(len(out)):
        Pi = float(p.iloc[i])
        if i == 0:
            if genesis:                               # anchor เริ่มต้น: ทุกค่า = 0
                R[i] = dA[i] = A[i] = E[i] = 0.0
                acted, A_prev = Pi, 0.0
            else:                                     # chain ตัดหน้ามา -> เชื่อ anchor เดิม
                acted, A_prev = Pi, float(A[i])
            continue
        if not gated[i]:                              # legacy: คงค่าเดิม, เลื่อน pointer ทุกแถว
            acted, A_prev = Pi, float(A[i])
            continue
        if acted is None:                             # ไม่มี genesis ในชุด -> ใช้แถวก่อนหน้า
            acted, A_prev = float(p.iloc[i - 1]), float(A[i - 1])
        if can_ref:
            R[i] = fix_c * math.log(Pi / p0)
        if traded[i]:                                 # act: ก้าวหนึ่งก้อนจาก P_acted
            d = fix_c * (Pi / acted - 1.0)
            A_prev += d
            dA[i], A[i], E[i] = d, A_prev, A_prev - R[i]
            acted = Pi
        else:                                         # pass: แช่แข็ง
            dA[i], A[i] = 0.0, A_prev
            if can_ref:
                E[i] = A_prev - fix_c * math.log(acted / p0)

    out["Rₙ อ้างอิง (USD)"] = R
    out["ΔAₙ ต่อสเต็ป (USD)"] = dA
    out["Aₙ สะสม (USD)"] = A
    out["Eₙ ส่วนเกินสะสม (USD)"] = E
    return _apply_reference_column(out, source, p0)


def count_ledger_corrections(stored: pd.DataFrame, fixed: pd.DataFrame,
                             tol: float = 1e-6) -> int:
    """นับแถวที่ recompute_gated_ledger แก้ค่า recurrence (stored ≠ fixed)

    รวม Rₙ ด้วย เพราะ Rₙ ต้องไม่เคยถูกแช่แข็ง — engine ที่เขียน Rₙ ค้างไว้ก็คือแถวที่ผิด
    ใช้เตือนบน dashboard ว่า engine เขียน ledger ผิดกี่แถว (ควรไปแก้ engine ต้นทาง)
    """
    if stored.empty or any(c not in stored.columns or c not in fixed.columns
                           for c in LEDGER_COLS):
        return 0
    a = stored.reset_index(drop=True)[LEDGER_COLS].astype(float)
    b = fixed.reset_index(drop=True)[LEDGER_COLS].astype(float)
    return int(((a - b).abs().max(axis=1) > tol).sum())


# ============================================================================
# Rebalancing 101 — gated demo (DNA gate + บัญชีแบบแช่แข็ง / frozen ledger)
# ----------------------------------------------------------------------------
# playground เชิงสอน: สุ่มราคาแล้วเดิน ledger ทฤษฎีแบบ gated_theoretical_v2 เดียวกับ
# engine เพื่อเห็นว่า "รอบที่ DNA signal=1 แต่ตัดสินใจ PASS (จำนวนสั่ง = 0)" ถูกบันทึก
# ในบัญชีแบบแช่แข็งอย่างไร — holdings แช่ตั้งแต่ act ล่าสุด, ΔAₙ = 0, Eₙ ค้าง (smooth)
# สูตรตรงกับ gated_rebalancing_cashflow_from_prices ของ Webull_Dashboard/manual_tools.py
# และ compute_recurrence ของ lego-firebase/lego_one_row.py
# ============================================================================

MAX_REBALANCING_STEPS = 2000
DEFAULT_GATED_DNA = "26021034252903219354832053493"


def simulate_rebalancing_prices(p0: float, vol: float, drift: float,
                                steps: int, seed: int) -> list[float]:
    """เส้นราคาสุ่มของ Testing Lab (geometric Brownian ต่อรอบ) — deterministic

    ``Pᵢ = Pᵢ₋₁ × exp((drift − vol²/2) + vol × Z)`` โดย ``Z`` จาก default_rng(seed)
    floor ที่ ``P₀ × 1e-8`` เพื่อให้ ln(Pᵢ/P₀) นิยามได้เสมอ
    """
    if not all(math.isfinite(float(x)) for x in (p0, vol, drift)):
        raise ValueError("p0, vol, and drift must be finite")
    if p0 <= 0:
        raise ValueError("p0 must be greater than 0")
    if vol < 0:
        raise ValueError("vol cannot be negative")
    if steps < 2 or steps > MAX_REBALANCING_STEPS:
        raise ValueError(f"steps must be between 2 and {MAX_REBALANCING_STEPS}")

    rng = np.random.default_rng(int(seed))
    prices: list[float] = []
    price = float(p0)
    for _ in range(int(steps)):
        shock = (drift - 0.5 * vol * vol) + vol * float(rng.standard_normal())
        price = max(price * math.exp(shock), p0 * 1e-8)
        prices.append(price)
    return prices


def rebalancing_cashflow_from_prices(prices: Iterable[float], fix_c: float,
                                     p0: float) -> list[dict[str, float]]:
    """เส้น pass-all (act ทุกรอบ) — baseline เทียบกับ gated

    Step 0 anchor ที่ P₀ ; ทุกราคา ``Pᵢ``: ``ΔAᵢ = Fix_c × (Pᵢ/Pᵢ₋₁ − 1)``,
    ``Rₙ = Fix_c × ln(Pₙ/P₀)``, ``Eₙ = Aₙ − Rₙ``
    (เท่ากับ gated ที่ actions เป็น 1 ทุกตัว)
    """
    if not math.isfinite(float(fix_c)) or not math.isfinite(float(p0)):
        raise ValueError("fix_c and p0 must be finite")
    if fix_c <= 0 or p0 <= 0:
        raise ValueError("fix_c and p0 must be greater than 0")

    rows: list[dict[str, float]] = [{
        "step": 0, "price": float(p0), "delta_actual": 0.0,
        "actual_cumulative": 0.0, "ln_reference": 0.0, "excess": 0.0,
    }]
    previous = float(p0)
    actual = 0.0
    for step, raw_price in enumerate(prices, start=1):
        price = float(raw_price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Every price must be finite and greater than 0")
        delta = fix_c * (price / previous - 1.0)
        actual += delta
        reference = fix_c * math.log(price / p0)
        rows.append({
            "step": step, "price": price, "delta_actual": float(delta),
            "actual_cumulative": float(actual), "ln_reference": float(reference),
            "excess": float(actual - reference),
        })
        previous = price
    return rows


def gated_rebalancing_cashflow_from_prices(prices: Iterable[float], fix_c: float,
                                           p0: float,
                                           actions: Iterable[int]
                                           ) -> list[dict[str, float]]:
    """Gated demo ledger (gated_theoretical_v2) — เหมือน lego-firebase engine

    ``actions[i] ∈ {0,1}`` คือ gate ของรอบ ``i+1`` (รอบ 0 = จุด anchor เสมอ):
      act (1):  ``ΔAᵢ = Fix_c × (Pᵢ/P_acted − 1)`` โดย ``P_acted`` = ราคารอบ act
                ล่าสุด แล้วเลื่อน ``P_acted = Pᵢ`` ; ``Eᵢ = Aᵢ − Rᵢ``
      pass (0): ``ΔAᵢ = 0``, ``Aᵢ`` ค้าง, ``P_acted`` แช่แข็ง และ
                ``Eᵢ = Aᵢ − Fix_c × ln(P_acted/P₀)`` (smooth — ค้างค่า act ล่าสุด)

    เหตุผลเศรษฐศาสตร์: ช่วง pass ไม่มีการ rebalance — holdings แช่แข็งตั้งแต่
    act ล่าสุด กำไรจริงจึงเป็นก้อนเดียวเทียบราคาแช่แข็งตอน act ใหม่
    คุณสมบัติ: ``Eₙ`` ไม่ลด และ ≥ 0 เสมอ (จาก ``x − 1 ≥ ln x`` ต่อ segment)
    """
    if not math.isfinite(float(fix_c)) or not math.isfinite(float(p0)):
        raise ValueError("fix_c and p0 must be finite")
    if fix_c <= 0 or p0 <= 0:
        raise ValueError("fix_c and p0 must be greater than 0")

    rows: list[dict[str, float]] = [{
        "step": 0, "action": 1, "price": float(p0), "acted_price": float(p0),
        "delta_actual": 0.0, "actual_cumulative": 0.0, "ln_reference": 0.0,
        "excess": 0.0,
    }]
    acted = float(p0)
    actual = 0.0
    for step, (raw_price, raw_action) in enumerate(zip(prices, actions), start=1):
        price = float(raw_price)
        action = int(raw_action)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Every price must be finite and greater than 0")
        if action not in (0, 1):
            raise ValueError("Every action must be 0 or 1")
        reference = fix_c * math.log(price / p0)
        if action == 1:
            delta = fix_c * (price / acted - 1.0)
            actual += delta
            acted = price
            excess = actual - reference
        else:
            delta = 0.0
            excess = actual - fix_c * math.log(acted / p0)
        rows.append({
            "step": step, "action": action, "price": price,
            "acted_price": float(acted), "delta_actual": float(delta),
            "actual_cumulative": float(actual), "ln_reference": float(reference),
            "excess": float(excess),
        })
    return rows


def build_gate_actions(dna_code: str, n_rounds: int) -> list[int]:
    """decode DNA -> gate array 0/1 แล้วตัด/วนให้ยาวเท่า n_rounds (รอบ act ต่อ demo)

    ยึด decode เดียวกับ engine (dna_engine.decode_dna) — gate เพี้ยนไม่ได้
    DNA สั้นกว่าจำนวนรอบ -> วนซ้ำ (เฉพาะ demo; engine จริง step +1 จน DNA หมดแล้ว fail)
    """
    if n_rounds < 0:
        raise ValueError("n_rounds ต้อง >= 0")
    gate = [int(x) for x in decode_dna(dna_code.strip())]
    if not gate:
        raise ValueError("decode_dna คืน array ว่าง")
    if len(gate) < n_rounds:
        reps = -(-n_rounds // len(gate))       # ceil division
        gate = gate * reps
    return gate[:n_rounds]
