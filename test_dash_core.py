"""test_dash_core.py — ตรวจ pure logic ของ dashboard (ไม่ต้องมี streamlit/firebase)

ครอบคลุม:
  - rows_to_df   : กรอง committed (fail closed), เรียง version, ทน payload แปลก
  - order_columns: บังคับลำดับสัญญา 17 คอลัมน์แม้ key มาสลับ
  - integrity    : chain ที่ถูกสูตร -> ผ่านทุกข้อ; ปนค่าผิด -> ต้องจับได้ (E3–E8)
"""
from __future__ import annotations

import math
import sys

from lego_dash_core import (COLUMN_ORDER, count_ledger_corrections,
                            default_chain_index, filter_audit_rows,
                            integrity_report, order_columns,
                            recompute_gated_ledger, rows_to_df)

FIX_C = 1500.0
DIFF = 60.0


def _mk_chain(prices: list[float], holdings: list[float], signals: list[int]) -> dict:
    """สร้างแถว committed ตามสูตร LEGO gated_theoretical_v2 เป๊ะ คืน dict แบบ RTDB

    ledger คีย์ที่ "เทรดจริงไหม" (การตัดสินใจ) ไม่ใช่ DNA signal ดิบ:
    act (READY_BUY/READY_SELL): ΔA = FIX_C×(Pₙ/P_acted − 1) แล้วเลื่อน P_acted = Pₙ ; E = A − R
    pass (จำนวนสั่ง=0 — PASS_DNA_ZERO/PASS_THRESHOLD): ΔA = 0, A ค้าง, P_acted แช่แข็ง ;
         E = A − FIX_C×ln(P_acted/P₀) — รวมถึง PASS_THRESHOLD (signal=1 แต่ |gap| ≤ DIFF)
    """
    rows = {}
    p0 = prices[0]
    A_prev = 0.0
    acted = p0
    for n, (p, h, sig) in enumerate(zip(prices, holdings, signals)):
        v = h * p
        gap = FIX_C - v

        # ตัดสินใจก่อน แล้วค่อย gate ledger ที่ "เทรดจริงไหม" (ไม่ใช่ DNA signal)
        if sig == 0:
            status, action, side, qty = "PASS_DNA_ZERO", "PASS", "", 0.0
        elif abs(gap) <= DIFF:
            status, action, side, qty = "PASS_THRESHOLD", "PASS", "", 0.0
        elif gap > DIFF:
            status, action, side, qty = "READY_BUY", "TRIGGER_ACTION", "BUY", round(gap / p, 5)
        else:
            status, action, side, qty = "READY_SELL", "TRIGGER_ACTION", "SELL", round(-gap / p, 5)
        traded = status in ("READY_BUY", "READY_SELL")

        if n == 0:
            R = dA = A = E = 0.0
            acted = p
        elif traded:
            R = FIX_C * math.log(p / p0)
            dA = FIX_C * (p / acted - 1.0)
            A = A_prev + dA
            E = A - R
            acted = p
        else:                                     # pass (รวม PASS_THRESHOLD) -> แช่แข็ง
            R = FIX_C * math.log(p / p0)
            dA = 0.0
            A = A_prev
            E = A - FIX_C * math.log(acted / p0)
        A_prev = A

        row = {
            "เวลา (UTC)": f"2026-07-17T14:{n:02d}:00Z",
            "สินทรัพย์": "APLS",
            "สถานะ": status,
            "DNA step": n,
            "DNA signal": sig,
            "ราคา Pₙ (USD)": p,
            "จำนวนถือครอง (หุ้น)": h,
            "คำสั่ง": action,
            "ฝั่ง": side,
            "เหตุผล": status,
            "จำนวนสั่ง (หุ้น)": qty,
            "มูลค่าพอร์ต (USD)": v,
            "ส่วนต่างเป้าหมาย (USD)": gap,
            "Rₙ อ้างอิง (USD)": R,
            "ΔAₙ ต่อสเต็ป (USD)": dA,
            "Aₙ สะสม (USD)": A,
            "Eₙ ส่วนเกินสะสม (USD)": E,
            "run_id": "rid" + f"{n:029d}",
            "chain_key": "APLS_abc123def456",
            "version": n + 1,
            "committed": True,
            "semantics": "gated_theoretical_v2",
        }
        rows[row["run_id"]] = row
    return rows


PRICES = [10.0, 12.0, 11.0, 9.5, 10.5]
HOLD = [150.0, 150.0, 125.0, 136.36364, 157.89474]
SIGNALS = [1, 1, 0, 1, 1]


def test_rows_to_df_filters_and_sorts():
    data = _mk_chain(PRICES, HOLD, SIGNALS)
    # แทรก orphan (committed=False) + pending ไม่มี flag -> ต้องไม่โผล่
    data["orphan" + "0" * 26] = {**list(data.values())[0], "run_id": "orphan",
                                 "committed": False, "version": 99}
    df = rows_to_df(data)
    assert len(df) == 5
    assert list(df["version"]) == [1, 2, 3, 4, 5]
    assert rows_to_df(None).empty and rows_to_df({}).empty
    # ไม่มีคอลัมน์ committed เลย -> fail closed (ว่าง)
    assert rows_to_df({"x": {"a": 1}}).empty


def test_order_columns_contract():
    data = _mk_chain(PRICES[:2], HOLD[:2], SIGNALS[:2])
    # จำลอง RTDB คืน key สลับ (เรียงตามอักษร)
    scrambled = {k: dict(sorted(v.items())) for k, v in data.items()}
    df = order_columns(rows_to_df(scrambled))
    assert list(df.columns)[:17] == COLUMN_ORDER
    assert list(df.columns)[17:21] == ["run_id", "chain_key", "version", "committed"]


def test_integrity_all_pass():
    df = rows_to_df(_mk_chain(PRICES, HOLD, SIGNALS))
    report, ok = integrity_report(df, p0_hint=None)   # P₀ จากแถว genesis
    assert ok, f"ต้องผ่านทุกข้อ:\n{report}"
    report2, ok2 = integrity_report(df, p0_hint=10.0)  # P₀ จาก state pointer
    assert ok2
    # แถวเดียว (genesis) ก็ต้องผ่าน
    r3, ok3 = integrity_report(rows_to_df(_mk_chain(PRICES[:1], HOLD[:1], [1])))
    assert ok3


def test_integrity_catches_corruption():
    base = _mk_chain(PRICES, HOLD, SIGNALS)

    def corrupt(field, value, row_idx=2):
        data = {k: dict(v) for k, v in base.items()}
        key = sorted(data, key=lambda k: data[k]["version"])[row_idx]
        data[key][field] = value
        rep, ok = integrity_report(rows_to_df(data), p0_hint=10.0)
        assert not ok, f"ต้องจับ {field} ผิดได้:\n{rep}"

    corrupt("Rₙ อ้างอิง (USD)", 999.0)          # E3
    corrupt("ΔAₙ ต่อสเต็ป (USD)", 999.0)        # E4/E5/E6
    corrupt("Aₙ สะสม (USD)", 999.0)             # E5/E6
    corrupt("Eₙ ส่วนเกินสะสม (USD)", 999.0)     # E6
    corrupt("มูลค่าพอร์ต (USD)", 999.0)         # E1/E2
    corrupt("DNA step", 7)                      # E7
    corrupt("DNA signal", 5)                    # E7
    corrupt("สถานะ", "READY_BUY", row_idx=2)    # E8 (แถว signal=0)
    corrupt("จำนวนสั่ง (หุ้น)", 3.0, row_idx=2)  # E8 (PASS ต้อง qty=0)
    # แถว pass (row_idx=2, signal=0): ΔA ต้อง 0 และ E ต้อง smooth — จับได้ทั้งคู่
    corrupt("ΔAₙ ต่อสเต็ป (USD)", 5.0, row_idx=2)          # E4 pass ≠ 0
    corrupt("Eₙ ส่วนเกินสะสม (USD)", 123.0, row_idx=2)     # E6 smooth


def test_pass_threshold_signal1_freezes_ledger():
    # หลักการ: PASS_THRESHOLD (signal=1 แต่ |gap| ≤ DIFF, จำนวนสั่ง=0) = ไม่เทรด
    # ledger ต้องแช่แข็งเหมือน PASS_DNA_ZERO: ΔAₙ=0, Aₙ ค้าง, P_acted แช่แข็ง
    prices = [10.0, 12.0, 11.0, 9.0]
    holdings = [150.0, 150.0, 136.36364, 136.36364]
    signals = [1, 1, 1, 1]            # signal=1 ทุกแถว — แต่แถว idx 2 เป็น PASS_THRESHOLD
    data = _mk_chain(prices, holdings, signals)
    df = rows_to_df(data)

    row2 = df.iloc[2]
    assert row2["สถานะ"] == "PASS_THRESHOLD" and int(row2["DNA signal"]) == 1
    assert row2["ΔAₙ ต่อสเต็ป (USD)"] == 0.0                     # แช่แข็ง: ΔA=0
    assert row2["Aₙ สะสม (USD)"] == df.iloc[1]["Aₙ สะสม (USD)"]  # Aₙ ค้างจากแถว act ล่าสุด
    report, ok = integrity_report(df, p0_hint=10.0)
    assert ok, f"PASS_THRESHOLD (signal=1) ที่แช่แข็งถูกต้องต้องผ่าน:\n{report}"

    # ตรงข้าม: ถ้า ledger ยัง "act" บนแถว PASS_THRESHOLD (ΔA≠0 ตามราคา) ต้องถูกจับผิด
    key = sorted(data, key=lambda k: data[k]["version"])[2]
    bad = {k: dict(v) for k, v in data.items()}
    bad[key]["ΔAₙ ต่อสเต็ป (USD)"] = FIX_C * (11.0 / 12.0 - 1.0)   # ค่าแบบ act (ผิด)
    _, ok2 = integrity_report(rows_to_df(bad), p0_hint=10.0)
    assert not ok2, "PASS_THRESHOLD ที่ยัง act (ΔA≠0) ต้องถูกจับ (E4/E5)"


def test_recompute_gated_ledger_freezes_engine_pass_rows():
    # จำลอง engine bug: แถว PASS_THRESHOLD (signal=1) ถูกเขียนแบบ act (ΔA≠0, A เพี้ยน)
    prices = [10.0, 12.0, 11.0, 9.0]
    holdings = [150.0, 150.0, 136.36364, 136.36364]
    data = _mk_chain(prices, holdings, [1, 1, 1, 1])
    df = rows_to_df(data)
    a1 = float(df.iloc[1]["Aₙ สะสม (USD)"])              # A หลังแถว act ล่าสุด (idx 1)
    key = sorted(data, key=lambda k: data[k]["version"])[2]
    bad = {k: dict(v) for k, v in data.items()}
    act_dA = FIX_C * (11.0 / 12.0 - 1.0)
    bad[key]["ΔAₙ ต่อสเต็ป (USD)"] = act_dA               # engine act ผิดบน PASS_THRESHOLD
    bad[key]["Aₙ สะสม (USD)"] = a1 + act_dA
    bad_df = rows_to_df(bad)
    _, ok_bad = integrity_report(bad_df, p0_hint=10.0)
    assert not ok_bad                                    # ค่าดิบจาก engine = ผิด

    fixed = recompute_gated_ledger(bad_df, p0=10.0)
    assert fixed.iloc[2]["ΔAₙ ต่อสเต็ป (USD)"] == 0.0     # แช่แข็ง: ΔA=0
    assert fixed.iloc[2]["Aₙ สะสม (USD)"] == a1          # Aₙ ค้างจาก act ล่าสุด
    _, ok_fixed = integrity_report(fixed, p0_hint=10.0)
    assert ok_fixed, "หลัง recompute ต้องผ่านทุกสมการ"
    assert count_ledger_corrections(bad_df, fixed) == 1  # แก้ 1 แถว


def test_recompute_gated_ledger_noop_on_correct_chain():
    df = rows_to_df(_mk_chain(PRICES, HOLD, SIGNALS))
    fixed = recompute_gated_ledger(df, p0=10.0)
    cols = ["Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)",
            "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"]
    for col in cols:
        for a, b in zip(df[col].astype(float), fixed[col].astype(float)):
            assert abs(a - b) <= 1e-9                    # chain ถูกอยู่แล้ว -> no-op
    assert count_ledger_corrections(df, fixed) == 0
    assert recompute_gated_ledger(rows_to_df(None)).empty  # ว่าง -> ว่าง


def test_recompute_all_pass_signal1_stays_frozen_at_anchor():
    # ตรงกับหน้าจอจริง: ทุกแถว PASS_THRESHOLD (signal=1), ไม่เคยเทรด -> Aₙ=0 ตลอด
    prices, holdings = [321.2, 321.96], [9.3449, 9.3449]
    # บังคับสถานะ PASS_THRESHOLD ทั้งคู่ด้วย FIX_C ใหญ่ (gap เล็ก) — สร้างแถวตรง ๆ
    import math as _m
    rows = {}
    for n, (p, h) in enumerate(zip(prices, holdings)):
        rows[f"r{n:029d}"] = {
            "เวลา (UTC)": f"t{n}", "สินทรัพย์": "AAPL", "สถานะ": "PASS_THRESHOLD",
            "DNA step": n, "DNA signal": 1, "ราคา Pₙ (USD)": p,
            "จำนวนถือครอง (หุ้น)": h, "คำสั่ง": "PASS", "ฝั่ง": "",
            "เหตุผล": "PASS_THRESHOLD", "จำนวนสั่ง (หุ้น)": 0.0,
            "มูลค่าพอร์ต (USD)": h * p, "ส่วนต่างเป้าหมาย (USD)": 3000.0 - h * p,
            "Rₙ อ้างอิง (USD)": 0.0 if n == 0 else 3000.0 * _m.log(p / prices[0]),
            # engine เขียนผิด: act บนแถว 1
            "ΔAₙ ต่อสเต็ป (USD)": 0.0 if n == 0 else 3000.0 * (p / prices[0] - 1.0),
            "Aₙ สะสม (USD)": 0.0 if n == 0 else 3000.0 * (p / prices[0] - 1.0),
            "Eₙ ส่วนเกินสะสม (USD)": 0.0,
            "run_id": f"r{n:029d}", "chain_key": "AAPL_x", "version": n + 1,
            "committed": True, "semantics": "gated_theoretical_v2",
        }
    fixed = recompute_gated_ledger(rows_to_df(rows))       # genesis -> P₀ = ราคาแถวแรก
    assert list(fixed["ΔAₙ ต่อสเต็ป (USD)"].astype(float)) == [0.0, 0.0]
    assert list(fixed["Aₙ สะสม (USD)"].astype(float)) == [0.0, 0.0]
    assert list(fixed["Eₙ ส่วนเกินสะสม (USD)"].astype(float)) == [0.0, 0.0]
    _, ok = integrity_report(fixed)
    assert ok


def test_default_chain_index_latest_by_updated_at():
    chains = ["AAPL_aaa", "APLS_zzz", "TSLA_mmm"]
    state = {
        "AAPL_aaa": {"updated_at": "2026-07-20T15:00:00Z"},
        "APLS_zzz": {"updated_at": "2026-07-01T10:00:00Z"},
        "TSLA_mmm": {"updated_at": "2026-07-19T09:00:00Z"},
    }
    assert default_chain_index(chains, state) == 0            # AAPL ล่าสุด ไม่ใช่เรียงอักษร
    assert default_chain_index(chains, None) == len(chains) - 1   # ไม่มี state -> เดิม
    assert default_chain_index(chains, {}) == len(chains) - 1
    assert default_chain_index([], state) == 0
    # state ไม่มี updated_at เลย -> fallback ตัวสุดท้าย
    assert default_chain_index(chains, {"AAPL_aaa": {}}) == len(chains) - 1


def test_filter_audit_rows_by_chain_run_ids():
    audit = {
        "o1": {"run_id": "r1", "side": "BUY", "status": "FILLED"},
        "o2": {"run_id": "r2", "side": "SELL", "status": "PLACING"},
        "o3": {"run_id": "r9", "side": "BUY", "status": "FILLED"},
        "junk": "not-a-dict",
    }
    adf = filter_audit_rows(audit, ["r1", "r2"])
    assert set(adf["run_id"]) == {"r1", "r2"}
    assert filter_audit_rows(None, ["r1"]).empty
    assert filter_audit_rows({}, ["r1"]).empty
    # payload เก่าไม่มี run_id -> คืนทั้งหมด (ไม่ตัดข้อมูลที่กรองไม่ได้)
    legacy = {"o1": {"side": "BUY"}, "o2": {"side": "SELL"}}
    assert len(filter_audit_rows(legacy, ["r1"])) == 2




# ---- semantics=cycle_realized_v1: ΔAₙ/Aₙ = กำไรรอบที่จับคู่ปิด (บทที่ 4) ----
def _mk_realized_chain(prices: list[float], realized_deltas: list[float],
                       start_version: int = 1, start_step: int = 0,
                       A_start: float = 0.0, p0: float | None = None) -> dict:
    """แถว semantics ใหม่: R จากสูตรราคา (Reference), ΔA = กำไร realized ที่ให้มา"""
    rows = {}
    p0 = prices[0] if p0 is None else p0
    A_prev = A_start
    for n, (p, d) in enumerate(zip(prices, realized_deltas)):
        genesis = (start_version == 1 and n == 0)
        R = 0.0 if genesis else FIX_C * math.log(p / p0)
        dA = 0.0 if genesis else d
        A = A_prev + dA
        E = A - R if not genesis else 0.0
        A_prev = A
        v = 150.0 * p
        row = {
            "เวลา (UTC)": f"2026-07-21T14:{n:02d}:00Z",
            "สินทรัพย์": "APLS", "สถานะ": "PASS_THRESHOLD",
            "DNA step": start_step + n, "DNA signal": 1,
            "ราคา Pₙ (USD)": p, "จำนวนถือครอง (หุ้น)": 150.0,
            "คำสั่ง": "PASS", "ฝั่ง": "", "เหตุผล": "PASS_THRESHOLD",
            "จำนวนสั่ง (หุ้น)": 0.0,
            "มูลค่าพอร์ต (USD)": v, "ส่วนต่างเป้าหมาย (USD)": FIX_C - v,
            "Rₙ อ้างอิง (USD)": R, "ΔAₙ ต่อสเต็ป (USD)": dA,
            "Aₙ สะสม (USD)": A, "Eₙ ส่วนเกินสะสม (USD)": A - R if not genesis else 0.0,
            "run_id": "rlz" + f"{start_version + n:029d}",
            "chain_key": "APLS_abc123def456",
            "version": start_version + n, "committed": True,
            "semantics": "cycle_realized_v1",
        }
        rows[row["run_id"]] = row
    return rows


def test_integrity_legacy_realized_rows_skipped_not_judged_by_gated_formula():
    # แถว semantics เก่า (cycle_realized_v1): ΔA ไม่ตรงสูตร gated — ต้องข้าม ไม่ false-alarm
    data = _mk_realized_chain([10.0, 11.0, 9.0, 10.0], [0.0, 0.0, 0.0, 13.64])
    report, ok = integrity_report(rows_to_df(data), p0_hint=10.0)
    assert ok, f"แถว semantics เก่าต้องถูกข้าม ไม่ถูกตัดสินด้วยสูตร gated:\n{report}"
    e4 = report[report["ข้อ"] == "E4"].iloc[0]
    assert "semantics เก่า" in str(e4["หมายเหตุ"])


def _mk_gated_cont(prices, signals, start_version, start_step,
                   acted0, A_start, p0):
    """แถว gated_theoretical_v2 ต่อท้าย chain เดิม (ไม่ใช่ genesis)

    ทุกแถวเป็น PASS (จำนวนสั่ง=0) จึงแช่แข็ง ledger ทั้งหมด — ไม่ว่า signal=1
    (PASS_THRESHOLD) หรือ signal=0 (PASS_DNA_ZERO): ΔA=0, A ค้าง, P_acted แช่แข็ง
    """
    rows = {}
    A_prev, acted = A_start, acted0
    for n, (p, sig) in enumerate(zip(prices, signals)):
        status = "PASS_DNA_ZERO" if sig == 0 else "PASS_THRESHOLD"
        traded = status in ("READY_BUY", "READY_SELL")   # PASS ทุกแถว -> แช่แข็ง
        R = FIX_C * math.log(p / p0)
        if traded:
            dA = FIX_C * (p / acted - 1.0)
            A = A_prev + dA
            E = A - R
            acted = p
        else:
            dA, A = 0.0, A_prev
            E = A - FIX_C * math.log(acted / p0)
        A_prev = A
        v = 150.0 * p
        row = {
            "เวลา (UTC)": f"2026-07-22T14:{n:02d}:00Z",
            "สินทรัพย์": "APLS",
            "สถานะ": status,
            "DNA step": start_step + n, "DNA signal": sig,
            "ราคา Pₙ (USD)": p, "จำนวนถือครอง (หุ้น)": 150.0,
            "คำสั่ง": "PASS", "ฝั่ง": "",
            "เหตุผล": "PASS_DNA_ZERO" if sig == 0 else "PASS_THRESHOLD",
            "จำนวนสั่ง (หุ้น)": 0.0,
            "มูลค่าพอร์ต (USD)": v, "ส่วนต่างเป้าหมาย (USD)": FIX_C - v,
            "Rₙ อ้างอิง (USD)": R, "ΔAₙ ต่อสเต็ป (USD)": dA,
            "Aₙ สะสม (USD)": A, "Eₙ ส่วนเกินสะสม (USD)": E,
            "run_id": "gat" + f"{start_version + n:029d}",
            "chain_key": "APLS_abc123def456",
            "version": start_version + n, "committed": True,
            "semantics": "gated_theoretical_v2",
        }
        rows[row["run_id"]] = row
    return rows


def test_integrity_mixed_legacy_then_gated_boundary_ok():
    # legacy (cycle_realized) v1–4 แล้ว migrate เป็น gated v5+ (baseline A รีเซ็ต 0,
    # P_acted ตั้งต้น = ราคาแถวเก่าล่าสุด — ตรง read_anchor migration ฝั่ง engine)
    legacy = _mk_realized_chain([10.0, 11.0, 9.0, 10.0], [0.0, 0.0, 0.0, 13.64])
    gated = _mk_gated_cont([10.2, 10.8], [1, 0], start_version=5, start_step=4,
                           acted0=10.0, A_start=0.0, p0=10.0)
    report, ok = integrity_report(rows_to_df({**legacy, **gated}), p0_hint=10.0)
    assert ok, f"รอยต่อ legacy→gated ต้องไม่ false-alarm:\n{report}"
    e5 = report[report["ข้อ"] == "E5"].iloc[0]
    assert "รอยต่อ" in str(e5["หมายเหตุ"])


def test_integrity_still_catches_broken_A_chain_in_realized_rows():
    data = _mk_realized_chain([10.0, 11.0, 9.0, 10.0], [0.0, 0.0, 0.0, 13.64])
    key = sorted(data, key=lambda k: data[k]["version"])[2]
    data[key]["Aₙ สะสม (USD)"] = 777.0                  # ทำลาย chain ภายใน realized
    report, ok = integrity_report(rows_to_df(data), p0_hint=10.0)
    assert not ok, f"E5 ภายในโซน realized ต้องยังจับได้:\n{report}"


def test_order_columns_groups_semantics_with_meta():
    data = _mk_realized_chain([10.0, 11.0], [0.0, 0.0])
    df = order_columns(rows_to_df(data))
    assert list(df.columns)[:17] == COLUMN_ORDER
    assert list(df.columns)[17:22] == ["run_id", "chain_key", "version",
                                       "committed", "semantics"]


# ---- Rebalancing 101: gated demo (DNA gate + frozen ledger) ----------------
# golden จาก gated demo CSV เดียวกับ Webull_Dashboard/manual_tools.py (source of truth)
from lego_dash_core import (build_gate_actions,
                            gated_rebalancing_cashflow_from_prices,
                            rebalancing_cashflow_from_prices,
                            simulate_rebalancing_prices)
from dna_engine import DNAError, decode_dna


def _close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def test_gated_cashflow_matches_gated_demo_csv_goldens():
    prices = [96.81133514, 89.17306809, 91.27812248]
    rows = gated_rebalancing_cashflow_from_prices(prices, 1500.0, 100.0, [1, 1, 0])
    # รอบ 2 act: ΔA เทียบราคา act ก่อนหน้า (แช่ที่ 96.811)
    assert _close(rows[2]["delta_actual"], -118.3477179)
    assert _close(rows[2]["actual_cumulative"], -166.1776908)
    assert _close(rows[2]["excess"], 5.708988114)
    # รอบ 3 pass: A ค้าง, P_acted แช่แข็ง, smooth E ค้างค่า act ล่าสุด
    assert rows[3]["delta_actual"] == 0.0
    assert _close(rows[3]["actual_cumulative"], -166.1776908)
    assert _close(rows[3]["acted_price"], 89.17306809)
    assert _close(rows[3]["excess"], 5.708988114)


def test_gated_cashflow_react_after_freeze_one_big_step():
    # แช่ที่ 89.173 ช่วง pass แล้ว act ที่ 110.725 -> ΔA ก้อนเดียว 362.53
    prices = [96.81133514, 89.17306809, 91.27812248, 93.96129271, 110.7252073]
    rows = gated_rebalancing_cashflow_from_prices(
        prices, 1500.0, 100.0, [1, 1, 0, 0, 1])
    assert _close(rows[5]["delta_actual"], 362.533325)
    assert _close(rows[5]["actual_cumulative"], 196.3556342)
    assert _close(rows[5]["excess"], 43.53362969)


def test_gated_cashflow_excess_monotone_nonnegative():
    prices = simulate_rebalancing_prices(100.0, 0.05, 0.0, 50, 7)
    actions = build_gate_actions("26021034252903219354832053493", len(prices))
    rows = gated_rebalancing_cashflow_from_prices(prices, 1500.0, 100.0, actions)
    excesses = [r["excess"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(excesses, excesses[1:]))
    assert excesses[-1] >= 0.0


def test_gated_cashflow_all_pass_equals_frozen_anchor():
    rows = gated_rebalancing_cashflow_from_prices(
        [120.0, 80.0], 1500.0, 100.0, [0, 0])
    assert all(r["actual_cumulative"] == 0.0 for r in rows)
    assert all(r["acted_price"] == 100.0 for r in rows)
    assert all(r["excess"] == 0.0 for r in rows)          # smooth ค้างค่า anchor


def test_gated_cashflow_validation_fail_closed():
    for bad in ([2], ):                                   # action นอก {0,1}
        try:
            gated_rebalancing_cashflow_from_prices([100.0], 1500.0, 100.0, bad)
            assert False, "action นอก {0,1} ต้อง raise"
        except ValueError:
            pass
    for kwargs in ([0.0], 1500.0, 100.0, [1]), ([100.0], -1.0, 100.0, [1]):
        try:
            gated_rebalancing_cashflow_from_prices(*kwargs)
            assert False, "input ผิดต้อง raise"
        except ValueError:
            pass


def test_pass_all_equals_gated_all_ones():
    prices = [120.0, 80.0, 130.0]
    passed = rebalancing_cashflow_from_prices(prices, 1500.0, 100.0)
    gated = gated_rebalancing_cashflow_from_prices(prices, 1500.0, 100.0, [1, 1, 1])
    for a, b in zip(passed, gated):
        assert _close(a["actual_cumulative"], b["actual_cumulative"])
        assert _close(a["excess"], b["excess"])


def test_build_gate_actions_matches_engine_decode_and_tiles():
    # decode ต้องตรง engine: 60 slot, 25 ones, dna[0]=1, head ตาม champion
    gate = decode_dna("26021034252903219354832053493")
    assert len(gate) == 60 and sum(gate) == 25 and gate[0] == 1
    assert gate[:20] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 1, 0, 0, 1, 1, 0, 1, 1, 1]
    # build_gate_actions: ตัดให้พอดี n, และวนซ้ำเมื่อ DNA สั้นกว่า n
    assert build_gate_actions("bypass:5", 3) == [1, 1, 1]
    tiled = build_gate_actions("26021034252903219354832053493", 100)
    assert len(tiled) == 100
    assert tiled[:60] == gate                             # 60 แรกตรง decode
    assert tiled[60:100] == gate[:40]                     # ที่เหลือวนซ้ำ
    try:
        build_gate_actions("not-a-dna", 10)
        assert False, "DNA ผิดต้อง raise DNAError"
    except DNAError:
        pass


def test_simulate_prices_deterministic_and_positive():
    a = simulate_rebalancing_prices(100.0, 0.04, 0.0, 30, 101)
    b = simulate_rebalancing_prices(100.0, 0.04, 0.0, 30, 101)
    assert a == b and len(a) == 30 and all(p > 0 for p in a)   # deterministic
    c = simulate_rebalancing_prices(100.0, 0.04, 0.0, 30, 202)
    assert a != c                                              # seed ต่าง -> เส้นต่าง


def _with_slots(steps: list[int], ordinals: list[int] | None) -> dict:
    """chain เดิม แต่ให้ DNA step / market_ordinal ตามที่กำหนด (จำลอง scheduler พลาด slot)"""
    data = _mk_chain(PRICES, HOLD, SIGNALS)
    for i, key in enumerate(sorted(data, key=lambda k: data[k]["version"])):
        data[key]["DNA step"] = steps[i]
        if ordinals is not None:
            data[key]["market_ordinal"] = ordinals[i]
            data[key]["market_slot_id"] = f"2026-07-17:{ordinals[i]}"
            data[key]["clock_mode"] = "market"
    return data


def test_e7_accepts_step_jump_that_matches_market_slot():
    # scheduler พลาด slot 2: step ต้องกระโดดตาม ordinal ไม่ใช่ +1
    df = rows_to_df(_with_slots([0, 1, 3, 4, 5], [0, 1, 3, 4, 5]))
    report, ok = integrity_report(df, p0_hint=10.0)
    assert ok, f"step ที่ตรง market slot ต้องผ่าน:\n{report}"
    assert "ข้าม 1 ช่วง" in report.loc[report["ข้อ"] == "E7", "หมายเหตุ"].iloc[0]


def test_e7_rejects_step_out_of_sync_with_market_slot():
    df = rows_to_df(_with_slots([0, 1, 2, 3, 4], [0, 1, 3, 4, 5]))
    report, ok = integrity_report(df, p0_hint=10.0)
    assert not ok, f"step ไม่ตรง market slot ต้องจับได้:\n{report}"


def test_e7_rejects_repeated_market_slot():
    df = rows_to_df(_with_slots([0, 1, 1, 2, 3], [0, 1, 1, 2, 3]))
    _, ok = integrity_report(df, p0_hint=10.0)
    assert not ok


def test_e7_keeps_plus_one_rule_for_rows_without_slot_provenance():
    assert integrity_report(rows_to_df(_with_slots([0, 1, 2, 3, 4], None)), p0_hint=10.0)[1]
    assert not integrity_report(rows_to_df(_with_slots([0, 1, 3, 4, 5], None)), p0_hint=10.0)[1]


def test_order_columns_keeps_slot_provenance_after_the_17():
    df = order_columns(rows_to_df(_with_slots([0, 1, 3, 4, 5], [0, 1, 3, 4, 5])))
    assert list(df.columns)[:17] == COLUMN_ORDER
    assert "market_ordinal" in list(df.columns)[17:]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
