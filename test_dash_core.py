"""test_dash_core.py — ตรวจ pure logic ของ dashboard (ไม่ต้องมี streamlit/firebase)

ครอบคลุม:
  - rows_to_df   : กรอง committed (fail closed), เรียง version, ทน payload แปลก
  - order_columns: บังคับลำดับสัญญา 17 คอลัมน์แม้ key มาสลับ
  - integrity    : chain ที่ถูกสูตร -> ผ่านทุกข้อ; ปนค่าผิด -> ต้องจับได้ (E3–E8)
"""
from __future__ import annotations

import math
import sys

from lego_dash_core import (COLUMN_ORDER, default_chain_index,
                            filter_audit_rows, integrity_report,
                            order_columns, rows_to_df)

FIX_C = 1500.0
DIFF = 60.0


def _mk_chain(prices: list[float], holdings: list[float], signals: list[int]) -> dict:
    """สร้างแถว committed ตามสูตร LEGO เป๊ะ (spec 17 คอลัมน์) คืน dict แบบ RTDB"""
    rows = {}
    p0 = prices[0]
    A_prev = 0.0
    for n, (p, h, sig) in enumerate(zip(prices, holdings, signals)):
        v = h * p
        gap = FIX_C - v
        if n == 0:
            R = dA = A = E = 0.0
        else:
            R = FIX_C * math.log(p / p0)
            dA = FIX_C * (p / prices[n - 1] - 1.0)
            A = A_prev + dA
            E = A - R
        A_prev = A

        if sig == 0:
            status, action, side, qty = "PASS_DNA_ZERO", "PASS", "", 0.0
        elif abs(gap) <= DIFF:
            status, action, side, qty = "PASS_THRESHOLD", "PASS", "", 0.0
        elif gap > DIFF:
            status, action, side, qty = "READY_BUY", "TRIGGER_ACTION", "BUY", round(gap / p, 5)
        else:
            status, action, side, qty = "READY_SELL", "TRIGGER_ACTION", "SELL", round(-gap / p, 5)

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


def test_integrity_realized_rows_skip_price_formula_for_dA():
    # ราคาขยับแรงแต่ ΔA = 0 (ไม่มีรอบปิด) แล้วค่อย +13.64 เมื่อรอบปิด — ต้องผ่าน
    # ถ้ายังใช้สูตรราคา (E4 เดิม) จะ fail ทันที
    data = _mk_realized_chain([10.0, 11.0, 9.0, 10.0], [0.0, 0.0, 0.0, 13.64])
    report, ok = integrity_report(rows_to_df(data), p0_hint=10.0)
    assert ok, f"แถว realized ต้องไม่ถูกตัดสินด้วยสูตรราคา:\n{report}"
    e4 = report[report["ข้อ"] == "E4"].iloc[0]
    assert "realized" in str(e4["หมายเหตุ"])


def test_integrity_mixed_legacy_then_realized_boundary_ok():
    legacy = _mk_chain(PRICES, HOLD, SIGNALS)          # A ทฤษฎีสะสมถึง version 5
    realized = _mk_realized_chain([10.5, 10.8, 10.2], [0.0, 2.5, 0.0],
                                  start_version=6, start_step=5,
                                  A_start=0.0, p0=10.0)   # baseline รีเซ็ต 0
    report, ok = integrity_report(rows_to_df({**legacy, **realized}), p0_hint=10.0)
    assert ok, f"รอยต่อ legacy→realized ต้องไม่ false-alarm:\n{report}"
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
