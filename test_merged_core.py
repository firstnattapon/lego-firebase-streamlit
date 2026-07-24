"""lego_dash_core ต้องเป็นโมดูลเดียว และคำนวณรอบเดียวโดยผลลัพธ์ไม่เปลี่ยน

เดิมไฟล์นี้ถูก package ชื่อเดียวกันบังไว้ แล้ว package โหลด module ผ่าน importlib
เพื่อ override 3 ฟังก์ชัน — ledger จึงถูกคำนวณ 2 รอบต่อ 1 render และ deploy ที่หลุด
ไฟล์ใดไฟล์หนึ่งจะ import ไม่ผ่าน เทสต์ชุดนี้ล็อกสัญญาที่ package เคยรับประกันไว้
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import lego_dash_core
from lego_dash_core import (GATED_SEMANTICS, LEGACY_REALIZED_SEMANTICS,
                            count_ledger_corrections, integrity_report,
                            recompute_gated_ledger)

FIX_C = 1500.0
P0 = 100.0


def _row(step, price, status, signal, qty, holdings=15.0,
         semantics=GATED_SEMANTICS, **ledger):
    value = holdings * price
    row = {
        "เวลา (UTC)": f"2026-07-23T18:{step:02d}:00Z", "สินทรัพย์": "AAPL",
        "สถานะ": status, "DNA step": step, "DNA signal": signal,
        "ราคา Pₙ (USD)": price, "จำนวนถือครอง (หุ้น)": holdings,
        "คำสั่ง": "TRIGGER_ACTION" if qty else "PASS",
        "ฝั่ง": "BUY" if status == "READY_BUY" else ("SELL" if status == "READY_SELL" else ""),
        "เหตุผล": status, "จำนวนสั่ง (หุ้น)": qty,
        "มูลค่าพอร์ต (USD)": value, "ส่วนต่างเป้าหมาย (USD)": FIX_C - value,
        "Rₙ อ้างอิง (USD)": 0.0, "ΔAₙ ต่อสเต็ป (USD)": 0.0,
        "Aₙ สะสม (USD)": 0.0, "Eₙ ส่วนเกินสะสม (USD)": 0.0,
        "run_id": f"r{step}", "chain_key": "AAPL_x", "version": step + 1,
        "committed": True, "semantics": semantics,
    }
    row.update(ledger)
    return row


def test_core_is_a_single_module_with_no_shadowing_package():
    path = Path(lego_dash_core.__file__)
    assert path.name == "lego_dash_core.py"
    assert not (path.parent / "lego_dash_core").exists()
    assert not hasattr(lego_dash_core, "_IMPL")


def test_decision_columns_are_never_rewritten():
    """คอลัมน์ตัดสินใจคือหลักฐานที่ commit แล้ว — dashboard แก้ไม่ได้"""
    df = pd.DataFrame([
        _row(0, 100.0, "READY_BUY", 1, 0.0),
        _row(1, 104.0, "READY_BUY", 0, 3.0),      # signal 0 -> ต้องแช่แข็ง
        _row(2, 108.0, "READY_SELL", 1, 0.0),     # qty 0 -> ต้องแช่แข็ง
        _row(3, 96.0, "READY_BUY", 1, 1.5),
    ])
    fixed = recompute_gated_ledger(df, p0=P0)
    for col in ("สถานะ", "DNA signal", "จำนวนสั่ง (หุ้น)"):
        assert list(fixed[col]) == list(df[col])
    # เฉพาะแถว 3 ที่เป็น act จริง
    assert list(fixed["ΔAₙ ต่อสเต็ป (USD)"].astype(float))[1:3] == [0.0, 0.0]
    assert fixed["ΔAₙ ต่อสเต็ป (USD)"].astype(float).iloc[3] != 0.0


def test_recompute_never_mutates_the_callers_frame():
    df = pd.DataFrame([
        _row(0, 100.0, "READY_BUY", 1, 0.0),
        _row(1, 104.0, "PASS_THRESHOLD", 1, 0.0, **{"Rₙ อ้างอิง (USD)": 99.0}),
    ])
    before = df.copy(deep=True)
    fixed = recompute_gated_ledger(df, p0=P0)
    pd.testing.assert_frame_equal(df, before)
    assert count_ledger_corrections(df, fixed) == 1     # Rₙ ที่ถูกแช่แข็งไว้ นับเป็นแถวผิด

    # เส้นทาง fail safe (คอลัมน์ ledger ไม่ครบ) ก็ต้องไม่แตะ frame ผู้เรียกเช่นกัน
    partial = df.drop(columns=["ΔAₙ ต่อสเต็ป (USD)"])
    snapshot = partial.copy(deep=True)
    recompute_gated_ledger(partial, p0=P0)
    pd.testing.assert_frame_equal(partial, snapshot)


def test_reference_is_live_on_every_gated_row():
    df = pd.DataFrame([
        _row(0, 100.0, "READY_BUY", 1, 0.0),
        _row(1, 104.0, "PASS_DNA_ZERO", 0, 0.0),
        _row(2, 92.0, "PASS_THRESHOLD", 1, 0.0),
    ])
    fixed = recompute_gated_ledger(df, p0=P0)
    assert [round(x, 10) for x in fixed["Rₙ อ้างอิง (USD)"].astype(float)] == [
        round(FIX_C * math.log(p / P0), 10) for p in (100.0, 104.0, 92.0)]


def test_legacy_semantics_rows_keep_their_stored_ledger():
    df = pd.DataFrame([
        _row(0, 100.0, "READY_BUY", 1, 0.0, semantics=LEGACY_REALIZED_SEMANTICS,
             **{"Rₙ อ้างอิง (USD)": 7.0, "ΔAₙ ต่อสเต็ป (USD)": 7.0,
                "Aₙ สะสม (USD)": 7.0, "Eₙ ส่วนเกินสะสม (USD)": 7.0}),
        _row(1, 104.0, "READY_SELL", 1, 1.0, semantics=LEGACY_REALIZED_SEMANTICS,
             **{"Rₙ อ้างอิง (USD)": 8.0, "ΔAₙ ต่อสเต็ป (USD)": 8.0,
                "Aₙ สะสม (USD)": 15.0, "Eₙ ส่วนเกินสะสม (USD)": 7.0}),
        _row(2, 108.0, "PASS_THRESHOLD", 1, 0.0),
    ])
    fixed = recompute_gated_ledger(df, p0=P0)
    assert float(fixed["ΔAₙ ต่อสเต็ป (USD)"].iloc[1]) == 8.0
    assert float(fixed["Aₙ สะสม (USD)"].iloc[1]) == 15.0
    assert float(fixed["ΔAₙ ต่อสเต็ป (USD)"].iloc[2]) == 0.0    # gated row freezes


def test_e8_states_the_strict_rule():
    df = pd.DataFrame([
        _row(0, 100.0, "PASS_DNA_ZERO", 0, 0.0),
        _row(1, 104.0, "READY_BUY", 0, 2.0),
    ])
    report, ok = integrity_report(recompute_gated_ledger(df, p0=P0), p0_hint=P0)
    e8 = report.loc[report["ข้อ"] == "E8"].iloc[0]
    assert e8["สมการ/กฎ"] == "signal=1 + READY + qty>0 เท่านั้นที่เทรด; ทุกกรณีอื่นแช่แข็ง"
    assert bool(e8["ผ่าน"]) is False and e8["หมายเหตุ"] == "ผิด 1 แถว"
    assert ok is False
