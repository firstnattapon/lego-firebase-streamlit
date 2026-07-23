"""Regression tests for gated_theoretical_v2 frozen-ledger policy."""
from __future__ import annotations

import math

import pandas as pd

from lego_dash_core import integrity_report, recompute_gated_ledger


FIX_C = 3000.0
P0 = 320.0


def _rows(cases: list[tuple[int, str, float]]) -> pd.DataFrame:
    rows = []
    acted = P0
    accumulated = 0.0
    prices = [P0 + i * 2.0 for i in range(len(cases))]

    for i, ((signal, status, qty), price) in enumerate(zip(cases, prices)):
        gap = FIX_C - 9.3 * price
        reference = 0.0 if i == 0 else FIX_C * math.log(price / P0)
        executed = signal == 1 and status in {"READY_BUY", "READY_SELL"} and qty > 0
        if i == 0:
            delta = accumulated = excess = 0.0
            acted = price
        elif executed:
            delta = FIX_C * (price / acted - 1.0)
            accumulated += delta
            acted = price
            excess = accumulated - reference
        else:
            # Deliberately write an incorrect act-like value. recompute must freeze it.
            delta = FIX_C * (price / acted - 1.0)
            excess = accumulated - FIX_C * math.log(acted / P0)

        rows.append({
            "เวลา (UTC)": f"2026-07-23T14:{i:02d}:00Z",
            "สินทรัพย์": "AAPL",
            "สถานะ": status,
            "DNA step": i,
            "DNA signal": signal,
            "ราคา Pₙ (USD)": price,
            "จำนวนถือครอง (หุ้น)": 9.3,
            "คำสั่ง": "TRIGGER_ACTION" if executed else "PASS",
            "ฝั่ง": "BUY" if status == "READY_BUY" else ("SELL" if status == "READY_SELL" else ""),
            "เหตุผล": status,
            "จำนวนสั่ง (หุ้น)": qty,
            "มูลค่าพอร์ต (USD)": 9.3 * price,
            "ส่วนต่างเป้าหมาย (USD)": gap,
            "Rₙ อ้างอิง (USD)": reference,
            "ΔAₙ ต่อสเต็ป (USD)": delta,
            "Aₙ สะสม (USD)": accumulated + (delta if not executed and i > 0 else 0.0),
            "Eₙ ส่วนเกินสะสม (USD)": excess,
            "run_id": f"r{i}",
            "chain_key": "AAPL_test",
            "version": i + 1,
            "committed": True,
            "semantics": "gated_theoretical_v2",
        })
    return pd.DataFrame(rows)


def test_all_non_ready_statuses_freeze_for_both_signals():
    df = _rows([
        (1, "READY_BUY", 1.0),
        (0, "PASS", 0.0),
        (0, "PASS_DNA_ZERO", 0.0),
        (1, "PASS", 0.0),
        (1, "PASS_DNA_ZERO", 0.0),
        (1, "PASS_THRESHOLD", 0.0),
        (0, "UNKNOWN_STATUS", 0.0),
    ])
    fixed = recompute_gated_ledger(df, p0=P0)

    assert list(fixed["ΔAₙ ต่อสเต็ป (USD)"].astype(float))[1:] == [0.0] * 6
    assert len(set(fixed["Aₙ สะสม (USD)"].astype(float).tolist())) == 1


def test_signal_zero_ready_status_is_still_frozen():
    df = _rows([
        (1, "READY_BUY", 1.0),
        (0, "READY_BUY", 1.0),
        (0, "READY_SELL", 1.0),
    ])
    fixed = recompute_gated_ledger(df, p0=P0)

    assert fixed.iloc[1]["ΔAₙ ต่อสเต็ป (USD)"] == 0.0
    assert fixed.iloc[2]["ΔAₙ ต่อสเต็ป (USD)"] == 0.0

    report, ok = integrity_report(fixed, p0_hint=P0)
    assert not ok
    assert not bool(report.loc[report["ข้อ"] == "E8", "ผ่าน"].iloc[0])


def test_ready_without_positive_quantity_is_frozen_and_invalid():
    df = _rows([
        (1, "READY_BUY", 1.0),
        (1, "READY_BUY", 0.0),
        (1, "READY_SELL", 0.0),
    ])
    fixed = recompute_gated_ledger(df, p0=P0)

    assert fixed.iloc[1]["ΔAₙ ต่อสเต็ป (USD)"] == 0.0
    assert fixed.iloc[2]["ΔAₙ ต่อสเต็ป (USD)"] == 0.0

    report, ok = integrity_report(fixed, p0_hint=P0)
    assert not ok
    assert not bool(report.loc[report["ข้อ"] == "E8", "ผ่าน"].iloc[0])
