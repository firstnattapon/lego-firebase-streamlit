"""Canonical LEGO dashboard core facade.

This package intentionally shadows the legacy ``lego_dash_core.py`` module and
re-exports its public API.  The two ledger functions are wrapped with the
gated_theoretical_v2 execution policy:

- act only when signal == 1, status is READY_BUY/READY_SELL, and qty > 0
- freeze every other combination, regardless of PASS label
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


_IMPL_NAME = "_lego_dash_core_impl"
_IMPL_PATH = Path(__file__).resolve().parent.parent / "lego_dash_core.py"
_SPEC = importlib.util.spec_from_file_location(_IMPL_NAME, _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise ImportError(f"cannot load legacy core from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_IMPL_NAME] = _IMPL
_SPEC.loader.exec_module(_IMPL)

# Re-export the existing API first; policy wrappers below replace two functions.
for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)


def _has_policy_columns(df: pd.DataFrame) -> bool:
    return all(
        col in df.columns
        for col in ("สถานะ", "DNA signal", "จำนวนสั่ง (หุ้น)")
    )


def _executed_trade_mask(df: pd.DataFrame) -> pd.Series:
    """True only for a real, DNA-enabled READY order."""
    status = df["สถานะ"].astype(str)
    signal = df["DNA signal"].astype(int)
    qty = df["จำนวนสั่ง (หุ้น)"].astype(float)
    return (
        status.isin([READY_BUY, READY_SELL])
        & signal.eq(1)
        & qty.gt(0)
    )


def _normalized_policy_view(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all non-executed rows to a frozen PASS view for legacy math."""
    if df.empty or not _has_policy_columns(df):
        return df

    out = df.reset_index(drop=True).copy()
    executed = _executed_trade_mask(out)
    signal = out["DNA signal"].astype(int)

    # The legacy recurrence keys only on status. Normalize every non-executed
    # combination so it cannot advance P_acted/A_n, while retaining the original
    # columns in the public output.
    frozen = ~executed
    out.loc[frozen & signal.eq(0), "สถานะ"] = PASS_DNA_ZERO
    out.loc[frozen & signal.ne(0), "สถานะ"] = PASS_THRESHOLD
    out.loc[frozen, "จำนวนสั่ง (หุ้น)"] = 0.0
    return out


def recompute_gated_ledger(df: pd.DataFrame,
                           p0: float | None = None) -> pd.DataFrame:
    """Recompute with the strict frozen-ledger execution policy.

    Only ``signal=1 + READY_BUY/READY_SELL + qty>0`` advances the ledger.
    Signal 0, PASS, PASS_DNA_ZERO, PASS_THRESHOLD, unknown statuses, and
    malformed READY rows are frozen: delta A is zero, A and P_acted stay put.
    """
    if df.empty or not _has_policy_columns(df):
        return _IMPL.recompute_gated_ledger(df, p0=p0)

    original = df.reset_index(drop=True).copy()
    policy_view = _normalized_policy_view(original)
    fixed = _IMPL.recompute_gated_ledger(policy_view, p0=p0)

    # Recurrence values come from the policy view; decision columns remain the
    # exact committed facts so the dashboard never rewrites audit evidence.
    for col in ("สถานะ", "DNA signal", "จำนวนสั่ง (หุ้น)"):
        fixed[col] = original[col]
    return fixed


def integrity_report(df: pd.DataFrame, p0_hint: float | None = None,
                     tol: float = 1e-6) -> tuple[pd.DataFrame, bool]:
    """Validate recurrence against the strict READY-only execution policy."""
    if df.empty or not _has_policy_columns(df):
        return _IMPL.integrity_report(df, p0_hint=p0_hint, tol=tol)

    original = df.reset_index(drop=True).copy()
    policy_view = _normalized_policy_view(original)
    report, _ = _IMPL.integrity_report(
        policy_view, p0_hint=p0_hint, tol=tol
    )

    status = original["สถานะ"].astype(str)
    signal = original["DNA signal"].astype(int)
    qty = original["จำนวนสั่ง (หุ้น)"].astype(float)
    gap = original["ส่วนต่างเป้าหมาย (USD)"].astype(float)

    ready_buy = status.eq(READY_BUY)
    ready_sell = status.eq(READY_SELL)
    ready = ready_buy | ready_sell
    valid_trade = (
        (ready_buy & signal.eq(1) & gap.gt(0) & qty.gt(0))
        | (ready_sell & signal.eq(1) & gap.lt(0) & qty.gt(0))
    )
    valid_frozen = ~ready & qty.eq(0)
    bad = int((~(valid_trade | valid_frozen)).sum())

    e8 = report["ข้อ"].eq("E8")
    report.loc[e8, "สมการ/กฎ"] = (
        "signal=1 + READY + qty>0 เท่านั้นที่เทรด; ทุกกรณีอื่นแช่แข็ง"
    )
    report.loc[e8, "ผ่าน"] = bad == 0
    report.loc[e8, "หมายเหตุ"] = "" if bad == 0 else f"ผิด {bad} แถว"
    return report, bool(report["ผ่าน"].all())
