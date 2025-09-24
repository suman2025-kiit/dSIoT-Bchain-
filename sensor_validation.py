
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class ValidationResult:
    sensor_id: str
    kind: str
    n: int
    ok_rate: float               # fraction of valid samples after cleaning
    gap_ratio: float             # fraction missing/NaN before cleaning
    outlier_ratio: float         # fraction flagged as outliers
    drift_flag: bool             # significant drift detected (e.g., MQ-135)
    validation_score: float      # 0..1, higher is better (feeds Reliability/ISR)
    notes: Dict[str, float]      # extra metrics per sensor type

# ---------- common helpers ----------

def _nan_gap_ratio(x: np.ndarray) -> float:
    return float(np.isnan(x).mean()) if x.size else 1.0

def _median_abs_dev(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def _clip_roc(x: np.ndarray, max_roc: float) -> np.ndarray:
    """Rate-of-change limiter (symmetric), preserves NaNs."""
    y = x.copy()
    for i in range(1, len(y)):
        if np.isnan(y[i-1]) or np.isnan(y[i]):
            continue
        delta = y[i] - y[i-1]
        if delta > max_roc:
            y[i] = y[i-1] + max_roc
        elif delta < -max_roc:
            y[i] = y[i-1] - max_roc
    return y

def _iqr_mask(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    """True = keep, False = outlier (IQR rule), robust to NaNs."""
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x >= lo) & (x <= hi)

def _scale01(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

# ---------- DHT22: temperature (°C) & humidity (%) ----------

def validate_dht22(
    sensor_id: str,
    temp_c: List[Optional[float]],
    rh_pct: List[Optional[float]],
    max_temp_roc: float = 1.0,   # °C per sample
    max_rh_roc: float = 3.0,     # % per sample
    temp_bounds: Tuple[float, float] = (-20.0, 80.0),
    rh_bounds: Tuple[float, float] = (0.0, 100.0)
) -> ValidationResult:
    t = np.array([np.nan if v is None else float(v) for v in temp_c], dtype=float)
    h = np.array([np.nan if v is None else float(v) for v in rh_pct], dtype=float)
    n = len(t)

    gap = (_nan_gap_ratio(t) + _nan_gap_ratio(h)) / 2.0

    # plausibility clip
    t_pl = np.where(np.isnan(t), np.nan, np.clip(t, *temp_bounds))
    h_pl = np.where(np.isnan(h), np.nan, np.clip(h, *rh_bounds))

    # rate-of-change guard
    t_rc = _clip_roc(t_pl, max_temp_roc)
    h_rc = _clip_roc(h_pl, max_rh_roc)

    # robust outlier rejection (IQR)
    keep_t = _iqr_mask(t_rc)
    keep_h = _iqr_mask(h_rc)
    keep = keep_t & keep_h
    outlier_ratio = 1.0 - float(np.nanmean(keep.astype(float)))

    ok = np.sum(keep & ~np.isnan(t_rc) & ~np.isnan(h_rc))
    valid_total = np.sum(~np.isnan(t_rc) & ~np.isnan(h_rc))
    ok_rate = 0.0 if valid_total == 0 else ok / valid_total

    # stability score via MAD (lower MAD → higher stability)
    mad_t = _median_abs_dev(t_rc)
    mad_h = _median_abs_dev(h_rc)
    stab = 0.5 * (_scale01(max(0.0, 2.5 - mad_t), 0.0, 2.5) +
                  _scale01(max(0.0, 5.0 - mad_h), 0.0, 5.0))

    # aggregate validation score (emphasize ok_rate, penalize gaps/outliers)
    validation_score = float(
        0.65 * ok_rate +
        0.20 * stab +
        0.10 * (1.0 - gap) +
        0.05 * (1.0 - outlier_ratio)
    )

    return ValidationResult(
        sensor_id=sensor_id, kind="DHT22", n=n,
        ok_rate=ok_rate, gap_ratio=gap, outlier_ratio=outlier_ratio,
        drift_flag=False,  # DHT22 typically stable; drift handled via stab
        validation_score=validation_score,
        notes={"MAD_temp": mad_t, "MAD_rh": mad_h}
    )

# ---------- MQ-135: air-quality (arbitrary units or ppm proxy) ----------

def validate_mq135(
    sensor_id: str,
    aq: List[Optional[float]],
    max_roc: float = 50.0,                 # units/sample, tune to your sampling rate
    plausible_bounds: Tuple[float, float] = (0.0, 1024.0),
    drift_win: int = 50,                   # samples; use > 2× daily cycles if available
    drift_thresh: float = 0.15             # 15% relative mean shift triggers drift
) -> ValidationResult:
    x = np.array([np.nan if v is None else float(v) for v in aq], dtype=float)
    n = len(x)
    gap = _nan_gap_ratio(x)

    x_pl = np.where(np.isnan(x), np.nan, np.clip(x, *plausible_bounds))
    x_rc = _clip_roc(x_pl, max_roc)
    keep = _iqr_mask(x_rc, k=2.0)
    outlier_ratio = 1.0 - float(np.nanmean(keep.astype(float)))

    ok = np.sum(keep & ~np.isnan(x_rc))
    valid_total = np.sum(~np.isnan(x_rc))
    ok_rate = 0.0 if valid_total == 0 else ok / valid_total

    # drift detection: compare early vs late windows (robust means)
    drift_flag = False
    rel_shift = 0.0
    if n >= 2 * drift_win:
        a = x_rc[:drift_win]
        b = x_rc[-drift_win:]
        ma = np.nanmedian(a)
        mb = np.nanmedian(b)
        denom = max(1e-6, np.nanmedian(np.abs([ma, mb])))
        rel_shift = abs(mb - ma) / denom
        drift_flag = rel_shift >= drift_thresh

    # smoothness/stability via MAD
    mad = _median_abs_dev(x_rc)
    stab = _scale01(max(0.0, 50.0 - mad), 0.0, 50.0)

    # validation score: penalize drift & gaps
    validation_score = float(
        0.55 * ok_rate +
        0.20 * stab +
        0.15 * (1.0 - gap) +
        0.10 * (1.0 - outlier_ratio)
    )
    if drift_flag:
        validation_score *= 0.75  # soft penalty; downstream can revoke via VC

    return ValidationResult(
        sensor_id=sensor_id, kind="MQ-135", n=n,
        ok_rate=ok_rate, gap_ratio=gap, outlier_ratio=outlier_ratio,
        drift_flag=drift_flag,
        validation_score=validation_score,
        notes={"rel_shift": rel_shift, "MAD": mad}
    )

# ---------- PIR (HC-SR501): motion events (binary) ----------

def validate_pir(
    sensor_id: str,
    events: List[Optional[int]],
    debounce_min_samples: int = 2,     # merge bursts within this window
    max_event_rate_hz: Optional[float] = None,  # optional sanity cap
) -> ValidationResult:
    # normalize to {0,1} with NaNs preserved
    raw = np.array([
        np.nan if v is None else (1 if int(v) != 0 else 0)
        for v in events
    ], dtype=float)
    n = len(raw)
    gap = _nan_gap_ratio(raw)

    # simple debouncing (merge micro-bursts)
    sig = raw.copy()
    if debounce_min_samples > 1:
        for i in range(1, n):
            if np.isnan(sig[i]) or np.isnan(sig[i-1]):
                continue
            if sig[i] == 1 and sig[i-1] == 1:
                continue
            # extend single 1s for stability
            if sig[i] == 1 and i + 1 < n and not np.isnan(sig[i+1]) and sig[i+1] == 0:
                sig[i+1] = 1

    # outlier logic for PIR is rate-centric; compute event count & rate
    valid = ~np.isnan(sig)
    ev_count = int(np.nansum(sig))
    duration = float(np.sum(valid))  # samples (unitless here)
    rate = 0.0 if duration == 0 else ev_count / duration

    outlier_ratio = 0.0
    ok_rate = 1.0 - gap

    # optional sanity: improbable high rate → treat as noisy placement/power issue
    rate_ok = 1.0
    if max_event_rate_hz is not None:
        # here, ‘hz’ means per-sample if sampling is 1 Hz; adapt externally if needed
        rate_ok = 1.0 if rate <= max_event_rate_hz else _scale01(max_event_rate_hz / rate, 0.0, 1.0)

    validation_score = float(0.70 * ok_rate + 0.30 * rate_ok)

    return ValidationResult(
        sensor_id=sensor_id, kind="PIR", n=n,
        ok_rate=ok_rate, gap_ratio=gap, outlier_ratio=outlier_ratio,
        drift_flag=False,
        validation_score=validation_score,
        notes={"events": ev_count, "rate": rate}
    )

# ---------- Aggregation: produce inputs for T_intrinsic / ISR ----------

def make_reliability_from_validation(v: ValidationResult) -> float:
    """
    Convert ValidationResult → Reliability in [0,1]
    Heavier penalty for gaps/outliers and (for MQ-135) drift.
    """
    r = 0.7 * v.validation_score + 0.2 * (1.0 - v.gap_ratio) + 0.1 * (1.0 - v.outlier_ratio)
    if v.kind == "MQ-135" and v.drift_flag:
        r *= 0.85
    return float(np.clip(r, 0.0, 1.0))

def make_isr_from_validation(v: ValidationResult) -> float:
    """
    Interaction Success Rate proxy per sensor stream.
    For continuous sensors, treat ‘ok’ sample as a successful interaction.
    For PIR, use debounced rate_ok embedded in validation_score weight.
    """
    return float(np.clip(v.ok_rate, 0.0, 1.0))


if __name__ == "__main__":
    # Quick self-test / demo
    vr_dht = validate_dht22(
        "dht_12",
        temp_c=[25.1, 25.4, None, 26.0, 80.5],
        rh_pct=[56.0, 55.4, 55.2, None, 54.9]
    )
    vr_mq = validate_mq135(
        "mq_07",
        aq=[420, 415, 600, 610, 620, 800, 820, 830, 835, 840]
    )
    vr_pir = validate_pir(
        "pir_03",
        events=[0, 1, 0, 0, 1, 1, 0, 0, 0],
        max_event_rate_hz=0.4
    )

    print("DHT22:", vr_dht)
    print("  Reliability =", make_reliability_from_validation(vr_dht))
    print("  ISR         =", make_isr_from_validation(vr_dht))

    print("MQ-135:", vr_mq)
    print("  Reliability =", make_reliability_from_validation(vr_mq))
    print("  ISR         =", make_isr_from_validation(vr_mq))

    print("PIR:", vr_pir)
    print("  Reliability =", make_reliability_from_validation(vr_pir))
    print("  ISR         =", make_isr_from_validation(vr_pir))
