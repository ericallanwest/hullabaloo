"""Parameterized Tobler hiking function and slope-aware travel-time integration.

Tobler's (1993) empirical walking speed:

    W(S) = 6 * exp(-3.5 * |S + 0.05|)      km/h,   S = dz/dx  (rise over run, a tangent)

The ``+0.05`` offset puts peak speed on a gentle *downhill* (about -2.9 deg). That
asymmetry is the whole reason the routing graph has to be directed: climbing a trail and
descending it are genuinely different costs, and the optimizer can exploit the difference
by choosing which way round to run a loop.

Everything here is vectorized over numpy arrays so a whole elevation profile — or every
cell of a cost raster — costs one call.
"""

from __future__ import annotations

import numpy as np

from .config import ToblerParams


def tobler_speed_kmh(slope: np.ndarray | float, params: ToblerParams) -> np.ndarray:
    """Walking speed in km/h for a signed slope (rise/run, positive = uphill)."""
    slope = np.asarray(slope, dtype=float)
    speed = params.base_kmh * np.exp(-params.k * np.abs(slope + params.s0))
    return np.maximum(speed * params.pace_factor, params.min_speed_kmh)


def tobler_speed_ms(
    slope: np.ndarray | float, params: ToblerParams, *, off_trail: bool = False
) -> np.ndarray:
    """Walking speed in m/s, optionally with the off-trail penalty applied."""
    speed = tobler_speed_kmh(slope, params) * (1000.0 / 3600.0)
    if off_trail:
        speed = speed * params.off_trail_factor
    return speed


def profile_travel_time(
    distances: np.ndarray,
    elevations: np.ndarray,
    params: ToblerParams,
    *,
    off_trail: bool = False,
    reverse: bool = False,
) -> float:
    """Seconds to walk an elevation profile in one direction.

    ``distances`` is cumulative planimetric distance along the line (metres) and
    ``elevations`` the co-located heights (metres). Time is integrated segment by
    segment, because averaging slope over a whole trail and applying Tobler once badly
    underestimates rolling terrain — the function is convex in |slope|.

    Note the run used is the *planimetric* distance, not the 3D slope distance. Tobler's
    speed is calibrated against map distance, so this is the correct convention; the
    slope-distance correction is already baked into the empirical constants.
    """
    distances = np.asarray(distances, dtype=float)
    elevations = np.asarray(elevations, dtype=float)
    if distances.size < 2:
        return 0.0

    run = np.diff(distances)
    rise = np.diff(elevations)
    if reverse:
        run = run[::-1]
        rise = -rise[::-1]

    valid = run > 1e-9
    if not valid.any():
        return 0.0
    run, rise = run[valid], rise[valid]

    slope = rise / run
    speed = tobler_speed_ms(slope, params, off_trail=off_trail)
    return float(np.sum(run / speed))


def profile_gain_m(elevations: np.ndarray, *, reverse: bool = False) -> float:
    """Cumulative positive elevation gain along a profile, metres."""
    diffs = np.diff(np.asarray(elevations, dtype=float))
    if reverse:
        diffs = -diffs
    return float(diffs[diffs > 0].sum())


def flat_pace_summary(params: ToblerParams) -> dict[str, float]:
    """Human-readable sanity check: what does this parameter set actually imply?"""
    kmh_flat = float(tobler_speed_kmh(0.0, params))
    return {
        "flat_kmh": round(kmh_flat, 2),
        "flat_mph": round(kmh_flat * 0.621371, 2),
        "flat_min_per_mile": round(60.0 / (kmh_flat * 0.621371), 1),
        "up_10pct_mph": round(float(tobler_speed_kmh(0.10, params)) * 0.621371, 2),
        "down_10pct_mph": round(float(tobler_speed_kmh(-0.10, params)) * 0.621371, 2),
        "up_20pct_mph": round(float(tobler_speed_kmh(0.20, params)) * 0.621371, 2),
        "peak_slope": -params.s0,
    }
