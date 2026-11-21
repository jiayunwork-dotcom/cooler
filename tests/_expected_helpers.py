"""
Helpers for the distance-background test suite.

All test inputs are HDF5 cooler archives generated locally from in-memory
bin/pixel tables; no external or downloaded data is used.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd

import cooler


def make_cooler(
    path: str,
    bins: pd.DataFrame,
    pixels: pd.DataFrame,
    *,
    symmetric_upper: bool = True,
    chromsizes: dict | None = None,
    sparse: bool = False,
    extra_pixel_cols: dict | None = None,
) -> cooler.Cooler:
    """Create a cooler archive and return an open :class:`Cooler` handle."""
    if extra_pixel_cols:
        pixels = pixels.copy()
        for name, values in extra_pixel_cols.items():
            pixels[name] = values
    cooler.create_cooler(
        path,
        bins,
        pixels,
        symmetric_upper=symmetric_upper,
        chromsizes=chromsizes,
        sparse=sparse,
    )
    return cooler.Cooler(path)


def add_weight_column(path: str, name: str, values: np.ndarray) -> None:
    """Attach (or replace) a bin-table weight column in a written cooler."""
    with h5py.File(path, "r+") as f:
        root = f["/"]
        if name in root["bins"]:
            del root["bins"][name]
        root["bins"].create_dataset(name, data=np.asarray(values, dtype=np.float64))


def midpoint(bins: pd.DataFrame) -> np.ndarray:
    return (bins["start"].to_numpy() + bins["end"].to_numpy()) / 2.0


def reference_stats(
    bins: pd.DataFrame,
    pixels: pd.DataFrame,
    edges,
    *,
    weights: np.ndarray | None = None,
    divisive: bool = False,
    chrom_offset: int = 0,
    global_chrom: np.ndarray | None = None,
):
    """Exhaustive O(n^2) reference for a single-chromosome bin table.

    Returns dict with observed_sum, raw_sum, n_pairs, raw_mean and the
    per-bucket expected mean (raw mode), computed by enumerating every pair
    i <= j exactly once, including pairs with no stored pixel.
    """
    edges = np.asarray(edges, dtype=np.float64)
    n_buckets = len(edges) - 1
    mids = midpoint(bins)
    n = len(bins)
    if weights is None:
        valid = np.ones(n, dtype=bool)
    else:
        w = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(w) & (w > 0)

    stored = {}
    for _, row in pixels.iterrows():
        i = int(row["bin1_id"]) - chrom_offset
        j = int(row["bin2_id"]) - chrom_offset
        if (
            global_chrom is not None
            and global_chrom[int(row["bin1_id"])] != global_chrom[int(row["bin2_id"])]
        ):
            continue
        stored[(i, j)] = float(row["count"])

    observed_sum = np.zeros(n_buckets)
    raw_sum = np.zeros(n_buckets)
    n_pairs = np.zeros(n_buckets, dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            if not (valid[i] and valid[j]):
                continue
            d = abs(mids[j] - mids[i])
            b = np.searchsorted(edges, d, side="right") - 1
            if not (0 <= b < n_buckets):
                continue
            n_pairs[b] += 1
            c = stored.get((i, j), 0.0)
            raw_sum[b] += c
            if weights is None:
                observed_sum[b] += c
            elif divisive:
                observed_sum[b] += c / (weights[i] * weights[j])
            else:
                observed_sum[b] += c * weights[i] * weights[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_mean = np.where(n_pairs > 0, raw_sum / np.maximum(n_pairs, 1), np.nan)
    raw_mean[(n_pairs > 0) & (raw_sum == 0)] = 0.0
    return {
        "observed_sum": observed_sum,
        "raw_sum": raw_sum,
        "n_pairs": n_pairs,
        "raw_mean": raw_mean,
        "valid": valid,
        "stored": stored,
    }


def reference_oe_matrix(
    bins: pd.DataFrame,
    pixels: pd.DataFrame,
    edges,
    expected_per_bucket: np.ndarray,
    usable: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    divisive: bool = False,
):
    """Dense OE reference on one chromosome, including implicit zeros/NaN."""
    edges = np.asarray(edges, dtype=np.float64)
    n = len(bins)
    mids = midpoint(bins)
    valid = (
        np.ones(n, bool)
        if weights is None
        else (
            np.isfinite(np.asarray(weights, float)) & (np.asarray(weights, float) > 0)
        )
    )
    stored = {}
    for _, row in pixels.iterrows():
        stored[(int(row["bin1_id"]), int(row["bin2_id"]))] = float(row["count"])
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if not (valid[i] and valid[j]):
                continue
            d = abs(mids[j] - mids[i])
            b = np.searchsorted(edges, d, side="right") - 1
            if not (0 <= b < len(edges) - 1) or not usable[b]:
                continue
            c = stored.get((min(i, j), max(i, j)), stored.get((i, j), 0.0))
            if weights is None:
                obs = c
            elif divisive:
                obs = c / (weights[i] * weights[j])
            else:
                obs = c * weights[i] * weights[j]
            out[i, j] = obs / expected_per_bucket[b]
    return out


def write_h5py_cooler_with_explicit_zeros(
    path: str,
    chroms: list[tuple[str, int]],
    bins_df: pd.DataFrame,
    pixel_df: pd.DataFrame,
) -> None:
    """A cooler where zero-count pixel rows are guaranteed to stay stored."""
    # create_cooler already preserves explicit zero rows; this helper exists
    # so tests state that assumption explicitly at the call site.
    cooler.create_cooler(path, bins_df, pixel_df)
    assert os.path.exists(path)
