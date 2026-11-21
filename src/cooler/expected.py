"""
Intra-chromosomal distance-background estimation and observed/expected queries.

The distance background of a contact map is the expected signal level as a
function of the genomic distance between two bins.  This module estimates it
from a fixed, read-only cooler archive without reordering or re-aggregating
bins or pixels.  Bin layouts may be non-uniform and contain gaps; distance is
always the absolute difference of the *midpoints* of two bins, which can be a
half-integer number of base pairs.

Estimates are per chromosome and per distance bucket (left-closed,
right-open).  Every pair of valid bins with ``i <= j`` is counted once,
including pairs that have no stored pixel (implicit zeros).  The geometry is
therefore computed from the bin table directly and never from the sparse
pixel coordinates.

An estimate can live in memory (:class:`ExpectedResult`) or be stored as a
named analysis layer inside the cooler group (:func:`save_expected`,
:func:`load_expected`).  Layers bind the exact bin layout, the weight vector
and its multiplicative/divisive interpretation, the distance edges and the
fitting policy, so they are rejected if the underlying archive changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .core import CSRReader
from .util import open_hdf5

__all__ = [
    "BackgroundModel",
    "ExpectedResult",
    "estimate_expected",
    "list_expected",
    "load_expected",
    "remove_expected",
    "save_expected",
]

# Default stabilization policy.  See ``_stabilize_chromosome`` for the
# meaning of each parameter.
DEFAULT_POLICY = {
    "stabilize": False,
    "min_support": 30,
    "window": 7,
    "huber_delta": 1.345,
    "spike_z": 3.0,
    "tail_monotone": True,
}

_EXPECTED_GRP = "expected"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_edges(distance_edges: Any) -> np.ndarray:
    """Validate the distance bucket edges and return them as int64 bp."""
    if isinstance(distance_edges, np.ndarray):
        edges = distance_edges
        if edges.dtype == bool:
            raise TypeError("distance_edges must be non-boolean integers")
    else:
        edges = np.asarray(distance_edges)
        if any(isinstance(x, bool) for x in np.atleast_1d(distance_edges)):
            raise TypeError("distance_edges must be non-boolean integers")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("distance_edges must contain at least two edges")
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("distance_edges must contain non-negative integers (bp)")
    if np.any(edges < 0):
        raise ValueError("distance_edges must be non-negative")
    if edges[0] != 0:
        raise ValueError("distance_edges must start at 0")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("distance_edges must be strictly increasing")
    return edges.astype(np.int64, copy=True)


def _validate_chunksize(chunksize: Any) -> int:
    if isinstance(chunksize, bool) or not isinstance(chunksize, (int, np.integer)):
        raise TypeError("chunksize must be a non-boolean positive integer")
    chunksize = int(chunksize)
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    return chunksize


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _resolve_weight_spec(
    balance: Any, divisive_weights: Any
) -> tuple[bool, str | None, bool]:
    """Normalize the ``balance`` / ``divisive_weights`` arguments.

    Returns (use_weights, weight_column, divisive).
    """
    if isinstance(balance, bool):
        use_weights = balance
        name = "weight" if use_weights else None
    elif isinstance(balance, str):
        use_weights = True
        name = balance
    else:
        raise TypeError("balance must be a boolean or a weight column name")
    if divisive_weights is None:
        # Alias convention shared with Cooler.matrix: 4DN vectors are divisive.
        divisive = use_weights and name in {"KR", "VC", "VC_SQRT"}
    else:
        if not isinstance(divisive_weights, bool):
            raise TypeError("divisive_weights must be a boolean or None")
        divisive = divisive_weights
    return use_weights, name, divisive


def _validity_from_weights(weights: np.ndarray) -> np.ndarray:
    """Finite positive weights mark valid bins; NaN and 0 exclude a bin.

    Negative or infinite weights are rejected.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weight vector must be 1-dimensional")
    finite = np.isfinite(w)
    if np.any(finite & (w < 0)):
        raise ValueError("balancing weights must be non-negative; got negative values")
    if np.any(np.isinf(w)):
        raise ValueError("balancing weights must be finite; got infinite values")
    return finite & (w > 0)


# ---------------------------------------------------------------------------
# Fingerprints for binding an estimate to its source archive
# ---------------------------------------------------------------------------


def _hash_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _bin_fingerprint_from_h5(root: h5py.Group) -> dict[str, Any]:
    """Content hash of the on-disk bin layout datasets.

    Uses the raw stored dtypes (integer ENUM codes for ``chrom``), so the
    fingerprint is independent of categorical/string conversion performed by
    the table readers.
    """
    return {
        "nbins": int(root["bins/start"].shape[0]),
        "chrom_sha256": _hash_array(root["bins/chrom"][:]),
        "start_sha256": _hash_array(root["bins/start"][:]),
        "end_sha256": _hash_array(root["bins/end"][:]),
    }


def _weight_fingerprint(weights: np.ndarray) -> dict[str, Any]:
    # Normalize to the canonical float64 representation actually stored in a
    # layer, so float32 source columns compare equal after a save/load cycle.
    w = np.asarray(weights, dtype=np.float64)
    return {
        "nbins": len(w),
        "dtype": "float64",
        "sha256": _hash_array(w),
    }


# ---------------------------------------------------------------------------
# Geometric opportunity counts
# ---------------------------------------------------------------------------


def _opportunity_counts(mid2: np.ndarray, edges2: np.ndarray) -> np.ndarray:
    """Number of valid ``i < j`` pairs falling in each distance bucket.

    Parameters
    ----------
    mid2 : int64 array
        Twice the midpoint coordinate of every valid bin of one chromosome,
        sorted (bin order is genomic order).
    edges2 : int64 array
        Twice the distance bucket edges.

    Returns
    -------
    ndarray of shape (len(edges2) - 1,)
        Off-diagonal pair counts per bucket.  Diagonal pairs are added by the
        caller.

    Notes
    -----
    For pair ``(i, j)`` with ``i < j`` the distance is ``mid2[j] - mid2[i]``
    (coordinates are sorted).  Bucket ``d`` holds the pair when
    ``edges2[d] <= mid2[j] - mid2[i] < edges2[d+1]``, i.e. when
    ``mid2[j] - edges2[d+1] < mid2[i] <= mid2[j] - edges2[d]``.  Both bounds
    are vectorized binary searches, so the cost is O(D * n log n) and the
    working memory O(n); no bin pair is ever enumerated.
    """
    n_buckets = len(edges2) - 1
    counts = np.zeros(n_buckets, dtype=np.int64)
    n = mid2.size
    j = np.arange(n)
    for d in range(n_buckets):
        # Number of i < j with mid2[i] <= mid2[j] - edges2[d], capped at j.
        upper = np.minimum(j, np.searchsorted(mid2, mid2 - edges2[d], side="right"))
        # Number of i with mid2[i] <= mid2[j] - edges2[d+1] (strictly below
        # the bucket's lower distance bound).
        lower = np.searchsorted(mid2, mid2 - edges2[d + 1], side="right")
        counts[d] = int(np.sum(np.maximum(upper - lower, 0)))
    return counts


def _bucket_of(dist2: np.ndarray, edges2: np.ndarray) -> np.ndarray:
    """Bucket index of doubled distances, left-closed/right-open.

    A distance equal to an edge belongs to the bucket starting at that edge,
    matching :func:`_opportunity_counts`.  Returns -1 for distances below
    ``edges2[0]`` and ``len(edges2) - 1`` for distances at/above the last
    edge (callers treat both as out of range).
    """
    # For an exact edge value e_k: left = right = k, and the value belongs to
    # bucket k. For an interior value in (e_k, e_{k+1}): left = right = k+1,
    # and it belongs to bucket k. So interior values need a -1 shift.
    left_pos = np.searchsorted(edges2, dist2, side="left")
    right_pos = np.searchsorted(edges2, dist2, side="right")
    on_edge = left_pos < right_pos
    return np.where(on_edge, left_pos, left_pos - 1)


# ---------------------------------------------------------------------------
# Stabilization
# ---------------------------------------------------------------------------


def _stabilize_chromosome(
    raw_mean: np.ndarray,
    n_pairs: np.ndarray,
    *,
    min_support: int,
    window: int,
    huber_delta: float,
    spike_z: float,
    tail_monotone: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stabilize raw bucket means against sparse tails and local spikes.

    The strategy acts on a *single chromosome* (no borrowing across
    chromosomes), in log space since distance decay spans orders of
    magnitude:

    1. A smooth local trend is fit with a centered window.  Each bucket
       contributes with a Huber weight (so isolated deviations do not drag
       the trend) and with its pair-count support (so well-measured buckets
       dominate poorly measured ones).
    2. A single robust residual scale (MAD) is estimated chromosome-wide.
    3. Well-supported buckets whose log-mean deviates from the trend by more
       than ``spike_z`` scales *and whose immediate neighbours do not* are
       isolated spikes: they are replaced by the trend.  Sustained
       enhancements spanning several adjacent buckets move the local trend
       and scale with them, so the signal is retained.
    4. Low-support buckets (``n_pairs < min_support``) are replaced by the
       trend, which reuses neighbouring evidence without borrowing across
       chromosomes or fabricating values for unsupported buckets.
    5. Optionally, the low-support tail is forced monotone non-increasing to
       suppress noise upturns past the informative range.

    Buckets with no valid pairs are never filled (they stay unavailable), and
    a bucket whose observed total is zero keeps an exact zero expected value
    that remains non-divisive.  A constant positive background is preserved
    exactly.  The raw statistics are always kept alongside for auditing.

    Returns (stabilized, trend, status) where ``status`` is an object array
    with values ``"raw"``, ``"despiked"``, ``"smoothed"`` or ``"unavailable"``.
    """
    n = raw_mean.size
    trend = np.full(n, np.nan)
    stabilized = np.full(n, np.nan)
    status = np.full(n, "unavailable", dtype=object)

    supported = n_pairs > 0
    positive = supported & (raw_mean > 0)
    if not np.any(positive):
        zero_ok = supported & (raw_mean == 0)
        stabilized[zero_ok] = 0.0
        status[zero_ok] = "raw"
        return stabilized, trend, status

    with np.errstate(divide="ignore"):
        log_mean = np.where(positive, np.log(raw_mean), np.nan)
    # Down-weight, never up-weight, low-support buckets when fitting.
    support = n_pairs.astype(np.float64)
    avg_support = support[positive].mean()
    support_weight = np.minimum(1.0, support / max(avg_support, 1.0))
    support_weight = np.where(positive, support_weight, 0.0)

    w = int(window)
    if w < 1:
        raise ValueError("stabilization window must be >= 1")
    if huber_delta <= 0 or spike_z <= 0:
        raise ValueError("huber_delta and spike_z must be positive")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")

    for k in range(n):
        if not positive[k]:
            continue
        lo = max(0, k - w)
        hi = min(n, k + w + 1)
        neigh = np.arange(lo, hi)[positive[lo:hi]]
        vals = log_mean[neigh]
        wt = support_weight[neigh].copy()
        if wt.sum() <= 0:
            wt = np.ones_like(wt)
        loc = np.average(vals, weights=wt)
        # Two iterations of Huber-weighted local mean.
        for _ in range(2):
            resid = vals - loc
            scale = np.median(np.abs(vals - np.median(vals)))
            scale = max(scale, 1e-12)
            h = np.minimum(1.0, huber_delta * scale / np.maximum(np.abs(resid), 1e-12))
            loc = np.average(vals, weights=wt * h)
        trend[k] = loc

    with_trend = positive & ~np.isnan(trend)
    residuals = log_mean[with_trend] - trend[with_trend]
    mad = np.median(np.abs(residuals - np.median(residuals))) if residuals.size else 0.0
    sigma = max(1.4826 * mad, 1e-12)
    dev = log_mean - trend

    for k in range(n):
        if not supported[k]:
            continue
        if raw_mean[k] == 0:
            stabilized[k] = 0.0
            status[k] = "raw"
            continue
        if np.isnan(trend[k]):  # pragma: no cover - defensive
            continue
        trend_value = np.exp(trend[k])
        if n_pairs[k] < min_support:
            stabilized[k] = trend_value
            status[k] = "smoothed"
            continue
        # Isolated spike: large positive deviation while both neighbours stay
        # within the robust band. Sustained multi-bucket enhancements do not
        # satisfy the neighbour condition and are kept as measured. Edge
        # buckets cannot be isolated by this definition (they miss a side),
        # and the steepest decay bins sit there, so they stay raw.
        isolated = 0 < k < n - 1
        if isolated:
            for nb in (k - 1, k + 1):
                if (
                    positive[nb]
                    and not np.isnan(trend[nb])
                    and dev[nb] > spike_z * sigma
                ):
                    isolated = False
        is_spike = isolated and dev[k] > spike_z * sigma
        if is_spike:
            stabilized[k] = trend_value
            status[k] = "despiked"
        else:
            stabilized[k] = raw_mean[k]
            status[k] = "raw"

    # Monotone low-support tail: from the first unsupported-by-policy bucket
    # onward, available values may only decrease with distance.
    if tail_monotone:
        low_idx = np.flatnonzero(supported & (n_pairs < min_support))
        if low_idx.size:
            start = int(low_idx[0])
            running = np.inf
            for k in range(start, n):
                if np.isfinite(stabilized[k]):
                    stabilized[k] = min(stabilized[k], running)
                    running = stabilized[k]

    trend_out = np.where(np.isnan(trend), np.nan, np.exp(trend))
    return stabilized, trend_out, status


# ---------------------------------------------------------------------------
# Estimate result container
# ---------------------------------------------------------------------------


@dataclass
class ExpectedResult:
    """A distance-background estimate held in memory.

    Attributes
    ----------
    distance_edges : ndarray (int64 bp)
        Bucket edges; bucket ``d`` covers ``[edges[d], edges[d+1])``.
    chrom_names : list of str
        Chromosomes in archive order, including chromosomes without bins.
    chrom_offsets : ndarray (int64)
        Bin-ID boundaries of each chromosome.
    valid : ndarray (bool)
        Per-bin validity flag (all bins valid without weights).
    mid2 : ndarray (int64)
        Twice the midpoint coordinate of each bin within its chromosome
        (absolute bp coordinate, so gaps are respected).
    observed_sum : ndarray, shape (n_chroms, n_buckets)
        Total weighted observed signal per bucket (raw counts when
        ``balance=False``).
    raw_sum : ndarray, shape (n_chroms, n_buckets)
        Total raw counts per bucket regardless of weighting.
    n_pairs : ndarray (int64), shape (n_chroms, n_buckets)
        Number of valid bin pairs per bucket, including implicit-zero pairs.
    raw_mean : ndarray (float64), shape (n_chroms, n_buckets)
        ``raw_sum / n_pairs``; zero for empty buckets, NaN without support.
    expected : ndarray (float64), shape (n_chroms, n_buckets)
        Final expected values; NaN where unavailable (no support or a zero
        mean that cannot be a divisor).
    usable : ndarray (bool), shape (n_chroms, n_buckets)
        Whether ``expected`` is a finite positive divisor.
    status : ndarray (object), shape (n_chroms, n_buckets)
        Per-bucket provenance: ``"raw"``, ``"despiked"``, ``"smoothed"`` or
        ``"unavailable"``.
    policy : dict
        Fitting policy, including the stabilization parameters.
    balance : bool
        Whether weights were used.
    weight_name : str or None
        Source weight column name.
    divisive_weights : bool
        Multiplicative (False) vs divisive (True) interpretation.
    weight_values : ndarray or None
        Full weight vector bound to the estimate (NaN for excluded bins when
        weighted), or None for the unweighted estimate.
    bin_binding : dict
        Fingerprint of the source bin layout.
    weight_binding : dict or None
        Fingerprint of the source weight vector.
    """

    distance_edges: np.ndarray
    chrom_names: list[str]
    chrom_offsets: np.ndarray
    valid: np.ndarray
    mid2: np.ndarray
    observed_sum: np.ndarray
    raw_sum: np.ndarray
    n_pairs: np.ndarray
    raw_mean: np.ndarray
    expected: np.ndarray
    usable: np.ndarray
    status: np.ndarray
    policy: dict[str, Any]
    balance: bool
    weight_name: str | None
    divisive_weights: bool
    weight_values: np.ndarray | None
    bin_binding: dict[str, Any]
    weight_binding: dict[str, Any] | None
    layer_name: str | None = None

    # -- introspection ------------------------------------------------------

    @property
    def n_buckets(self) -> int:
        return len(self.distance_edges) - 1

    def summary(self, chrom: str | None = None) -> pd.DataFrame:
        """Long-form table of the per-chromosome, per-bucket statistics.

        Columns: ``chrom``, ``distance_lo``, ``distance_hi``,
        ``observed_sum`` (weighted observed total), ``n_pairs`` (valid bin
        pairs), ``raw_sum``, ``raw_mean``, ``expected``, ``usable``,
        ``status``.
        """
        if chrom is not None and chrom not in self.chrom_names:
            raise KeyError(f"unknown chromosome {chrom!r}")
        frames = []
        for ci, name in enumerate(self.chrom_names):
            if chrom is not None and name != chrom:
                continue
            n_buckets = self.n_buckets
            frames.append(
                pd.DataFrame(
                    {
                        "chrom": np.full(n_buckets, name, dtype=object),
                        "distance_lo": self.distance_edges[:-1],
                        "distance_hi": self.distance_edges[1:],
                        "observed_sum": self.observed_sum[ci],
                        "n_pairs": self.n_pairs[ci],
                        "raw_sum": self.raw_sum[ci],
                        "raw_mean": self.raw_mean[ci],
                        "expected": self.expected[ci],
                        "usable": self.usable[ci],
                        "status": self.status[ci],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    # -- binding checks -----------------------------------------------------

    def check_bins(self, root: h5py.Group) -> None:
        """Raise if the cooler's bin table does not match the bound layout.

        ``root`` is the open cooler group; the check hashes the raw bin
        datasets exactly as at estimation time.
        """
        fp = _bin_fingerprint_from_h5(root)
        if fp != self.bin_binding:
            raise ValueError(
                "The bin layout of the cooler has changed since this expected "
                "value layer was estimated; re-estimate the background."
            )

    def check_weights(self, weights: np.ndarray | None) -> None:
        """Raise if the given weight vector does not match the bound one."""
        if not self.balance:
            return
        if weights is None:
            raise ValueError(
                "This expected value layer requires balancing weights, but no "
                "matching weight column is available."
            )
        fp = _weight_fingerprint(weights)
        if fp != self.weight_binding:
            raise ValueError(
                "The balancing weights have changed since this expected value "
                "layer was estimated; re-estimate the background."
            )

    # -- query-time machinery ----------------------------------------------

    def _chrom_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        """bin_id -> chrom id arrays (cached on first use)."""
        cached = getattr(self, "_lookup_cache", None)
        if cached is None:
            n = self.valid.size
            chrom_id = np.empty(n, dtype=np.int64)
            for ci in range(len(self.chrom_names)):
                chrom_id[self.chrom_offsets[ci] : self.chrom_offsets[ci + 1]] = ci
            edges2 = (2 * self.distance_edges).astype(np.int64)
            cached = (chrom_id, edges2)
            object.__setattr__(self, "_lookup_cache", cached)
        return cached

    def bucket_for_pairs(
        self, bin1_id: np.ndarray, bin2_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify stored pixel coordinates.

        Returns ``(chrom_id, bucket, in_range)``: same-chromosome pixels get
        the chromosome id and the bucket of their midpoint distance, and are
        flagged in-range only when the distance falls inside the edges.
        """
        chrom_id, edges2 = self._chrom_lookup()
        c1 = chrom_id[bin1_id]
        c2 = chrom_id[bin2_id]
        same = c1 == c2
        dist2 = np.abs(self.mid2[bin2_id] - self.mid2[bin1_id])
        bucket = _bucket_of(dist2, edges2)
        in_range = same & (bucket >= 0) & (bucket < self.n_buckets)
        return c1, bucket, in_range

    def as_background(self) -> BackgroundModel:
        """Build the query-time :class:`BackgroundModel` for this estimate."""
        return BackgroundModel(self)


# ---------------------------------------------------------------------------
# Query-time background
# ---------------------------------------------------------------------------


class BackgroundModel:
    """Applies an :class:`ExpectedResult` to pixel records and dense boxes.

    This is the object handed to the low-level range query engines.  It knows
    how to annotate stored/reflected pixel dicts with the virtual ``expected``
    and ``oe`` fields and how to build the dense expected/NaN layout of any
    bounding box.
    """

    def __init__(self, result: ExpectedResult):
        self.result = result
        r = result
        self.chrom_id, self.edges2 = r._chrom_lookup()
        self.mid2 = r.mid2
        self.valid = r.valid
        self.chrom_offsets = r.chrom_offsets
        self.expected = r.expected
        self.usable = r.usable
        self.balance = r.balance
        self.divisive_weights = r.divisive_weights
        # Multiplicative bias factors for OE computation.
        if r.balance:
            w = np.asarray(r.weight_values, dtype=np.float64)
            self.bias = (1.0 / w) if r.divisive_weights else w
        else:
            self.bias = None

    @property
    def virtual_fields(self) -> tuple[str, ...]:
        return ("expected", "oe")

    # -- pixel-level annotation --------------------------------------------

    def annotate(
        self, dct: dict[str, np.ndarray], fields: list[str]
    ) -> dict[str, np.ndarray]:
        """Add requested virtual columns to a pixel-record dict."""
        if not fields:
            return dct
        r = self.result
        b1 = dct["bin1_id"].astype(np.int64, copy=False)
        b2 = dct["bin2_id"].astype(np.int64, copy=False)
        c1 = self.chrom_id[b1]
        c2 = self.chrom_id[b2]
        dist2 = np.abs(self.mid2[b2] - self.mid2[b1])
        bucket = _bucket_of(dist2, self.edges2)
        same = c1 == c2
        in_range = same & (bucket >= 0) & (bucket < r.n_buckets)
        n = b1.size

        want_expected = "expected" in fields
        want_oe = "oe" in fields
        exp_col = None
        if want_expected or want_oe:
            exp_col = np.full(n, np.nan, dtype=np.float64)
            if in_range.any():
                sel = np.flatnonzero(in_range)
                vals = r.expected[c1[sel], bucket[sel]]
                ok = r.usable[c1[sel], bucket[sel]] & np.isfinite(vals) & (vals > 0)
                exp_col[sel[ok]] = vals[ok]
            if want_expected:
                dct["expected"] = exp_col

        if want_oe:
            obs = dct["count"].astype(np.float64, copy=False)
            if self.bias is not None:
                obs = self.bias[b1] * self.bias[b2] * obs
            oe = np.full(n, np.nan, dtype=np.float64)
            if in_range.any():
                sel = np.flatnonzero(in_range)
                e = r.expected[c1[sel], bucket[sel]]
                ok = r.usable[c1[sel], bucket[sel]] & np.isfinite(e) & (e > 0)
                oe[sel[ok]] = obs[sel[ok]] / e[ok]
            dct["oe"] = oe
        return dct

    # -- dense box layout ---------------------------------------------------

    def dense_expected(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Full dense expected-value matrix for a bounding box.

        Inter-chromosomal cells, out-of-edge distances, unsupported buckets
        and excluded bins are NaN.
        """
        i0, i1, j0, j1 = bbox
        rows = np.arange(i0, i1, dtype=np.int64)
        cols = np.arange(j0, j1, dtype=np.int64)
        r = self.result
        exp = np.full((i1 - i0, j1 - j0), np.nan, dtype=np.float64)
        if rows.size == 0 or cols.size == 0:
            return exp
        cr = self.chrom_id[rows]
        cc = self.chrom_id[cols]
        valid = self.valid
        # Process each chromosome block appearing on both axes, without ever
        # materializing more than the requested box.
        for ci in np.unique(cr):
            rm = cr == ci
            cm = cc == ci
            if not rm.any() or not cm.any():
                continue
            ridx = rows[rm]
            cidx = cols[cm]
            if not (valid[ridx].all() and valid[cidx].all()):
                # Mixed valid/invalid: mark valid-valid cells, leave rest NaN.
                rv = valid[ridx]
                cv = valid[cidx]
            else:
                rv = np.ones(ridx.size, dtype=bool)
                cv = np.ones(cidx.size, dtype=bool)
            d2 = np.abs(self.mid2[cidx][None, :] - self.mid2[ridx][:, None])
            bucket = _bucket_of(d2, self.edges2)
            in_range = (bucket >= 0) & (bucket < r.n_buckets)
            block = np.full((ridx.size, cidx.size), np.nan, dtype=np.float64)
            if in_range.any():
                b = np.where(in_range, bucket, 0)
                vals = r.expected[ci, b]
                ok = in_range & r.usable[ci, b] & (vals > 0)
                block[ok] = vals[ok]
            block[~rv, :] = np.nan
            block[:, ~cv] = np.nan
            # Scatter into the output box.
            rr = np.flatnonzero(rm)
            ccpos = np.flatnonzero(cm)
            exp[np.ix_(rr, ccpos)] = block
        return exp

    def dense_validity(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Boolean mask of cells whose two bins are valid."""
        i0, i1, j0, j1 = bbox
        v = self.valid
        return v[i0:i1][:, None] & v[j0:j1][None, :]

    def dense_same_chrom(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Boolean mask of same-chromosome cells."""
        i0, i1, j0, j1 = bbox
        cr = self.chrom_id[i0:i1]
        cc = self.chrom_id[j0:j1]
        return cr[:, None] == cc[None, :]

    def bias_vectors(
        self, i0: int, i1: int, j0: int, j1: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multiplicative balancing factors for the two axes (ones if raw)."""
        if self.bias is None:
            return (
                np.ones(i1 - i0, dtype=np.float64),
                np.ones(j1 - j0, dtype=np.float64),
            )
        return self.bias[i0:i1], self.bias[j0:j1]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def estimate_expected(
    clr,
    distance_edges,
    *,
    balance=False,
    divisive_weights=None,
    chunksize=100000,
    stabilize=False,
    **policy,
) -> ExpectedResult:
    """Estimate the intra-chromosomal distance background of a cooler.

    Parameters
    ----------
    clr : :class:`Cooler`
        Read-only archive.  Only ``symmetric-upper`` storage is supported;
        ``square`` files raise ``ValueError``.
    distance_edges : array-like of non-negative integers
        Bucket edges in bp, starting at 0, strictly increasing, with at least
        two values.  Bucket ``d`` is the half-open interval
        ``[edges[d], edges[d+1])`` of midpoint distance.
    balance : bool or str, optional
        ``False`` (default) estimates the background from raw counts.  ``True``
        uses the bin-table ``weight`` column; a string names another weight
        column.  Finite positive weights mark valid bins; NaN and zero weights
        exclude a bin; negative or infinite weights raise.
    divisive_weights : bool or None, optional
        Interpret the weights as divisive (``True``) or multiplicative
        (``False``).  ``None`` applies the same alias default as
        :meth:`Cooler.matrix`: columns named ``KR``, ``VC`` or ``VC_SQRT`` are
        divisive, everything else multiplicative.
    chunksize : positive int, optional
        Maximum number of pixel records read per storage access.  Default
        100000.
    stabilize : bool, optional
        Apply the robust local stabilization policy instead of returning raw
        bucket means.  Buckets without valid pairs are never filled.
    **policy
        Stabilization parameters: ``min_support`` (default 30), ``window``
        (7), ``huber_delta`` (1.345), ``spike_z`` (3.0),
        ``tail_monotone`` (True).  Unknown keywords raise ``TypeError``.

    Returns
    -------
    :class:`ExpectedResult`

    Notes
    -----
    Distance is the absolute difference of bin midpoints; with non-uniform
    bins this can be a half-integer bp value, handled exactly via doubled
    integer coordinates.  Opportunity counts include valid bin pairs that have
    no stored pixel, so implicit zeros are part of every denominator.
    """
    edges = _validate_edges(distance_edges)
    chunksize = _validate_chunksize(chunksize)
    stabilize = _validate_bool(stabilize, "stabilize")
    use_weights, weight_name, divisive = _resolve_weight_spec(balance, divisive_weights)

    allowed = {
        "min_support",
        "window",
        "huber_delta",
        "spike_z",
        "tail_monotone",
    }
    unknown = set(policy) - allowed
    if unknown:
        raise TypeError(
            f"unexpected stabilization policy parameters: {sorted(unknown)}"
        )
    full_policy = {k: v for k, v in DEFAULT_POLICY.items()}
    full_policy["stabilize"] = stabilize
    full_policy.update(policy)
    # Validate types of numeric policy parameters up front.
    for key in ("min_support", "window"):
        if not isinstance(full_policy[key], (int, np.integer)) or isinstance(
            full_policy[key], bool
        ):
            raise TypeError(f"{key} must be an integer")
    for key in ("huber_delta", "spike_z"):
        if not isinstance(full_policy[key], (int, float, np.integer, np.floating)):
            raise TypeError(f"{key} must be a number")
    if not isinstance(full_policy["tail_monotone"], bool):
        raise TypeError("tail_monotone must be a boolean")

    if clr.storage_mode == "square":
        raise ValueError(
            "Distance background estimation only supports symmetric-upper "
            "cooler storage; this file uses 'square' storage."
        )

    with open_hdf5(clr.store, **clr.open_kws) as h5:
        root = h5[clr.root]
        bins_df = clr.bins()[:]
        start = bins_df["start"].to_numpy(np.int64)
        end = bins_df["end"].to_numpy(np.int64)
        n_bins = len(bins_df)
        chrom_names = list(clr.chromnames)
        chrom_offsets = root["indexes"]["chrom_offset"][:].astype(np.int64)

        if use_weights:
            if weight_name not in root["bins"]:
                raise ValueError(
                    f"No column 'bins/{weight_name}' found. Set balance=False "
                    "or provide balancing weights."
                )
            raw_weights = root["bins"][weight_name][:]
            valid = _validity_from_weights(raw_weights)
            weight_values = np.asarray(raw_weights, dtype=np.float64).copy()
            # Bind against the raw stored vector, not the float64 copy.
            weight_binding = _weight_fingerprint(raw_weights)
        else:
            valid = np.ones(n_bins, dtype=bool)
            weight_values = None
            weight_binding = None

        bin_binding = _bin_fingerprint_from_h5(root)

        # Midpoint coordinates doubled, so half-integer midpoints stay exact.
        mid2 = (start + end).astype(np.int64)

        n_chroms = len(chrom_names)
        n_buckets = len(edges) - 1
        edges2 = (2 * edges).astype(np.int64)

        observed_sum = np.zeros((n_chroms, n_buckets), dtype=np.float64)
        raw_sum = np.zeros((n_chroms, n_buckets), dtype=np.float64)
        n_pairs = np.zeros((n_chroms, n_buckets), dtype=np.int64)

        # ---- Geometry: opportunity counts from the bin table alone ----
        for ci in range(n_chroms):
            lo, hi = int(chrom_offsets[ci]), int(chrom_offsets[ci + 1])
            if hi <= lo:
                continue
            vmask = valid[lo:hi]
            if not vmask.any():
                continue
            m = mid2[lo:hi][vmask]
            counts = _opportunity_counts(m, edges2)
            # Diagonal pairs (i == j): distance 0, always bucket 0.
            if edges2[0] <= 0 < edges2[1]:
                counts[0] += m.size
            n_pairs[ci] = counts

        # ---- Observations: stream stored pixels in bounded chunks ----
        if n_bins > 0:
            reader = CSRReader(root["pixels"], root["indexes"]["bin1_offset"][:])
            bbox = (0, n_bins, 0, n_bins)
            spans = reader.get_spans(bbox, chunksize)
            chrom_id = np.empty(n_bins, dtype=np.int64)
            for ci in range(n_chroms):
                chrom_id[chrom_offsets[ci] : chrom_offsets[ci + 1]] = ci

            for span in spans:
                dct = reader(["count"], bbox, span, False, False)
                b1 = dct["bin1_id"].astype(np.int64, copy=False)
                b2 = dct["bin2_id"].astype(np.int64, copy=False)
                count64 = dct["count"].astype(np.float64, copy=False)
                if np.any(~np.isfinite(count64)) or np.any(count64 < 0):
                    raise ValueError("pixel counts must be finite and non-negative")
                c1 = chrom_id[b1]
                c2 = chrom_id[b2]
                same = c1 == c2
                if not same.any():
                    continue
                b1 = b1[same]
                b2 = b2[same]
                c1 = c1[same]
                cnt = count64[same]
                both_valid = valid[b1] & valid[b2]
                if not both_valid.any():
                    continue
                b1 = b1[both_valid]
                b2 = b2[both_valid]
                c1 = c1[both_valid]
                cnt = cnt[both_valid]
                dist2 = np.abs(mid2[b2] - mid2[b1])
                bucket = _bucket_of(dist2, edges2)
                in_range = (bucket >= 0) & (bucket < n_buckets)
                if not in_range.any():
                    continue
                b1 = b1[in_range]
                b2 = b2[in_range]
                c1 = c1[in_range]
                bucket = bucket[in_range]
                cnt = cnt[in_range]
                np.add.at(raw_sum, (c1, bucket), cnt)
                if use_weights:
                    if divisive:
                        wobs = cnt / (weight_values[b1] * weight_values[b2])
                    else:
                        wobs = cnt * weight_values[b1] * weight_values[b2]
                else:
                    wobs = cnt
                np.add.at(observed_sum, (c1, bucket), wobs)

    # ---- Bucket-level summaries and expected values ----
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_mean = np.where(n_pairs > 0, raw_sum / np.maximum(n_pairs, 1), np.nan)
    # Buckets with pairs but zero observed total have an exact raw mean of 0.
    raw_mean[(n_pairs > 0) & (raw_sum == 0)] = 0.0

    expected = np.full((n_chroms, n_buckets), np.nan, dtype=np.float64)
    status = np.full((n_chroms, n_buckets), "unavailable", dtype=object)
    for ci in range(n_chroms):
        # The observed mean per valid pair: raw counts without weights, or
        # weighted counts otherwise. Buckets with zero pairs stay NaN.
        mean_obs = np.where(
            n_pairs[ci] > 0,
            observed_sum[ci] / np.maximum(n_pairs[ci], 1),
            np.nan,
        )
        mean_obs[(n_pairs[ci] > 0) & (observed_sum[ci] == 0)] = 0.0
        if stabilize:
            stab, _trend, st = _stabilize_chromosome(
                mean_obs,
                n_pairs[ci],
                min_support=full_policy["min_support"],
                window=full_policy["window"],
                huber_delta=full_policy["huber_delta"],
                spike_z=full_policy["spike_z"],
                tail_monotone=full_policy["tail_monotone"],
            )
            expected[ci] = stab
            status[ci] = st
        else:
            expected[ci] = mean_obs
            status[ci] = np.where(n_pairs[ci] > 0, "raw", "unavailable")

    # Finite non-negativity check for stabilized outputs.
    finite_vals = expected[np.isfinite(expected)]
    if np.any(finite_vals < 0):  # pragma: no cover - defensive
        raise RuntimeError("internal error: negative stabilized expectation")

    usable = np.isfinite(expected) & (expected > 0)
    # Unusable values are either NaN (no support) or exact zero (cannot be a
    # divisor); normalize all non-positive entries to NaN for query time.
    expected[~usable] = np.nan

    return ExpectedResult(
        distance_edges=edges,
        chrom_names=chrom_names,
        chrom_offsets=chrom_offsets,
        valid=valid,
        mid2=mid2,
        observed_sum=observed_sum,
        raw_sum=raw_sum,
        n_pairs=n_pairs,
        raw_mean=raw_mean,
        expected=expected,
        usable=usable,
        status=status,
        policy=full_policy,
        balance=use_weights,
        weight_name=weight_name,
        divisive_weights=divisive,
        weight_values=weight_values,
        bin_binding=bin_binding,
        weight_binding=weight_binding,
    )


# ---------------------------------------------------------------------------
# Persistence as named analysis layers
# ---------------------------------------------------------------------------


def _layer_path(root: h5py.Group, name: str) -> str:
    return f"{_EXPECTED_GRP}/{name}"


def save_expected(
    clr,
    result: ExpectedResult,
    name: str,
    *,
    overwrite: bool = False,
) -> str:
    """Store an :class:`ExpectedResult` as a named layer in the cooler group.

    The layer records the source bin layout, the full weight vector and its
    interpretation, the distance edges and the fitting policy.  It is written
    via a temporary group so a validation or write failure never damages an
    existing layer of the same name; overwriting requires ``overwrite=True``.

    Returns the HDF5 path of the layer.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("layer name must be a non-empty string")
    if "/" in name:
        raise ValueError("layer name must not contain '/'")

    with clr.open("r+") as root:
        target = _layer_path(root, name)
        if target in root and not overwrite:
            raise ValueError(
                f"Expected value layer {name!r} already exists; pass "
                "overwrite=True to replace it."
            )

        # Re-validate the in-memory result against the current archive before
        # touching any existing layer.
        result.check_bins(root)
        if result.balance:
            result.check_weights(root["bins"][result.weight_name][:])

        grp = root.require_group(_EXPECTED_GRP)
        tmp_name = f".tmp-{name}-{id(result)}"
        tmp_path = f"{_EXPECTED_GRP}/{tmp_name}"
        if tmp_path in root:
            del root[tmp_path]
        try:
            g = grp.create_group(tmp_name)
            g.create_dataset("distance_edges", data=result.distance_edges)
            g.create_dataset("chrom_offset", data=result.chrom_offsets)
            g.create_dataset(
                "chrom_name",
                data=np.array(result.chrom_names, dtype=h5py.string_dtype()),
            )
            g.create_dataset("valid", data=result.valid)
            g.create_dataset("mid2", data=result.mid2)
            g.create_dataset("observed_sum", data=result.observed_sum)
            g.create_dataset("raw_sum", data=result.raw_sum)
            g.create_dataset("n_pairs", data=result.n_pairs)
            g.create_dataset("raw_mean", data=result.raw_mean)
            g.create_dataset("expected", data=result.expected)
            g.create_dataset("usable", data=result.usable)
            g.create_dataset(
                "status",
                data=np.array(result.status.tolist(), dtype=object),
                dtype=h5py.string_dtype(),
            )
            if result.weight_values is not None:
                g.create_dataset("weight", data=result.weight_values)

            g.attrs["balance"] = bool(result.balance)
            g.attrs["weight_name"] = (
                result.weight_name if result.weight_name is not None else ""
            )
            g.attrs["divisive_weights"] = bool(result.divisive_weights)
            g.attrs["policy"] = json.dumps(result.policy)
            g.attrs["bin_binding"] = json.dumps(result.bin_binding)
            g.attrs["weight_binding"] = (
                json.dumps(result.weight_binding)
                if result.weight_binding is not None
                else ""
            )
            g.attrs["layer_format"] = 1

            if target in root:
                del root[target]
            root.move(tmp_path, target)
        except BaseException:
            if tmp_path in root:
                del root[tmp_path]
            raise
    result.layer_name = name
    return target


def list_expected(clr) -> list[str]:
    """Return the names of expected-value layers stored in the cooler."""
    with clr.open("r") as root:
        grp = root.get(_EXPECTED_GRP)
        if grp is None:
            return []
        return sorted(k for k in grp if not k.startswith("."))


def load_expected(clr, name: str) -> ExpectedResult:
    """Load a named expected-value layer and verify its bindings.

    The bin layout and weight vector are checked against the current archive;
    a mismatch raises ``ValueError``.  Moving or copying the file to another
    path does not affect loading, because bindings are content hashes.
    """
    with clr.open("r") as root:
        path = _layer_path(root, name)
        if path not in root:
            raise KeyError(f"No expected value layer named {name!r}")
        g = root[path]

        bin_fp = _bin_fingerprint_from_h5(root)
        stored_bin_fp = json.loads(g.attrs["bin_binding"])
        if bin_fp != stored_bin_fp:
            raise ValueError(
                "The bin layout of the cooler has changed since expected value "
                f"layer {name!r} was saved; the layer cannot be used."
            )

        balance = bool(g.attrs["balance"])
        weight_name = g.attrs["weight_name"] or None
        divisive = bool(g.attrs["divisive_weights"])
        if balance:
            if weight_name not in root["bins"]:
                raise ValueError(
                    f"Layer {name!r} requires weight column {weight_name!r}, "
                    "which is no longer present."
                )
            current = root["bins"][weight_name][:]
            stored_w = g["weight"][:]
            if _weight_fingerprint(current) != _weight_fingerprint(stored_w):
                raise ValueError(
                    "The balancing weights have changed since expected value "
                    f"layer {name!r} was saved; the layer cannot be used."
                )
            weight_values = stored_w
            weight_binding = json.loads(g.attrs["weight_binding"])
        else:
            weight_values = None
            weight_binding = None

        result = ExpectedResult(
            distance_edges=g["distance_edges"][:],
            chrom_names=[
                v.decode() if isinstance(v, bytes) else str(v)
                for v in g["chrom_name"][:]
            ],
            chrom_offsets=g["chrom_offset"][:],
            valid=g["valid"][:],
            mid2=g["mid2"][:],
            observed_sum=g["observed_sum"][:],
            raw_sum=g["raw_sum"][:],
            n_pairs=g["n_pairs"][:],
            raw_mean=g["raw_mean"][:],
            expected=g["expected"][:],
            usable=g["usable"][:],
            status=np.array(
                [
                    v.decode() if isinstance(v, bytes) else str(v)
                    for v in g["status"][:].flat
                ],
                dtype=object,
            ).reshape(g["status"].shape),
            policy=json.loads(g.attrs["policy"]),
            balance=balance,
            weight_name=weight_name,
            divisive_weights=divisive,
            weight_values=weight_values,
            bin_binding=stored_bin_fp,
            weight_binding=weight_binding,
            layer_name=name,
        )
    return result


def remove_expected(clr, name: str) -> None:
    """Delete a named expected-value layer from the cooler group."""
    with clr.open("r+") as root:
        path = _layer_path(root, name)
        if path not in root:
            raise KeyError(f"No expected value layer named {name!r}")
        del root[path]
