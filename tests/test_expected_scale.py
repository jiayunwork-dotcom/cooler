"""
Scale and memory accounting for distance-background estimation.

A generated archive with at least 100,000 bins and 1,000,000 stored pixels is
summarized into 64 distance buckets.  The tests instrument the pixel reader to
prove that (a) no bin-pair enumeration is hidden behind the sparse input,
(b) each storage read stays within the requested chunksize and (c) the
resident working set stays O(N + D + C).
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd
import pytest
from _expected_helpers import make_cooler

import cooler
from cooler.core import CSRReader

N_BINS = 100_000
N_PIXELS = 1_000_000
N_BUCKETS = 64
CHUNKSIZE = 100_000


def _build_large_cooler(path: str) -> cooler.Cooler:
    rng = np.random.default_rng(42)
    # A handful of chromosomes, uniform 10 kb bins.
    chrom_width = N_BINS // 4
    chroms = [f"chr{i}" for i in range(4)]
    bins = pd.DataFrame(
        {
            "chrom": np.repeat(chroms, chrom_width),
            "start": np.tile(np.arange(chrom_width) * 10_000, 4),
            "end": np.tile(np.arange(1, chrom_width + 1) * 10_000, 4),
        }
    )
    # Sparse upper-triangle pixels with a distance-decaying distribution.
    # Oversample because the same-chromosome filter below drops some draws.
    n_draw = int(N_PIXELS * 1.1)
    bin1 = rng.integers(0, N_BINS - 1, size=n_draw)
    gap = 1 + np.round(rng.pareto(1.5, size=n_draw) * 50).astype(np.int64)
    bin2 = np.minimum(bin1 + gap, N_BINS - 1)
    # Keep pixels within the same chromosome (distance background is
    # intra-chromosomal) and strictly upper triangular.
    chrom_of = bins["chrom"].to_numpy()
    keep = (bin2 > bin1) & (chrom_of[bin1] == chrom_of[bin2])
    bin1, bin2 = bin1[keep], bin2[keep]
    # Collapse any duplicate coordinate draws (combine via sum) so the pixel
    # table has one row per pair, then trim to the target pixel count.
    unique_pairs, inverse = np.unique(
        np.stack([bin1, bin2], axis=1), axis=0, return_inverse=True
    )
    drawn_counts = rng.integers(1, 50, size=bin1.size)
    agg_counts = np.bincount(inverse, weights=drawn_counts, minlength=len(unique_pairs))
    if len(unique_pairs) > N_PIXELS:
        unique_pairs = unique_pairs[:N_PIXELS]
        agg_counts = agg_counts[:N_PIXELS]
    bin1, bin2 = unique_pairs[:, 0], unique_pairs[:, 1]
    counts = agg_counts.astype(np.int64)
    pixels = pd.DataFrame({"bin1_id": bin1, "bin2_id": bin2, "count": counts})
    return make_cooler(path, bins, pixels)


@pytest.fixture(scope="module")
def large_clr(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("scale") / "large.cool")
    clr = _build_large_cooler(path)
    assert clr.shape[0] >= N_BINS
    assert clr.info["nnz"] >= N_PIXELS
    return clr


def test_large_archive_geometry_and_bounded_reads(large_clr):
    edges = np.linspace(0, 4_000_000, N_BUCKETS + 1).astype(np.int64)

    # Instrument the CSR reader: every pixel batch must stay within chunksize
    # records, and the full pair space is never enumerated (the number of
    # reads is far smaller than the number of valid bin pairs).
    max_batch = 0
    n_calls = 0
    original_call = CSRReader.__call__

    def tracked_call(
        self, field, bbox, row_span=None, reflect=False, return_index=False
    ):
        nonlocal max_batch, n_calls
        result = original_call(self, field, bbox, row_span, reflect, return_index)
        batch = result["bin1_id"].size
        max_batch = max(max_batch, batch)
        n_calls += 1
        return result

    CSRReader.__call__ = tracked_call
    tracemalloc.start()
    try:
        result = cooler.estimate_expected(large_clr, edges, chunksize=CHUNKSIZE)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        CSRReader.__call__ = original_call

    # Record the opportunity-count growth and the bounded read/memory budget
    # so the scaling behavior is visible in the test output.
    print(
        "\nscale stats: "
        f"nbins={large_clr.shape[0]} nnz={large_clr.info['nnz']} "
        f"buckets={N_BUCKETS} pixel_read_calls={n_calls} "
        f"max_batch_pixels={max_batch} (chunksize={CHUNKSIZE}) "
        f"opportunity_pairs={int(result.n_pairs.sum())} "
        f"peak_extra_mib={peak_bytes / 1024 / 1024:.1f}"
    )

    assert result.n_buckets == N_BUCKETS
    assert max_batch <= CHUNKSIZE
    # A handful of chunk reads covers a million pixels; enumerating all
    # i <= j pairs would require ~5e9 iterations instead.
    assert n_calls < 200

    # Opportunity counts are geometric and independent of pixel density.
    # Bucket 0 contains the diagonal pairs: every bin contributes exactly
    # one i == j pair, so the genome-wide total is the number of bins.
    # Off-diagonal pairs may also fall in bucket 0 for wide first edges.
    n_bins = large_clr.shape[0]
    diagonal_total = int(result.n_pairs[:, 0].sum())
    assert diagonal_total >= n_bins
    # The diagonal contribution alone can be checked with edges whose first
    # bucket is narrower than the smallest inter-bin distance.
    tight_edges = np.unique(
        np.concatenate(
            [
                [0, 5_000],
                np.linspace(5_000, 300_000_000, N_BUCKETS).astype(np.int64),
            ]
        )
    )
    tight = cooler.estimate_expected(large_clr, tight_edges, chunksize=CHUNKSIZE)
    assert int(tight.n_pairs[:, 0].sum()) == n_bins
    # Pair counts are geometric: non-negative, and the tight-edge layout
    # (whose last edge covers the whole chromosome span) accounts for every
    # i <= j pair exactly once, independent of which pixels are stored.
    assert np.all(result.n_pairs >= 0)
    chrom_widths = np.diff(tight.chrom_offsets)
    expected_pairs = int(sum(w * (w + 1) // 2 for w in chrom_widths))
    assert int(tight.n_pairs.sum()) == expected_pairs
    # Every stored pixel was accounted for: raw sums positive somewhere.
    assert result.raw_sum.sum() > 0

    # Re-running with a different chunksize must not change any statistic.
    result_small_chunk = cooler.estimate_expected(large_clr, edges, chunksize=1_000)
    np.testing.assert_array_equal(result_small_chunk.n_pairs, result.n_pairs)
    np.testing.assert_allclose(result_small_chunk.raw_sum, result.raw_sum, rtol=1e-12)


def test_large_archive_memory_is_linear(large_clr):
    edges = np.linspace(0, 4_000_000, N_BUCKETS + 1).astype(np.int64)
    tracemalloc.start()
    _baseline_current, baseline_peak = tracemalloc.get_traced_memory()
    result = cooler.estimate_expected(large_clr, edges, chunksize=CHUNKSIZE)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n = large_clr.shape[0]
    d = N_BUCKETS
    c = CHUNKSIZE
    # The resident working set must scale like N + D + C, not N^2: the peak
    # extra allocation is bounded by a small multiple of that sum rather
    # than by the quadratic number of bin pairs.
    linear_scale = (n + d + c) * 8
    extra_peak = peak - baseline_peak
    assert extra_peak < 25 * linear_scale, (
        f"peak extra memory {extra_peak / 1e6:.1f} MiB exceeds the linear "
        f"budget {25 * linear_scale / 1e6:.1f} MiB"
    )
    assert result.expected.shape == (4, N_BUCKETS)


def test_large_archive_query_does_not_load_whole_matrix(large_clr):
    edges = np.linspace(0, 4_000_000, N_BUCKETS + 1).astype(np.int64)
    result = cooler.estimate_expected(large_clr, edges, chunksize=CHUNKSIZE)
    cooler.save_expected(large_clr, result, "bg")

    loaded_reads = 0
    original_call = CSRReader.__call__

    def tracked_call(
        self, field, bbox, row_span=None, reflect=False, return_index=False
    ):
        nonlocal loaded_reads
        out = original_call(self, field, bbox, row_span, reflect, return_index)
        loaded_reads += out["bin1_id"].size
        return out

    CSRReader.__call__ = tracked_call
    try:
        # A small windowed read must touch only pixels inside that window.
        window = large_clr.matrix(expected="bg")[100:140, 100:140]
    finally:
        CSRReader.__call__ = original_call
    assert window.shape == (40, 40)
    assert loaded_reads < 10_000
