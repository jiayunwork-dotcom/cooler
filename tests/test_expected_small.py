"""
Exhaustive small-graph reference tests for distance-background estimation.

Every statistic is cross-checked against a plain O(n^2) enumeration of all
valid i <= j pairs (including implicit zeros), so these tests pin down the
exact semantics: non-uniform and holey bins, half-integer midpoint
distances, bucket-edge alignment, missing pixels, explicit zeros, masked
bins, the diagonal, empty chromosomes and inter-chromosomal cells.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from _expected_helpers import (
    make_cooler,
    reference_oe_matrix,
    reference_stats,
)

import cooler

RTOL = 1e-10
ATOL = 1e-12


def assert_nan_equal(actual: np.ndarray, expected: np.ndarray):
    """Element-wise compare with exact NaN positioning."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    nan_a = np.isnan(actual)
    nan_e = np.isnan(expected)
    assert np.array_equal(nan_a, nan_e), "NaN positions differ"
    np.testing.assert_allclose(actual[~nan_a], expected[~nan_e], rtol=RTOL, atol=ATOL)


# A deliberately awkward layout:
#   chr1: 4 non-uniform bins with a gap (hole) between bin 1 and bin 2
#   chr2: 3 uniform bins
#   chr3: no bins at all
# Midpoints (bp): 5, 15, 45, 65 | 5, 15, 25
BIN_TABLE = pd.DataFrame(
    {
        "chrom": ["chr1"] * 4 + ["chr2"] * 3,
        "start": [0, 10, 40, 60, 0, 10, 20],
        "end": [10, 20, 50, 70, 10, 20, 30],
    }
)
# Global bin offsets: chr1 = 0..3, chr2 = 4..6.
# Includes an explicit zero (4,5), a missing pixel (0,2), a diagonal, an
# inter-chromosomal pixel that the estimator must ignore and a cross-chrom
# pixel in the reverse row order that symmetric-upper still accepts
# (bin1_id < bin2_id always).
PIXEL_TABLE = pd.DataFrame(
    {
        "bin1_id": [0, 0, 1, 2, 3, 4, 4, 5, 0, 2],
        "bin2_id": [0, 1, 2, 2, 3, 4, 5, 6, 4, 5],
        "count": [5, 2, 3, 7, 1, 4, 0, 2, 9, 8],
    }
)
# edges chosen so half-integer distances and exact edge values occur:
# distances (mids): chr1 pairs -> 0, 10, 40, 60, 30, 50, 20
EDGES = [0, 10, 20, 40, 100]


@pytest.fixture
def clr(tmp_path):
    path = str(tmp_path / "small.cool")
    # BIN_TABLE has an intra-chromosomal gap, so it is a marked sparse layout.
    # chr3 appears in the chromosome catalog but carries no bins.
    chromsizes = {"chr1": 100, "chr2": 40, "chr3": 20}
    return make_cooler(path, BIN_TABLE, PIXEL_TABLE, chromsizes=chromsizes, sparse=True)


def test_edges_and_chunksize_validation(clr):
    for bad in ([1, 2, 3], [0, 0, 3], [0, -1, 3], [0], [0, 1.5, 3]):
        with pytest.raises((TypeError, ValueError)):
            cooler.estimate_expected(clr, bad)
    with pytest.raises(TypeError):
        cooler.estimate_expected(clr, np.array([0, 1], dtype=bool))
    for bad in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            cooler.estimate_expected(clr, EDGES, chunksize=bad)
    with pytest.raises(TypeError):
        cooler.estimate_expected(clr, EDGES, stabilize="yes")
    with pytest.raises(TypeError):
        cooler.estimate_expected(clr, EDGES, bogus_kwarg=1)


def test_square_storage_rejected(tmp_path):
    path = str(tmp_path / "square.cool")
    make_cooler(
        path,
        BIN_TABLE,
        PIXEL_TABLE,
        symmetric_upper=False,
        chromsizes={"chr1": 100, "chr2": 40, "chr3": 20},
        sparse=True,
    )
    with pytest.raises(ValueError, match="square"):
        cooler.estimate_expected(cooler.Cooler(path), EDGES)


def test_raw_stats_match_exhaustive_reference(clr):
    result = cooler.estimate_expected(clr, EDGES, chunksize=2)

    # Per chromosome against the independent enumeration.
    chrom_offsets = result.chrom_offsets
    for ci, _name in enumerate(result.chrom_names):
        lo, hi = int(chrom_offsets[ci]), int(chrom_offsets[ci + 1])
        if hi <= lo:
            # Empty chromosome: every bucket unsupported.
            assert np.all(result.n_pairs[ci] == 0)
            assert np.all(~result.usable[ci])
            assert set(np.unique(result.status[ci])) == {"unavailable"}
            continue
        sub_bins = BIN_TABLE.iloc[lo:hi].reset_index(drop=True)
        sub_pixels = PIXEL_TABLE[
            PIXEL_TABLE["bin1_id"].between(lo, hi - 1)
            & PIXEL_TABLE["bin2_id"].between(lo, hi - 1)
        ].copy()
        sub_pixels["bin1_id"] -= lo
        sub_pixels["bin2_id"] -= lo
        ref = reference_stats(sub_bins, sub_pixels, EDGES)
        np.testing.assert_array_equal(result.n_pairs[ci], ref["n_pairs"])
        np.testing.assert_allclose(
            result.raw_sum[ci], ref["raw_sum"], rtol=RTOL, atol=ATOL
        )
        assert_nan_equal(result.raw_mean[ci], ref["raw_mean"])
        np.testing.assert_allclose(
            result.observed_sum[ci], ref["observed_sum"], rtol=RTOL, atol=ATOL
        )
        # Raw mode: expected is exactly the raw mean where usable.
        np.testing.assert_allclose(
            result.expected[ci, result.usable[ci]],
            ref["raw_mean"][result.usable[ci]],
            rtol=RTOL,
            atol=ATOL,
        )
        assert set(np.unique(result.status[ci])) <= {"raw", "unavailable"}

    # chr1 explicit values, hand-computed from midpoints [5,15,45,65].
    # Pairs i<=j (10 total): 4 diagonals at distance 0; off-diagonal
    # distances are (0,1)=10 -> bucket 1; (2,3)=20 -> bucket 2; (1,2)=30
    # -> bucket 2; (0,2)=40, (1,3)=50, (0,3)=60 -> bucket 3 ([40,100)),
    # which has no stored observations.
    # Stored chr1 counts: (0,0)=5,(2,2)=7,(3,3)=1 diagonals; (0,1)=2;
    # (1,2)=3; (2,3)=1; diagonal (1,1) is an implicit zero. Bucket 2
    # ([20,40)) holds pair (2,3)=1; pair (1,2) has distance 30 but its
    # stored pixel (1,2)=3 is counted too -- see the exhaustive reference
    # loop above, which is the authoritative check. The hand values below
    # pin the bucket-0 and bucket-3 edge cases.
    np.testing.assert_array_equal(result.n_pairs[0], [4, 1, 2, 3])
    np.testing.assert_allclose(result.raw_sum[0], [13, 2, 3, 0])
    np.testing.assert_allclose(result.raw_mean[0], [13 / 4, 2.0, 3 / 2, 0.0])
    # The zero-mean bucket 3 has pairs but cannot be a divisor.
    assert not result.usable[0, 3]
    assert np.isnan(result.expected[0, 3])
    # chr2: explicit zero pixel (4,5) still counts as a pair with 0 signal.
    # Mids [5,15,25]: distances 0 (3 diag), 10 (two pairs), 20 (one pair);
    # bucket 3 ([40,100)) has no pairs.
    np.testing.assert_array_equal(result.n_pairs[1], [3, 2, 1, 0])
    np.testing.assert_allclose(result.raw_sum[1], [4, 2, 0, 0])
    np.testing.assert_allclose(result.raw_mean[1], [4 / 3, 1.0, 0.0, np.nan])
    # Zero mean bucket (chr2 bucket 2) is not a divisor.
    assert not result.usable[1, 2]
    assert np.isnan(result.expected[1, 2])


def test_half_integer_distances_and_exact_edges(tmp_path):
    # Bins [0,9] mid 4.5; [9,21] mid 15; [21,30] mid 25.5
    bins = pd.DataFrame({"chrom": ["c"] * 3, "start": [0, 9, 21], "end": [9, 21, 30]})
    pixels = pd.DataFrame(
        {"bin1_id": [0, 0, 1], "bin2_id": [0, 1, 2], "count": [1, 2, 4]}
    )
    path = str(tmp_path / "half.cool")
    clr = make_cooler(path, bins, pixels)
    # Distances: 0, 10.5, 10.5, 0, 21, 0. Edges 10 and 11 isolate 10.5.
    edges = [0, 10, 11, 21, 100]
    res = cooler.estimate_expected(clr, edges)
    # bucket [10,11): two pairs (0,1) and (1,2) both at 10.5 bp; pair (0,2)
    # at exactly 21 bp falls into [21,100); bucket [11,21) has no pairs.
    # Only diagonal (0,0) has a stored pixel (count 1); the other two
    # diagonal pairs are implicit zeros, so bucket-0 raw sum is 1, not 5.
    np.testing.assert_array_equal(res.n_pairs[0], [3, 2, 0, 1])
    np.testing.assert_allclose(res.raw_sum[0], [1, 6, 0, 0])
    np.testing.assert_allclose(res.raw_mean[0], [1 / 3, 3.0, np.nan, 0.0])
    # Distance exactly on an edge goes to the bucket starting at that edge:
    # pair (0,2) is 21 -> last bucket, which has no observation.
    assert res.raw_mean[0, 3] == 0.0
    assert not res.usable[0, 2]  # bucket [11,21) has no pairs


def test_chunksize_does_not_change_stats(clr):
    r1 = cooler.estimate_expected(clr, EDGES, chunksize=1)
    r5 = cooler.estimate_expected(clr, EDGES, chunksize=5)
    r100 = cooler.estimate_expected(clr, EDGES, chunksize=100)
    for r in (r1, r5, r100):
        np.testing.assert_array_equal(r.n_pairs, r1.n_pairs)
        assert_nan_equal(r.raw_mean, r1.raw_mean)
        assert_nan_equal(r.expected, r1.expected)


def test_pixel_storage_chunking_does_not_change_stats(tmp_path):
    # The same bin layout and pixel content written with different HDF5
    # storage chunk shapes must yield identical raw statistics: geometry
    # comes from the bin table, never from how pixels are partitioned.
    n_chroms = 16
    bins_per = 4
    n = n_chroms * bins_per
    bins = pd.DataFrame(
        {
            "chrom": np.repeat([f"c{k}" for k in range(n_chroms)], bins_per),
            "start": np.tile(np.arange(bins_per) * 10, n_chroms),
            "end": np.tile(np.arange(1, bins_per + 1) * 10, n_chroms),
        }
    )
    pixels = pd.DataFrame(
        {
            "bin1_id": np.arange(n - 1),
            "bin2_id": np.arange(1, n),
            "count": np.arange(1, n, dtype=np.int32),
        }
    )
    edges = [0, 10, 30, 100]
    path_a = str(tmp_path / "chunk_a.cool")
    path_b = str(tmp_path / "chunk_b.cool")
    # Pixel datasets only: small vs default/large storage chunks. All tables
    # have at least 16 rows, so a 16-row chunk is valid everywhere.
    cooler.create_cooler(path_a, bins, pixels, h5opts={"chunks": (4,)})
    cooler.create_cooler(path_b, bins, pixels)
    with h5py.File(path_a) as fa, h5py.File(path_b) as fb:
        assert fa["pixels/count"].chunks != fb["pixels/count"].chunks
    ra = cooler.estimate_expected(cooler.Cooler(path_a), edges)
    rb = cooler.estimate_expected(cooler.Cooler(path_b), edges)
    np.testing.assert_array_equal(ra.n_pairs, rb.n_pairs)
    np.testing.assert_array_equal(ra.raw_sum, rb.raw_sum)
    assert_nan_equal(ra.raw_mean, rb.raw_mean)
    assert_nan_equal(ra.expected, rb.expected)


def test_dense_sparse_pixel_consistency(clr):
    result = cooler.estimate_expected(clr, EDGES)
    dense = clr.matrix(balance=False, expected=result)[:]
    sparse = clr.matrix(balance=False, expected=result, sparse=True)[:].toarray()
    assert_nan_equal(sparse, dense)

    # A windowed read equals the corresponding slice of the whole read.
    w_dense = clr.matrix(balance=False, expected=result)[1:5, 2:6]
    assert_nan_equal(w_dense, dense[1:5, 2:6])
    w_sparse = clr.matrix(balance=False, expected=result, sparse=True)[
        1:5, 2:6
    ].toarray()
    assert_nan_equal(w_sparse, dense[1:5, 2:6])

    # Inter-chromosomal rectangle is all NaN.
    cross = clr.matrix(balance=False, expected=result)[0:4, 4:7]
    assert np.all(np.isnan(cross))

    # Empty windows.
    empty = clr.matrix(balance=False, expected=result)[2:2, 0:3]
    assert empty.shape == (0, 3)
    empty_sp = clr.matrix(balance=False, expected=result, sparse=True)[
        2:2, 0:3
    ].toarray()
    assert empty_sp.shape == (0, 3)

    # as_pixels only returns stored rows within the window, with extra cols.
    px = clr.matrix(balance=False, expected=result, as_pixels=True)[0:4, 0:4]
    assert {"count", "expected", "oe"} <= set(px.columns)
    stored = PIXEL_TABLE[
        PIXEL_TABLE["bin1_id"].between(0, 3) & PIXEL_TABLE["bin2_id"].between(0, 3)
    ]
    assert len(px) == len(stored)  # no fabricated symmetric pixels
    # Raw count column preserved.
    merged = px.merge(stored, on=["bin1_id", "bin2_id"], suffixes=("", "_ref"))
    np.testing.assert_array_equal(merged["count"], merged["count_ref"])


def test_oe_values_match_reference(clr):
    result = cooler.estimate_expected(clr, EDGES)
    dense = clr.matrix(balance=False, expected=result)[:]
    for ci, _name in enumerate(result.chrom_names):
        lo, hi = int(result.chrom_offsets[ci]), int(result.chrom_offsets[ci + 1])
        if hi <= lo:
            continue
        sub_bins = BIN_TABLE.iloc[lo:hi].reset_index(drop=True)
        sub_pixels = PIXEL_TABLE[
            PIXEL_TABLE["bin1_id"].between(lo, hi - 1)
            & PIXEL_TABLE["bin2_id"].between(lo, hi - 1)
        ].copy()
        sub_pixels["bin1_id"] -= lo
        sub_pixels["bin2_id"] -= lo
        ref = reference_oe_matrix(
            sub_bins,
            sub_pixels,
            EDGES,
            result.expected[ci],
            result.usable[ci],
        )
        assert_nan_equal(dense[lo:hi, lo:hi], ref)

    # Inter-chrom cells and unsupported-bucket cells are NaN.
    assert np.all(np.isnan(dense[0:4, 4:7]))
    # Valid implicit zero cell with a usable divisor: chr1 diagonal (1,1)
    # has no stored pixel but bucket 0 is usable (expected 13/4) -> OE 0.
    assert dense[1, 1] == 0.0


def test_implicit_zero_not_confused_with_nan(clr):
    result = cooler.estimate_expected(clr, EDGES)
    dense = clr.matrix(balance=False, expected=result)[:]
    # Valid implicit zero cell with a usable expected value:
    # chr1 pair (1,1) is an implicit-zero diagonal in bucket 0 (usable).
    assert dense[1, 1] == 0.0
    # Cells in bucket 3 (distance >= 40) have pairs but zero raw mean, so the
    # bucket is unusable and valid cells there are NaN, not zero.
    assert np.isnan(dense[0, 2])  # distance 40, bucket 3
    assert np.isnan(dense[2, 0])
    # A cell whose bucket is unusable stays NaN even though valid.
    # chr2 pair (4,6): distance 20 -> bucket 2, raw mean zero -> unusable.
    assert np.isnan(dense[4, 6])
    assert np.isnan(dense[6, 4])
    # Sparse representation must keep NaN explicit rather than as zeros.
    sp = clr.matrix(balance=False, expected=result, sparse=True)[:]
    sp_dense = sp.toarray()
    assert_nan_equal(sp_dense, dense)
    # NaN entries are explicit in the COO data.
    assert np.isnan(sp.data).any()


def test_diagonal_counted_once(clr):
    result = cooler.estimate_expected(clr, EDGES)
    # Diagonal pairs land in bucket 0; n_pairs bucket 0 equals number of
    # valid bins per chromosome (no double counting of i==j).
    assert result.n_pairs[0, 0] == 4
    assert result.n_pairs[1, 0] == 3


def test_pixelless_map_keeps_geometric_denominator(tmp_path):
    empty_pixels = PIXEL_TABLE.iloc[0:0]
    chromsizes = {"chr1": 100, "chr2": 40, "chr3": 20}
    path = str(tmp_path / "empty.cool")
    clr = make_cooler(path, BIN_TABLE, empty_pixels, chromsizes=chromsizes, sparse=True)
    res = cooler.estimate_expected(clr, EDGES)
    # Opportunity counts identical to the populated map.
    populated = make_cooler(
        str(tmp_path / "full.cool"),
        BIN_TABLE,
        PIXEL_TABLE,
        chromsizes=chromsizes,
        sparse=True,
    )
    ref = cooler.estimate_expected(populated, EDGES)
    np.testing.assert_array_equal(res.n_pairs, ref.n_pairs)
    assert np.all(res.raw_sum == 0)
    assert np.all(res.raw_mean[res.n_pairs > 0] == 0)
    assert not np.any(res.usable)
    # No divisor available: whole OE matrix is NaN.
    dense = clr.matrix(balance=False, expected=res)[:]
    assert np.all(np.isnan(dense))


def test_non_count_field_rejected(clr):
    result = cooler.estimate_expected(clr, EDGES)
    with pytest.raises(ValueError):
        clr.matrix(field="foo", expected=result)


def test_summary_table_contents(clr):
    result = cooler.estimate_expected(clr, EDGES)
    summary = result.summary()
    assert list(summary.columns) == [
        "chrom",
        "distance_lo",
        "distance_hi",
        "observed_sum",
        "n_pairs",
        "raw_sum",
        "raw_mean",
        "expected",
        "usable",
        "status",
    ]
    assert len(summary) == result.n_buckets * len(result.chrom_names)
    chr1 = result.summary("chr1")
    assert (chr1["chrom"] == "chr1").all()
    with pytest.raises(KeyError):
        result.summary("nope")
