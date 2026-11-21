"""
Regression tests for the cooler merge chain: exact integer counts, bounded
merge buffers, axis validation, sparse layouts, empty inputs, transactional
failure handling, and the CLI.

The arbitrary-precision reference sums in these tests are computed with
plain Python integers, independently of the implementation under test.
"""

import os
import resource
import subprocess
import sys
import tracemalloc

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import cooler
from cooler import create_cooler, merge_coolers
from cooler._reduce import CoolerMerger
from cooler.cli.merge import merge as merge_cli
from cooler.create import BadInputError

# The values from the original report: two int64 pixels whose sum
# (9223372036854775810) overflows int64 and used to wrap to a negative count
BIG_INT64 = 4611686018427387905
INT64_MAX = np.iinfo(np.int64).max
UINT64_MAX = np.iinfo(np.uint64).max

BINS2 = pd.DataFrame({"chrom": ["a", "a"], "start": [0, 10], "end": [10, 20]})


def make_pixels(pixels, dtype=np.int64):
    """A pixel DataFrame from {(bin1, bin2): value}."""
    keys = sorted(pixels)
    return pd.DataFrame(
        {
            "bin1_id": np.array([k[0] for k in keys], dtype=np.int64),
            "bin2_id": np.array([k[1] for k in keys], dtype=np.int64),
            "count": np.array([pixels[k] for k in keys], dtype=dtype),
        }
    )


def make_cooler(path, pixels, bins=BINS2, dtype=np.int64, **kwargs):
    df = pixels if isinstance(pixels, pd.DataFrame) else make_pixels(pixels, dtype)
    create_cooler(str(path), bins, df, **kwargs)
    return str(path)


def empty_pixels():
    return pd.DataFrame(
        {
            "bin1_id": np.array([], dtype=np.int64),
            "bin2_id": np.array([], dtype=np.int64),
            "count": np.array([], dtype=np.int64),
        }
    )


def reference_sum(inputs):
    """Arbitrary-precision merged counts: {(bin1, bin2): exact total}."""
    total = {}
    for px in inputs:
        for b1, b2, c in zip(px["bin1_id"], px["bin2_id"], px["count"]):
            key = (int(b1), int(b2))
            total[key] = total.get(key, 0) + int(c)
    return total


def read_pixels(uri):
    """{(bin1, bin2): value as Python int} read back from a cooler."""
    px = cooler.Cooler(uri).pixels()[:]
    return {
        (int(b1), int(b2)): int(c)
        for b1, b2, c in zip(px["bin1_id"], px["bin2_id"], px["count"])
    }


def check_indexes(uri):
    """The index tables must be consistent with the bin and pixel tables."""
    clr = cooler.Cooler(uri)
    with clr.open("r") as grp:
        bin1_offset = grp["indexes/bin1_offset"][:]
        chrom_offset = grp["indexes/chrom_offset"][:]
        bin1 = grp["pixels/bin1_id"][:]
        chrom = grp["bins/chrom"][:]
    nbins = clr.info["nbins"]
    assert len(bin1_offset) == nbins + 1
    assert bin1_offset[-1] == clr.info["nnz"]
    assert np.all(np.diff(bin1_offset) >= 0)
    # Every pixel's bin1 falls in its offset range
    for b in np.unique(bin1):
        lo, hi = bin1_offset[b], bin1_offset[b + 1]
        assert np.all(bin1[lo:hi] == b)
    assert len(chrom_offset) == len(clr.chromnames) + 1
    assert chrom_offset[-1] == nbins
    for cid in np.unique(chrom):
        lo, hi = chrom_offset[cid], chrom_offset[cid + 1]
        assert np.all(chrom[lo:hi] == cid)


# ---------------------------------------------------------------------------
# Exact integer sums
# ---------------------------------------------------------------------------


def test_merge_int64_overflow_raises(tmp_path):
    # The original report: two legal int64 coolers whose sum at (0, 1)
    # doesn't fit in int64 must fail, not wrap to -9223372036854775806
    px = make_pixels({(0, 1): BIG_INT64})
    uris = [
        make_cooler(tmp_path / f"m{i}.cool", px, dtypes={"count": "int64"})
        for i in range(2)
    ]
    out = str(tmp_path / "out.cool")
    with pytest.raises(OverflowError, match="count"):
        merge_coolers(out, uris, mergebuf=100, dtypes={"count": "int64"})
    assert not os.path.exists(out)

    # The same two contributions as two chunks of create_cooler(ordered=False)
    chunks = [px.copy(), px.copy()]
    with pytest.raises(OverflowError, match="count"):
        create_cooler(
            str(tmp_path / "c.cool"),
            BINS2,
            iter(chunks),
            ordered=False,
            dtypes={"count": "int64"},
        )
    assert not os.path.exists(tmp_path / "c.cool")

    # And as a single in-memory table with a repeated pixel
    with pytest.raises(OverflowError, match="count"):
        create_cooler(
            str(tmp_path / "d.cool"),
            BINS2,
            pd.concat(chunks, ignore_index=True),
            ordered=False,
            dtypes={"count": "int64"},
        )


def test_merge_uint64_explicit_succeeds(tmp_path):
    px = make_pixels({(0, 1): BIG_INT64})
    uris = [
        make_cooler(tmp_path / f"m{i}.cool", px, dtypes={"count": "int64"})
        for i in range(2)
    ]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100, dtypes={"count": "uint64"})
    clr = cooler.Cooler(out)
    stored = clr.pixels()[:]
    assert stored["count"].dtype == np.uint64
    assert int(stored["count"][0]) == 9223372036854775810
    assert clr.info["sum"] == 9223372036854775810
    check_indexes(out)


@pytest.mark.parametrize(
    "values,dtype,ok",
    [
        # int64 boundaries
        ([INT64_MAX, 1], "int64", False),
        ([INT64_MAX, 0], "int64", True),
        ([INT64_MAX - 1, 1], "int64", True),
        # uint64 boundaries
        ([UINT64_MAX, 1], "uint64", False),
        ([UINT64_MAX, 0], "uint64", True),
        ([UINT64_MAX - 1, 1], "uint64", True),
        # int32 inferred from int32 inputs
        ([2**31 - 1, 1], "int32", False),
        ([2**31 - 1, 0], "int32", True),
    ],
)
def test_merge_dtype_boundaries(tmp_path, values, dtype, ok):
    # Two inputs, one pixel each at the same key; the sum is checked
    # against an arbitrary-precision reference
    uris = []
    for i, v in enumerate(values):
        px = make_pixels({(0, 1): v}, dtype=np.array([v]).dtype)
        uris.append(
            make_cooler(
                tmp_path / f"m{i}.cool", px, dtypes={"count": np.array([v]).dtype}
            )
        )
    out = str(tmp_path / "out.cool")
    total = sum(values)  # Python int: exact
    if not ok:
        with pytest.raises(OverflowError):
            merge_coolers(out, uris, mergebuf=100, dtypes={"count": dtype})
        assert not os.path.exists(out)
    else:
        merge_coolers(out, uris, mergebuf=100, dtypes={"count": dtype})
        assert read_pixels(out) == {(0, 1): total}
        assert cooler.Cooler(out).info["sum"] == total


def test_merge_default_dtype_inference(tmp_path):
    # int32 + int32 -> int32 output when the sums fit
    px = make_pixels({(0, 1): 5})
    uris = [make_cooler(tmp_path / f"m{i}.cool", px) for i in range(2)]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100)
    assert cooler.Cooler(out).pixels()["count"][:].dtype == np.int32
    assert read_pixels(out) == {(0, 1): 10}

    # int32 + int64 -> int64
    px64 = make_pixels({(0, 1): 2**40}, dtype=np.int64)
    make_cooler(tmp_path / "m64.cool", px64, dtypes={"count": "int64"})
    out = str(tmp_path / "out64.cool")
    merge_coolers(out, [uris[0], str(tmp_path / "m64.cool")], mergebuf=100)
    assert cooler.Cooler(out).pixels()["count"][:].dtype == np.int64
    assert read_pixels(out) == {(0, 1): 2**40 + 5}

    # int32 sums that overflow the inferred int32 raise and advise int64
    big = make_pixels({(0, 1): 2**31 - 1})
    uris = [make_cooler(tmp_path / f"b{i}.cool", big) for i in range(2)]
    with pytest.raises(OverflowError, match="int64"):
        merge_coolers(str(tmp_path / "x.cool"), uris, mergebuf=100)
    # ... and succeed when the caller widens the dtype
    merge_coolers(
        str(tmp_path / "ok.cool"), uris, mergebuf=100, dtypes={"count": "int64"}
    )
    assert read_pixels(str(tmp_path / "ok.cool")) == {(0, 1): 2 * (2**31 - 1)}


def test_merge_table_sum_exceeds_uint64(tmp_path):
    # Every pixel fits in uint64, but the whole-table sum (2**64) does not:
    # the exact total is still stored and readable through info['sum']
    px = make_pixels({(0, 0): 2**63, (0, 1): 2**63}, dtype=np.uint64)
    uri = make_cooler(tmp_path / "a.cool", px, dtypes={"count": "uint64"})
    out = str(tmp_path / "out.cool")
    merge_coolers(out, [uri], mergebuf=100, dtypes={"count": "uint64"})
    clr = cooler.Cooler(out)
    assert clr.info["sum"] == 2**64
    assert isinstance(clr.info["sum"], int)
    assert read_pixels(out) == {(0, 0): 2**63, (0, 1): 2**63}

    # Storage is lossless: too big for int64/uint64 attributes, the sum is
    # kept as a decimal string that info() parses back to an exact int
    with h5py.File(out, "r") as f:
        assert f.attrs["sum"] == str(2**64)

    # Larger still: 3 * 2**63, via the CLI readback
    bins3 = pd.DataFrame(
        {"chrom": ["a"] * 3, "start": [0, 10, 20], "end": [10, 20, 30]}
    )
    px3 = pd.DataFrame(
        {
            "bin1_id": [0, 0, 1],
            "bin2_id": [0, 1, 2],
            "count": np.array([2**63] * 3, dtype=np.uint64),
        }
    )
    create_cooler(str(tmp_path / "b.cool"), bins3, px3, dtypes={"count": "uint64"})
    assert cooler.Cooler(str(tmp_path / "b.cool")).info["sum"] == 3 * 2**63
    result = CliRunner().invoke(
        __import__("cooler.cli.info", fromlist=["info"]).info,
        [str(tmp_path / "b.cool"), "-f", "sum"],
    )
    assert result.exit_code == 0, result.output
    assert int(result.output.strip()) == 3 * 2**63


def test_merge_intermediate_round_overflow(tmp_path):
    # max_merge forces multi-round temporary merges; an overflow in an
    # intermediate round must be caught, not written through
    px = make_pixels({(0, 1): 2**63}, dtype=np.uint64)
    chunks = [px.copy() for _ in range(3)]
    with pytest.raises(OverflowError):
        create_cooler(
            str(tmp_path / "o.cool"),
            BINS2,
            chunks,
            ordered=False,
            max_merge=2,
            dtypes={"count": "uint64"},
        )
    assert not os.path.exists(tmp_path / "o.cool")


def test_merge_min_max_agg(tmp_path):
    # min/max keep their meaning; a missing pixel contributes nothing,
    # in particular not a zero
    a = make_pixels({(0, 0): -5, (0, 1): 7})
    b = make_pixels({(0, 1): 3})
    ua = make_cooler(tmp_path / "a.cool", a, dtypes={"count": "int64"})
    ub = make_cooler(tmp_path / "b.cool", b, dtypes={"count": "int64"})
    out = str(tmp_path / "out.cool")
    merge_coolers(out, [ua, ub], mergebuf=100, agg={"count": "min"})
    assert read_pixels(out) == {(0, 0): -5, (0, 1): 3}
    merge_coolers(
        str(tmp_path / "out2.cool"), [ua, ub], mergebuf=100, agg={"count": "max"}
    )
    assert read_pixels(str(tmp_path / "out2.cool")) == {(0, 0): -5, (0, 1): 7}


# ---------------------------------------------------------------------------
# Axis validation
# ---------------------------------------------------------------------------


def test_merge_rejects_different_chrom_lengths(tmp_path):
    # Identical bin rows, but the catalogs disagree on the chromosome length
    bins = pd.DataFrame({"chrom": ["a", "a"], "start": [10, 40], "end": [20, 60]})
    px = make_pixels({(0, 1): 1})
    ua = make_cooler(
        tmp_path / "a.cool", px, bins=bins, chromsizes={"a": 100}, sparse=True
    )
    ub = make_cooler(
        tmp_path / "b.cool", px, bins=bins, chromsizes={"a": 110}, sparse=True
    )
    out = str(tmp_path / "out.cool")
    with pytest.raises(ValueError, match="chromosome lengths"):
        merge_coolers(out, [ua, ub], mergebuf=100)
    assert not os.path.exists(out)  # rejected before the target is created


def test_merge_rejects_different_bins_or_order(tmp_path):
    px = make_pixels({(0, 1): 1})
    ua = make_cooler(tmp_path / "a.cool", px)
    # different coordinates
    ub = make_cooler(
        tmp_path / "b.cool",
        px,
        bins=pd.DataFrame({"chrom": ["a", "a"], "start": [0, 5], "end": [5, 20]}),
    )
    with pytest.raises(ValueError, match="bin table"):
        merge_coolers(str(tmp_path / "o1.cool"), [ua, ub], mergebuf=100)
    # different chromosome order in the catalog
    bins_ab = pd.DataFrame({"chrom": ["a", "b"], "start": [0, 0], "end": [10, 10]})
    px2 = make_pixels({(0, 1): 1})
    uc = make_cooler(tmp_path / "c.cool", px2, bins=bins_ab)
    bins_ba = pd.DataFrame({"chrom": ["b", "a"], "start": [0, 0], "end": [10, 10]})
    ud = make_cooler(tmp_path / "d.cool", px2, bins=bins_ba)
    with pytest.raises(ValueError, match="chromosomes"):
        merge_coolers(str(tmp_path / "o2.cool"), [uc, ud], mergebuf=100)
    # a chromosome with no bins is still part of the axis
    sizes = pd.Series({"a": 10, "b": 10, "c": 5})
    ue = make_cooler(
        tmp_path / "e.cool", px2, bins=bins_ab, chromsizes=sizes, sparse=True
    )
    with pytest.raises(ValueError, match="chromosomes"):
        merge_coolers(str(tmp_path / "o3.cool"), [uc, ue], mergebuf=100)


def test_merge_axis_integer_width_agnostic(tmp_path):
    # The same axis stored with different integer widths is the same axis
    px = make_pixels({(0, 1): 3})
    ua = make_cooler(tmp_path / "a.cool", px)
    ub = make_cooler(tmp_path / "b.cool", px)
    with h5py.File(ub, "r+") as f:
        for path in ("chroms/length", "bins/start", "bins/end"):
            data = f[path][:]
            assert data.dtype == np.int32
            del f[path]
            f.create_dataset(path, data=data.astype(np.int64))
    out = str(tmp_path / "out.cool")
    merge_coolers(out, [ua, ub], mergebuf=100)
    assert read_pixels(out) == {(0, 1): 6}


# ---------------------------------------------------------------------------
# Sparse layouts
# ---------------------------------------------------------------------------

SPARSE_BINS = pd.DataFrame({"chrom": ["a", "a"], "start": [10, 40], "end": [20, 60]})
SPARSE_SIZES = pd.Series({"a": 100, "b": 70})  # 'b' has no bins


def test_merge_sparse_layout_preserved(tmp_path):
    px = make_pixels({(0, 1): 5})
    uris = [
        make_cooler(
            tmp_path / f"s{i}.cool",
            px,
            bins=SPARSE_BINS,
            chromsizes=SPARSE_SIZES,
            sparse=True,
        )
        for i in range(2)
    ]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100)
    clr = cooler.Cooler(out)
    # The full catalog, the holes and the layout marking are preserved
    assert clr.chromnames == ["a", "b"]
    assert list(clr.chromsizes) == [100, 70]
    assert clr.info.get("bin-layout") == "sparse"
    assert clr.bins()[:][["chrom", "start", "end"]].values.tolist() == [
        ["a", 10, 20],
        ["a", 40, 60],
    ]
    assert read_pixels(out) == {(0, 1): 10}
    assert clr.info["sum"] == 10
    check_indexes(out)


def test_merge_zero_bin_sparse(tmp_path):
    ebins = SPARSE_BINS.iloc[:0]
    uris = [
        make_cooler(
            tmp_path / f"z{i}.cool",
            empty_pixels(),
            bins=ebins,
            chromsizes=SPARSE_SIZES,
            sparse=True,
        )
        for i in range(2)
    ]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100)
    clr = cooler.Cooler(out)
    assert clr.info["nbins"] == 0
    assert clr.info["nnz"] == 0
    assert clr.info["sum"] == 0
    assert clr.chromnames == ["a", "b"]
    assert clr.info.get("bin-layout") == "sparse"
    check_indexes(out)


# ---------------------------------------------------------------------------
# Empty pixel tables
# ---------------------------------------------------------------------------


def test_merge_all_empty(tmp_path):
    uris = [make_cooler(tmp_path / f"e{i}.cool", empty_pixels()) for i in range(3)]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100)
    clr = cooler.Cooler(out)
    assert clr.info["nnz"] == 0
    assert clr.info["sum"] == 0
    assert clr.chromnames == ["a"]
    assert clr.info["nbins"] == 2
    check_indexes(out)


def test_merge_empty_among_nonempty(tmp_path):
    px = make_pixels({(0, 1): 4})
    paths = [
        make_cooler(tmp_path / "n1.cool", px),
        make_cooler(tmp_path / "e1.cool", empty_pixels()),
        make_cooler(tmp_path / "e2.cool", empty_pixels()),
        make_cooler(tmp_path / "e3.cool", empty_pixels()),
        make_cooler(tmp_path / "n2.cool", px),
    ]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, paths, mergebuf=100)
    assert read_pixels(out) == {(0, 1): 8}
    assert cooler.Cooler(out).info["sum"] == 8


def test_merge_single_input(tmp_path):
    uri = make_cooler(tmp_path / "one.cool", make_pixels({(0, 1): 7}))
    out = str(tmp_path / "out.cool")
    merge_coolers(out, [uri], mergebuf=100)
    assert read_pixels(out) == {(0, 1): 7}
    check_indexes(out)


def test_create_unordered_all_empty(tmp_path):
    out = str(tmp_path / "out.cool")
    create_cooler(out, BINS2, [empty_pixels(), empty_pixels()], ordered=False)
    clr = cooler.Cooler(out)
    assert clr.info["nnz"] == 0
    assert clr.info["sum"] == 0
    check_indexes(out)


# ---------------------------------------------------------------------------
# Storage modes
# ---------------------------------------------------------------------------


def test_merge_symmetric_upper_not_mirrored(tmp_path):
    px = make_pixels({(0, 1): 3})
    uris = [make_cooler(tmp_path / f"s{i}.cool", px) for i in range(2)]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100)
    clr = cooler.Cooler(out)
    assert clr.info["storage-mode"] == "symmetric-upper"
    # the mirror pixel is not counted again
    assert read_pixels(out) == {(0, 1): 6}
    assert clr.info["sum"] == 6


def test_merge_square_keeps_direction(tmp_path):
    px = make_pixels({(1, 0): 3, (0, 1): 5})
    uris = [
        make_cooler(tmp_path / f"q{i}.cool", px, symmetric_upper=False)
        for i in range(2)
    ]
    out = str(tmp_path / "out.cool")
    merge_coolers(out, uris, mergebuf=100)
    clr = cooler.Cooler(out)
    assert clr.info["storage-mode"] == "square"
    assert read_pixels(out) == {(1, 0): 6, (0, 1): 10}
    assert clr.info["sum"] == 16


def test_merge_mixed_storage_modes_rejected(tmp_path):
    px = make_pixels({(0, 1): 1})
    ua = make_cooler(tmp_path / "a.cool", px)
    ub = make_cooler(tmp_path / "b.cool", px, symmetric_upper=False)
    with pytest.raises(ValueError, match="symmetric"):
        merge_coolers(str(tmp_path / "o.cool"), [ua, ub], mergebuf=100)


# ---------------------------------------------------------------------------
# Duplicates and multi-round merges
# ---------------------------------------------------------------------------


def one_pixel(b1, b2, c):
    return make_pixels({(b1, b2): c})


def test_duplicates_error_within_and_across_chunks(tmp_path):
    with pytest.raises(BadInputError, match="duplicate"):
        create_cooler(
            str(tmp_path / "a.cool"),
            BINS2,
            [pd.concat([one_pixel(0, 1, 1), one_pixel(0, 1, 2)])],
            ordered=False,
            duplicates="error",
        )
    with pytest.raises(BadInputError, match="duplicate"):
        create_cooler(
            str(tmp_path / "b.cool"),
            BINS2,
            [one_pixel(0, 1, 1), one_pixel(0, 1, 2)],
            ordered=False,
            duplicates="error",
        )


def test_duplicates_error_across_merge_rounds(tmp_path):
    # max_merge=2 forces a second merge round; the repeated pixel only
    # meets its duplicate there, and must still be caught
    chunks = [one_pixel(0, 1, 1), one_pixel(1, 1, 1), one_pixel(0, 1, 2)]
    with pytest.raises(BadInputError, match="duplicate"):
        create_cooler(
            str(tmp_path / "c.cool"),
            BINS2,
            chunks,
            ordered=False,
            duplicates="error",
            max_merge=2,
        )


def test_combine_duplicates_exact_any_chunking(tmp_path):
    bins = pd.DataFrame(
        {"chrom": ["a"] * 5, "start": np.arange(5) * 10, "end": (np.arange(5) + 1) * 10}
    )
    rng = np.random.default_rng(0)
    records = [
        (int(rng.integers(0, 4)), int(rng.integers(0, 5)), int(rng.integers(1, 50)))
        for _ in range(200)
    ]
    records = [(min(a, b), max(a, b), c) for a, b, c in records]
    ref = {}
    for a, b, c in records:
        ref[(a, b)] = ref.get((a, b), 0) + c
    df = pd.DataFrame(records, columns=["bin1_id", "bin2_id", "count"])

    # one chunk vs many chunks vs different merge rounds: identical result
    def split(df, n):
        return [df.iloc[i::n].copy() for i in range(n)]

    variants = [
        {"chunks": [df], "max_merge": 200},
        {"chunks": split(df, 7), "max_merge": 200},
        {"chunks": split(df, 7), "max_merge": 2},
        {"chunks": split(df, 50), "max_merge": 3},
    ]
    results = []
    for i, v in enumerate(variants):
        out = str(tmp_path / f"v{i}.cool")
        chunks = [c.copy() for c in v["chunks"]]
        create_cooler(out, bins, chunks, ordered=False, max_merge=v["max_merge"])
        results.append(read_pixels(out))
    for r in results:
        assert r == ref


def test_mergebuf_capped_extra_rounds(tmp_path):
    # mergebuf smaller than the number of runs forces additional temporary
    # merge rounds; the counts stay exact
    bins = pd.DataFrame(
        {"chrom": ["a"] * 5, "start": np.arange(5) * 10, "end": (np.arange(5) + 1) * 10}
    )
    chunks = [one_pixel(i % 3, 4, 1) for i in range(10)]
    out = str(tmp_path / "o.cool")
    create_cooler(out, bins, chunks, ordered=False, mergebuf=4)
    assert read_pixels(out) == {(0, 4): 4, (1, 4): 3, (2, 4): 3}


def test_mergebuf_smaller_than_inputs_raises(tmp_path):
    px = make_pixels({(0, 1): 1})
    uris = [make_cooler(tmp_path / f"m{i}.cool", px) for i in range(3)]
    with pytest.raises(ValueError, match="mergebuf"):
        merge_coolers(str(tmp_path / "o.cool"), uris, mergebuf=2)
    with pytest.raises(ValueError, match="mergebuf"):
        merge_coolers(str(tmp_path / "o.cool"), uris, mergebuf=0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_merge_order_and_buffering_invariant(tmp_path):
    bins = pd.DataFrame(
        {"chrom": ["a"] * 6, "start": np.arange(6) * 10, "end": (np.arange(6) + 1) * 10}
    )
    rng = np.random.default_rng(3)
    inputs = []
    for _ in range(3):
        n = 50
        b1 = rng.integers(0, 5, n)
        b2 = rng.integers(0, 6, n)
        df = pd.DataFrame(
            {"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2)}
        )
        df = df.drop_duplicates().sort_values(["bin1_id", "bin2_id"])
        df["count"] = rng.integers(1, 1000, len(df))
        inputs.append(df)
    uris = [
        make_cooler(tmp_path / f"i{k}.cool", df, bins=bins)
        for k, df in enumerate(inputs)
    ]
    ref = reference_sum(inputs)

    results = []
    for mergebuf in (3, 257, 100_000):
        out = str(tmp_path / f"b{mergebuf}.cool")
        merge_coolers(out, uris, mergebuf=mergebuf)
        results.append(read_pixels(out))
    # input order doesn't matter either
    merge_coolers(str(tmp_path / "rev.cool"), uris[::-1], mergebuf=5)
    results.append(read_pixels(str(tmp_path / "rev.cool")))
    for r in results:
        assert r == ref


# ---------------------------------------------------------------------------
# Bounded buffering on wide rows
# ---------------------------------------------------------------------------


def test_wide_rows_mergebuf_bounded(tmp_path):
    # Two inputs with 40 pixels each under the same bin1 and mergebuf=3:
    # the old code loaded all 80 at once and only warned
    bins = pd.DataFrame(
        {
            "chrom": ["a"] * 41,
            "start": np.arange(41) * 10,
            "end": (np.arange(41) + 1) * 10,
        }
    )
    px = pd.DataFrame(
        {
            "bin1_id": np.zeros(40, dtype=np.int64),
            "bin2_id": np.arange(1, 41),
            "count": np.ones(40, dtype=np.int64),
        }
    )
    uris = [make_cooler(tmp_path / f"r{i}.cool", px, bins=bins) for i in range(2)]
    merger = CoolerMerger([cooler.Cooler(u) for u in uris], mergebuf=3)
    batches = [len(chunk["bin1_id"]) for chunk in merger]
    assert max(merger.read_sizes) <= 3
    assert merger.max_buffered <= 3
    assert max(batches) <= 3
    assert sum(batches) == 40  # the 80 records combine into 40 pixels


def test_wide_rows_stress_bounded_and_exact(tmp_path):
    # >100k distinct pixels over two inputs, wide rows mixed with empty
    # rows and duplicate keys; mergebuf=257 must bound every source read,
    # the peak buffered pixels, and every output batch
    n_bins = 5000
    bins = pd.DataFrame(
        {
            "chrom": ["a"] * n_bins,
            "start": np.arange(n_bins) * 10,
            "end": (np.arange(n_bins) + 1) * 10,
        }
    )

    def make_input(seed, n_rows):
        rng = np.random.default_rng(seed)
        b1, b2, ct = [], [], []
        for i in range(n_rows):
            if rng.random() < 0.1:  # empty row
                continue
            w = rng.integers(20, 60)
            js = np.sort(rng.choice(n_bins, size=w, replace=False))
            js = js[js >= i]
            for j in js:
                b1.append(i)
                b2.append(j)
                ct.append(int(rng.integers(1, 100)))
        return pd.DataFrame({"bin1_id": b1, "bin2_id": b2, "count": ct})

    df_a = make_input(1, 2200)
    df_b = make_input(2, 2100)
    # duplicate keys across the inputs, not on a uniform grid
    dup = df_a.sample(frac=0.3, random_state=7)
    df_b = pd.concat([df_b, dup]).sort_values(["bin1_id", "bin2_id"])
    distinct = len(pd.concat([df_a, df_b])[["bin1_id", "bin2_id"]].drop_duplicates())
    assert distinct >= 100_000

    ua = make_cooler(tmp_path / "a.cool", df_a, bins=bins, ordered=False)
    ub = make_cooler(tmp_path / "b.cool", df_b, bins=bins, ordered=False)
    ref = reference_sum([df_a, df_b])

    mergebuf = 257
    merger = CoolerMerger([cooler.Cooler(ua), cooler.Cooler(ub)], mergebuf=mergebuf)
    tracemalloc.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out_sizes, got = [], {}
    for chunk in merger:
        out_sizes.append(len(chunk["bin1_id"]))
        for a, b, c in zip(chunk["bin1_id"], chunk["bin2_id"], chunk["count"]):
            got[(int(a), int(b))] = int(c)
    peak, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    stored_nnz = cooler.Cooler(ua).info["nnz"] + cooler.Cooler(ub).info["nnz"]
    print(
        f"\nwide-row stress: {distinct} distinct keys, {stored_nnz} stored "
        f"source pixels, mergebuf={mergebuf}\n"
        f"  reads: {len(merger.read_sizes)} slices, max {max(merger.read_sizes)}\n"
        f"  peak buffered source pixels: {merger.max_buffered}\n"
        f"  output batches: {len(out_sizes)}, max {max(out_sizes)}\n"
        f"  extra memory: tracemalloc peak {peak / 1e6:.1f} MB, "
        f"maxRSS delta {(rss_after - rss_before) / 1024:.1f} MB"
    )
    assert max(merger.read_sizes) <= mergebuf
    assert merger.max_buffered <= mergebuf
    assert max(out_sizes) <= mergebuf
    assert sum(merger.read_sizes) == stored_nnz
    # Bounded buffering: the merge never materializes the matrix
    assert peak < 64 * 2**20
    # Exact counts against the arbitrary-precision reference
    assert got == ref


# ---------------------------------------------------------------------------
# Transactional failures
# ---------------------------------------------------------------------------


def test_overflow_preserves_existing_target_and_siblings(tmp_path):
    path = str(tmp_path / "tx.h5")
    make_cooler(path + "::/old", make_pixels({(0, 1): 111}), mode="w")
    with h5py.File(path, "r+") as f:
        f.create_group("other").attrs["tag"] = "keep"

    px = make_pixels({(0, 1): BIG_INT64})
    uris = [
        make_cooler(tmp_path / f"m{i}.cool", px, dtypes={"count": "int64"})
        for i in range(2)
    ]
    with pytest.raises(OverflowError):
        merge_coolers(
            path + "::/old", uris, mergebuf=100, mode="a", dtypes={"count": "int64"}
        )
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"old", "other"}
        assert f["other"].attrs["tag"] == "keep"
        assert f["old/pixels/count"][:] == [111]
        assert f["old"].attrs["nnz"] == 1
    assert cooler.Cooler(path + "::/old").info["sum"] == 111

    # A failed brand-new group leaves no trace either
    with pytest.raises(OverflowError):
        merge_coolers(
            path + "::/new", uris, mergebuf=100, mode="a", dtypes={"count": "int64"}
        )
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"old", "other"}

    # In mode "w" the pre-existing file is preserved on failure
    with pytest.raises(OverflowError):
        merge_coolers(path, uris, mergebuf=100, mode="w", dtypes={"count": "int64"})
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"old", "other"}
        assert f["old/pixels/count"][:] == [111]


def test_axis_mismatch_validated_before_writing(tmp_path):
    path = str(tmp_path / "tx.h5")
    make_cooler(path + "::/old", make_pixels({(0, 1): 5}), mode="w")
    ua = make_cooler(tmp_path / "a.cool", make_pixels({(0, 1): 1}))
    ub = make_cooler(
        tmp_path / "b.cool",
        make_pixels({(0, 1): 1}),
        bins=pd.DataFrame({"chrom": ["a", "a"], "start": [0, 5], "end": [5, 20]}),
    )
    with pytest.raises(ValueError, match="bin table"):
        merge_coolers(path + "::/new", [ua, ub], mergebuf=100, mode="a")
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"old"}  # nothing was created


def test_failure_during_finish_not_exposed(tmp_path, monkeypatch):
    # Pixels written but indexes/metadata not finished: the destination
    # must not gain a half-written group, and an existing one survives
    path = str(tmp_path / "tx.h5")
    make_cooler(path + "::/old", make_pixels({(0, 1): 5}), mode="w")
    ua = make_cooler(tmp_path / "a.cool", make_pixels({(0, 1): 1}))
    ub = make_cooler(tmp_path / "b.cool", make_pixels({(0, 1): 2}))

    from cooler.create import _writer as writer_mod

    original = writer_mod.index_pixels

    def boom(*args, **kwargs):
        raise RuntimeError("simulated index failure")

    monkeypatch.setattr(writer_mod, "index_pixels", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        merge_coolers(path + "::/new", [ua, ub], mergebuf=100, mode="a")
    with h5py.File(path, "r") as f:
        assert set(f.keys()) == {"old"}  # no partial group exposed
        assert f["old/pixels/count"][:] == [5]
    monkeypatch.setattr(writer_mod, "index_pixels", original)

    # And the same merge succeeds once the failure is gone
    merge_coolers(path + "::/new", [ua, ub], mergebuf=100, mode="a")
    assert cooler.Cooler(path + "::/new").info["sum"] == 3


def test_unordered_failure_cleans_temp_files(tmp_path):
    px = make_pixels({(0, 1): BIG_INT64})
    chunks = [px.copy(), px.copy()]
    with pytest.raises(OverflowError):
        create_cooler(
            str(tmp_path / "o.cool"),
            BINS2,
            chunks,
            ordered=False,
            temp_dir=str(tmp_path),
            dtypes={"count": "int64"},
        )
    # no temporary run files or writing groups are left behind
    leftovers = [p for p in os.listdir(tmp_path) if "multi.cool" in p or "writing" in p]
    assert leftovers == []
    assert not os.path.exists(tmp_path / "o.cool")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_merge_overflow_and_uint64(tmp_path):
    px = make_pixels({(0, 1): BIG_INT64})
    uris = [
        make_cooler(tmp_path / f"m{i}.cool", px, dtypes={"count": "int64"})
        for i in range(2)
    ]
    runner = CliRunner()

    # default inferred int64: the overflow fails the command
    out = str(tmp_path / "bad.cool")
    result = runner.invoke(merge_cli, [out, *uris])
    assert result.exit_code != 0
    assert isinstance(result.exception, OverflowError)
    assert not os.path.exists(out)

    # explicit uint64: the large sum succeeds
    out = str(tmp_path / "ok.cool")
    result = runner.invoke(merge_cli, ["--field", "count:dtype=uint64", out, *uris])
    assert result.exit_code == 0, result.output
    clr = cooler.Cooler(out)
    assert clr.info["sum"] == 9223372036854775810
    assert read_pixels(out) == {(0, 1): 9223372036854775810}
    check_indexes(out)


def test_cli_merge_and_info_roundtrip(tmp_path):
    # Values and indexes read back through both Python and the CLI
    bins = pd.DataFrame(
        {"chrom": ["a"] * 4, "start": np.arange(4) * 10, "end": (np.arange(4) + 1) * 10}
    )
    rng = np.random.default_rng(11)
    inputs = []
    for _ in range(2):
        b1 = rng.integers(0, 3, 30)
        b2 = rng.integers(0, 4, 30)
        df = pd.DataFrame(
            {"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2)}
        )
        df = df.drop_duplicates().sort_values(["bin1_id", "bin2_id"])
        df["count"] = rng.integers(1, 500, len(df))
        inputs.append(df)
    uris = [
        make_cooler(tmp_path / f"i{k}.cool", df, bins=bins)
        for k, df in enumerate(inputs)
    ]
    ref = reference_sum(inputs)

    out = str(tmp_path / "merged.cool")
    result = CliRunner().invoke(merge_cli, [out, *uris])
    assert result.exit_code == 0, result.output
    assert read_pixels(out) == ref
    assert cooler.Cooler(out).info["sum"] == sum(ref.values())
    check_indexes(out)

    # cooler dump readback matches too
    dump = subprocess.run(
        [sys.executable, "-m", "cooler", "dump", "-t", "pixels", out],
        capture_output=True,
        text=True,
        check=False,
    )
    assert dump.returncode == 0, dump.stderr
    lines = [
        ln
        for ln in dump.stdout.strip().split("\n")
        if ln and not ln.startswith("bin1_id")
    ]
    got = {}
    for ln in lines:
        b1, b2, c = ln.split("\t")[:3]
        got[(int(b1), int(b2))] = int(c)
    assert got == ref
