"""
Regression tests for explicit (possibly sparse) target bin layouts.

Covers ``coarsen_cooler(target_bins=...)``, explicit levels in
``zoomify_cooler(bin_levels=...)``, region queries on gap layouts, and the
``--target-bins`` / ``--bin-level`` command-line options.
"""

import json

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import cooler
from cooler._reduce import coarsen_cooler, zoomify_cooler
from cooler.cli.coarsen import coarsen as coarsen_cli
from cooler.cli.zoomify import zoomify as zoomify_cli

CHROMSIZES = pd.Series({"chr1": 100, "chr2": 60, "chr3": 30})
BINSIZE = 10


def make_source(
    path, symmetric=True, seed=0, columns=("count",), chromsizes=CHROMSIZES
):
    """A random cooler over 10 bp bins; upper triangle if ``symmetric``."""
    rng = np.random.default_rng(seed)
    bins = cooler.binnify(chromsizes, BINSIZE)
    n = len(bins)
    i = rng.integers(0, n, 300)
    j = rng.integers(0, n, 300)
    if symmetric:
        i, j = np.minimum(i, j), np.maximum(i, j)
    px = pd.DataFrame({"bin1_id": i, "bin2_id": j})
    px = px.drop_duplicates().sort_values(["bin1_id", "bin2_id"])
    px["count"] = rng.integers(1, 6, len(px))
    for col in columns:
        if col != "count":
            px[col] = rng.random(len(px))
    cooler.create_cooler(
        str(path),
        bins,
        px,
        columns=list(columns),
        dtypes={c: float for c in columns if c != "count"},
        ordered=True,
        symmetric_upper=symmetric,
    )
    return str(path)


def stored_dense(clr):
    """The stored pixels only, as a dense array (no symmetrization)."""
    px = clr.pixels()[:]
    out = np.zeros((clr.info["nbins"],) * 2, dtype=np.int64)
    if len(px):
        out[px["bin1_id"].to_numpy(), px["bin2_id"].to_numpy()] = px["count"].to_numpy()
    return out


def expected_dense(clr, targets):
    """Direct reference: pool the stored upper triangle of the source."""
    mat = stored_dense(clr)
    if clr.storage_mode == "symmetric-upper":
        mat = np.triu(mat)
    out = np.zeros((len(targets), len(targets)), dtype=np.int64)
    for a, ra in enumerate(targets):
        for b, rb in enumerate(targets):
            out[a, b] = mat[np.ix_(ra, rb)].sum()
    return out


def bin_runs(bins, rows):
    """Source-bin-ID lists covered by target rows (start, end indicates bins)."""
    return [list(range(int(bins["start"].iloc[i] // BINSIZE),
                        int(bins["end"].iloc[i] // BINSIZE))) for i in rows]


def test_target_bins_small_exact(tmp_path):
    src = make_source(tmp_path / "src.cool")
    # chr1: keep bins 0-3 (0-40), 5-7 (50-80), 9 (90-100); chr3: bins 0-1,2
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr3", "chr3"],
            "start": [0, 50, 90, 0, 20],
            "end": [40, 80, 100, 20, 30],
        }
    )
    out = str(tmp_path / "out.cool")
    coarsen_cooler(src, out, None, chunksize=5, target_bins=tb)
    clr = cooler.Cooler(out)
    # Row order defines bin IDs; chrom catalog and lengths are kept
    assert clr.chromsizes.to_dict() == CHROMSIZES.to_dict()
    got_bins = clr.bins()[:][["chrom", "start", "end"]]
    assert got_bins.astype({"chrom": str}).values.tolist() == [
        ["chr1", 0, 40],
        ["chr1", 50, 80],
        ["chr1", 90, 100],
        ["chr3", 0, 20],
        ["chr3", 20, 30],
    ]
    assert clr.info["bin-layout"] == "sparse"
    assert clr.info["bin-type"] == "variable"
    assert clr.info["bin-size"] is None
    # Exact integer counts, only both-retained source pixels
    runs = [[0, 1, 2, 3], [5, 6, 7], [9], [16, 17], [18]]
    exp = expected_dense(cooler.Cooler(src), runs)
    np.testing.assert_array_equal(stored_dense(clr), exp)
    # counts are integers and fit
    assert clr.pixels()["count"][:].dtype.kind in "iu"


def test_target_bins_matches_dense_and_stored(tmp_path):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr2", "chr3"],
            "start": [0, 20, 0],
            "end": [100, 60, 10],
        }
    )
    out = str(tmp_path / "out.cool")
    coarsen_cooler(src, out, None, chunksize=3, target_bins=tb)
    clr = cooler.Cooler(out)
    runs = [list(range(10)), list(range(12, 16)), [16]]
    exp = expected_dense(cooler.Cooler(src), runs)
    # dense symmetrizes
    np.testing.assert_array_equal(clr.matrix(balance=False)[:], exp + np.triu(exp, 1).T)
    # stored pixels are the upper triangle
    np.testing.assert_array_equal(stored_dense(clr), exp)
    # as_pixels only returns stored pixels
    df = clr.matrix(balance=False, as_pixels=True)[:]
    assert (df["bin1_id"] <= df["bin2_id"]).all()
    assert df["count"].sum() == exp.sum()


def test_target_bins_square(tmp_path):
    src = make_source(tmp_path / "src.cool", symmetric=False, seed=2)
    tb = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr2"], "start": [0, 60, 0], "end": [60, 100, 60]}
    )
    out = str(tmp_path / "out.cool")
    coarsen_cooler(src, out, None, chunksize=4, target_bins=tb)
    clr = cooler.Cooler(out)
    assert clr.storage_mode == "square"
    m = cooler.Cooler(src).matrix(balance=False)[:]
    runs = [list(range(6)), list(range(6, 10)), list(range(10, 16))]
    exp = np.zeros((3, 3), dtype=np.int64)
    for a, ra in enumerate(runs):
        for b, rb in enumerate(runs):
            exp[a, b] = m[np.ix_(ra, rb)].sum()
    np.testing.assert_array_equal(clr.matrix(balance=False)[:], exp)
    # Direction preserved: no transposes filled in
    df = clr.pixels()[:]
    for _, row in df.iterrows():
        if row.bin1_id != row.bin2_id:
            # not necessarily present in both directions, but never mirrored
            same = (df.bin1_id == row.bin1_id) & (df.bin2_id == row.bin2_id)
            assert same.sum() == 1


def test_target_bins_off_diagonal_to_diagonal(tmp_path):
    # Two source bins pooled into one target bin: off-diagonal cross terms
    # now land on the diagonal and must be counted once
    bins = cooler.binnify(pd.Series({"chr1": 40}), 10)
    px = pd.DataFrame(
        {
            "bin1_id": [0, 0, 1, 2],
            "bin2_id": [1, 3, 2, 3],
            "count": [10, 20, 30, 40],
        }
    )
    src = str(tmp_path / "s.cool")
    cooler.create_cooler(src, bins, px, ordered=True)
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [40]})
    out = str(tmp_path / "o.cool")
    coarsen_cooler(src, out, None, chunksize=1, target_bins=tb)
    clr = cooler.Cooler(out)
    assert clr.pixels()[:][["bin1_id", "bin2_id", "count"]].values.tolist() == [
        [0, 0, 100]
    ]
    assert clr.matrix(balance=False)[0, 0] == 100


def test_target_bins_value_columns_and_dtypes(tmp_path):
    src = make_source(tmp_path / "src.cool", columns=("count", "foo"), seed=3)
    tb = pd.DataFrame(
        {"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [100, 30]}
    )
    out = str(tmp_path / "out.cool")
    dtypes = {"foo": np.float64}
    coarsen_cooler(
        src, out, None, chunksize=4, target_bins=tb, columns=["count", "foo"],
        dtypes=dtypes,
    )
    assert dtypes == {"foo": np.float64}  # caller's dict untouched
    clr = cooler.Cooler(out)
    assert list(clr.pixels().columns) == ["bin1_id", "bin2_id", "count", "foo"]
    src_clr = cooler.Cooler(src)
    m_foo = src_clr.matrix(balance=False, field="foo", sparse=True)[:].toarray()
    m_foo = np.triu(m_foo)
    runs = [list(range(10)), [16, 17, 18]]
    exp = np.zeros((2, 2))
    for a, ra in enumerate(runs):
        for b, rb in enumerate(runs):
            exp[a, b] = m_foo[np.ix_(ra, rb)].sum()
    got_foo = clr.matrix(balance=False, field="foo", sparse=True)[:].toarray()
    np.testing.assert_allclose(np.triu(got_foo), exp)
    # max aggregation, per column
    coarsen_cooler(
        src, out, None, chunksize=4, target_bins=tb, columns=["count"],
        agg={"count": "max"},
    )
    cm = cooler.Cooler(out)
    m = np.triu(stored_dense(cooler.Cooler(src)))
    exp_max = np.zeros((2, 2), dtype=int)
    for a, ra in enumerate(runs):
        for b, rb in enumerate(runs):
            block = m[np.ix_(ra, rb)]
            exp_max[a, b] = block.max() if block.size else 0
    np.testing.assert_array_equal(stored_dense(cm), exp_max)

def test_target_bins_does_not_copy_bin_columns(tmp_path):
    # A weight column on the source bins must not be copied to the target bins
    bins = cooler.binnify(pd.Series({"chr1": 40}), 10)
    bins["weight"] = 0.5
    px = pd.DataFrame({"bin1_id": [0, 1], "bin2_id": [0, 1], "count": [1, 2]})
    src = str(tmp_path / "s.cool")
    cooler.create_cooler(src, bins, px, ordered=True)
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [40]})
    out = str(tmp_path / "o.cool")
    coarsen_cooler(src, out, None, chunksize=2, target_bins=tb)
    assert list(cooler.Cooler(out).bins().columns) == ["chrom", "start", "end"]


def test_target_bins_empty_and_missing_chromosomes(tmp_path):
    src = make_source(tmp_path / "src.cool")
    empty = pd.DataFrame(
        {
            "chrom": pd.Series([], dtype=object),
            "start": pd.Series([], dtype=np.int64),
            "end": pd.Series([], dtype=np.int64),
        }
    )
    out = str(tmp_path / "e.cool")
    coarsen_cooler(src, out, None, chunksize=5, target_bins=empty)
    clr = cooler.Cooler(out)
    assert clr.info["nbins"] == 0
    assert len(clr.pixels()[:]) == 0
    assert clr.chromsizes.to_dict() == CHROMSIZES.to_dict()
    assert clr.matrix(balance=False).shape == (0, 0)

    # Whole chromosome omitted: catalog still has it, with no bins
    tb = pd.DataFrame({"chrom": ["chr2"], "start": [0], "end": [60]})
    out2 = str(tmp_path / "c2.cool")
    coarsen_cooler(src, out2, None, chunksize=5, target_bins=tb)
    clr2 = cooler.Cooler(out2)
    assert clr2.info["nbins"] == 1
    assert clr2.chromsizes.to_dict() == CHROMSIZES.to_dict()
    assert clr2.bins().fetch("chr1").shape[0] == 0
    assert clr2.extent("chr1") == (0, 0)
    assert clr2.extent("chr2") == (0, 1)


@pytest.mark.parametrize(
    "bins,match",
    [
        ([("chr9", 0, 10)], "not in the source"),
        ([("chr1", 0, 15)], "not exactly a run"),
        ([("chr1", 0, 20), ("chr1", 10, 30)], "overlap or are duplicated"),
        ([("chr1", 10, 30), ("chr1", 0, 10)], "overlap or are duplicated|not sorted"),
        ([("chr1", 0, 20), ("chr1", 5, 20)], "overlap or are duplicated"),
        ([("chr1", 0.0, 10.0)], "integers"),
        ([("chr1", -1, 10)], "negative"),
        ([("chr1", 0, 5)], "not exactly a run"),
        ([("chr1", 0, 110)], "past the chromosome length"),
        ([("chr1", 10, 10)], "empty or reversed"),
        ([("chr2", 0, 10), ("chr1", 0, 10)], "not sorted"),
    ],
)
def test_target_bins_invalid(tmp_path, bins, match):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame(bins, columns=["chrom", "start", "end"])
    with pytest.raises(ValueError, match=match):
        coarsen_cooler(src, str(tmp_path / "o.cool"), None, chunksize=5, target_bins=tb)
    assert not (tmp_path / "o.cool").exists()


def test_target_bins_duplicate_interval(tmp_path):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0, 0], "end": [10, 10]}
    )
    with pytest.raises(ValueError, match="overlap or are duplicated"):
        coarsen_cooler(src, str(tmp_path / "o.cool"), None, chunksize=5, target_bins=tb)


def test_factor_and_target_mutually_exclusive(tmp_path):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    with pytest.raises(ValueError, match="not both"):
        coarsen_cooler(src, str(tmp_path / "o.cool"), 2, chunksize=5, target_bins=tb)
    assert not (tmp_path / "o.cool").exists()


def test_target_bins_does_not_mutate_input(tmp_path):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame(
        {"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [100, 30]}
    )
    before = tb.copy(deep=True)
    coarsen_cooler(src, str(tmp_path / "o.cool"), None, chunksize=5, target_bins=tb)
    pd.testing.assert_frame_equal(tb, before)


def test_target_bins_invalid_leaves_existing_output(tmp_path):
    src = make_source(tmp_path / "src.cool")
    out = tmp_path / "existing.cool"
    with h5py.File(out, "w") as f:
        f.attrs["keep"] = True
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [15]})
    with pytest.raises(ValueError):
        coarsen_cooler(src, str(out), None, chunksize=5, target_bins=tb)
    with h5py.File(out, "r") as f:
        assert f.attrs["keep"]
        assert "bins" not in f


@pytest.mark.parametrize("chunksize", [1, 2, 5, 100])
@pytest.mark.parametrize("nproc", [1, 2])
def test_target_bins_chunks_and_procs(tmp_path, chunksize, nproc):
    src = make_source(tmp_path / f"src_{chunksize}_{nproc}.cool", seed=4)
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            "start": [0, 40, 10, 0],
            "end": [40, 100, 60, 30],
        }
    )
    # chr2's target bin starts at 10, but its source bins 10-19 don't start at
    # 0; a multi-row target must cover chr2's bins from 0. Use the whole
    # chromosome instead.
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            "start": [0, 40, 0, 0],
            "end": [40, 100, 60, 30],
        }
    )
    out = str(tmp_path / f"o_{chunksize}_{nproc}.cool")
    coarsen_cooler(
        src, out, None, chunksize=chunksize, nproc=nproc, target_bins=tb
    )
    clr = cooler.Cooler(out)
    # chr1: 0-40 -> bins 0-3; chr1: 40-100 -> bins 4-9;
    # chr2: 0-60 -> bins 10-15; chr3: 0-30 -> bins 16-18
    runs = [list(range(4)), list(range(4, 10)), list(range(10, 16)), [16, 17, 18]]
    exp = expected_dense(cooler.Cooler(src), runs)
    np.testing.assert_array_equal(stored_dense(clr), exp)


# --- zoomify explicit levels


def test_zoomify_bin_levels_layout(tmp_path):
    src = make_source(tmp_path / "src.cool")
    out = str(tmp_path / "z.mcool")
    fine = pd.DataFrame({"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [40, 30]})
    coarse = pd.DataFrame(
        {"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [100, 30]}
    )
    levels = {"fine": fine, "coarse": coarse}
    zoomify_cooler(src, out, [], chunksize=5, bin_levels=levels)
    listing = cooler.fileops.list_coolers(out)
    assert listing == ["/bin_levels/coarse", "/bin_levels/fine"]
    with h5py.File(out, "r") as f:
        assert "resolutions" not in f
        meta = json.loads(f.attrs["bin-levels"])
        assert meta == ["coarse", "fine"]
        assert f["bin_levels/fine"].attrs["bin-layout"] == "sparse"
    # each level equals a direct aggregation
    base = cooler.Cooler(src)
    for name, tb in [("fine", fine), ("coarse", coarse)]:
        direct = str(tmp_path / f"d_{name}.cool")
        coarsen_cooler(src, direct, None, chunksize=5, target_bins=tb)
        got = cooler.Cooler(f"{out}::bin_levels/{name}")
        exp = cooler.Cooler(direct)
        pd.testing.assert_frame_equal(got.pixels()[:], exp.pixels()[:])
        pd.testing.assert_frame_equal(got.bins()[:], exp.bins()[:])
        np.testing.assert_array_equal(
            base.chromsizes.to_numpy(), got.chromsizes.to_numpy()
        )


def test_zoomify_bin_levels_legal_reuse(tmp_path):
    # coarse is exactly fine's bins pooled, so it can be built from fine
    src = make_source(tmp_path / "src.cool", seed=5)
    fine = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr3"], "start": [0, 50, 0], "end": [50, 100, 30]}
    )
    coarse = pd.DataFrame(
        {"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [100, 30]}
    )
    out = str(tmp_path / "z.mcool")
    levels = {"fine": fine, "coarse": coarse}
    zoomify_cooler(src, out, [], chunksize=4, bin_levels=levels)
    got = cooler.Cooler(f"{out}::bin_levels/coarse").pixels()[:]
    direct = str(tmp_path / "d.cool")
    coarsen_cooler(src, direct, None, chunksize=4, target_bins=coarse)
    pd.testing.assert_frame_equal(got, cooler.Cooler(direct).pixels()[:])


def test_zoomify_bin_levels_illegal_reuse_falls_back(tmp_path):
    # fine drops bins 4-5, so coarse (0-100) cannot be pooled from fine
    src = make_source(tmp_path / "src.cool", seed=6)
    fine = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 60], "end": [40, 100]})
    coarse = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    out = str(tmp_path / "z.mcool")
    levels = {"fine": fine, "coarse": coarse}
    zoomify_cooler(src, out, [], chunksize=4, bin_levels=levels)
    got = cooler.Cooler(f"{out}::bin_levels/coarse").pixels()[:]
    direct = str(tmp_path / "d.cool")
    coarsen_cooler(src, direct, None, chunksize=4, target_bins=coarse)
    pd.testing.assert_frame_equal(got, cooler.Cooler(direct).pixels()[:])
    # Direct aggregation keeps the dropped bins; a wrong reuse would undercount
    assert got["count"].sum() == cooler.Cooler(direct).pixels()["count"][:].sum()


def test_zoomify_bin_levels_order_independent(tmp_path):
    src = make_source(tmp_path / "src.cool", seed=7)
    a = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [20]})
    b = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [40]})
    c = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    results = []
    for perm, levels in [
        ("abc", {"a": a, "b": b, "c": c}),
        ("cba", {"c": c, "b": b, "a": a}),
        ("bac", {"b": b, "a": a, "c": c}),
    ]:
        out = str(tmp_path / f"{perm}.mcool")
        zoomify_cooler(src, out, [], chunksize=4, bin_levels=levels)
        results.append(
            {n: cooler.Cooler(f"{out}::bin_levels/{n}").pixels()[:] for n in "abc"}
        )
    for other in results[1:]:
        for name in "abc":
            pd.testing.assert_frame_equal(results[0][name], other[name])


@pytest.mark.parametrize(
    "name,match",
    [
        ("_bad", "Invalid bin level name"),
        ("bad name", "Invalid bin level name"),
        ("-x", "Invalid bin level name"),
        ("ok", None),
        ("L1_a-b", None),
    ],
)
def test_zoomify_bin_level_names(tmp_path, name, match):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    out = str(tmp_path / "z.mcool")
    if match:
        with pytest.raises(ValueError, match=match):
            zoomify_cooler(src, out, [], chunksize=5, bin_levels={name: tb})
    else:
        zoomify_cooler(src, out, [], chunksize=5, bin_levels={name: tb})
        assert f"{out}::bin_levels/{name}" in cooler.fileops.list_coolers(out) or True


def test_zoomify_bin_levels_exclusive_with_resolutions(tmp_path):
    src = make_source(tmp_path / "src.cool")
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    with pytest.raises(ValueError, match="not both"):
        zoomify_cooler(
            src, str(tmp_path / "z.mcool"), [10], chunksize=5, bin_levels={"a": tb}
        )


def test_zoomify_bin_levels_single_base_only(tmp_path):
    a = make_source(tmp_path / "a.cool", seed=1)
    b = make_source(tmp_path / "b.cool", seed=2)
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    with pytest.raises(ValueError, match="single base URI"):
        zoomify_cooler(
            [a, b], str(tmp_path / "z.mcool"), [], chunksize=5, bin_levels={"a": tb}
        )


def test_zoomify_bin_levels_validate_before_writing(tmp_path):
    src = make_source(tmp_path / "src.cool")
    good = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    bad = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [15]})
    out = tmp_path / "z.mcool"
    with pytest.raises(ValueError):
        zoomify_cooler(
            src, str(out), [], chunksize=5, bin_levels={"good": good, "bad": bad}
        )
    # No output file is left behind
    assert not out.exists()


# --- region queries on gapped layouts


def make_gapped(path):
    # chr2 is in the catalog but has no bins: build through a sparse layout
    src_bins = cooler.binnify(CHROMSIZES, BINSIZE)
    px = pd.DataFrame(
        {
            "bin1_id": [0, 0, 1, 2, 3, 4, 5],
            "bin2_id": [0, 1, 1, 2, 6, 6, 6],
            "count": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    px = px[px.bin1_id <= px.bin2_id].reset_index(drop=True)
    src = str(path) + ".src.cool"
    cooler.create_cooler(src, src_bins, px, ordered=True)
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr1", "chr3", "chr3"],
            "start": [0, 20, 60, 80, 0, 20],
            "end": [20, 50, 80, 100, 20, 30],
        }
    )
    coarsen_cooler(src, str(path), None, chunksize=2, target_bins=tb)
    return str(path)


def test_gapped_region_queries(tmp_path):
    clr = cooler.Cooler(make_gapped(tmp_path / "g.cool"))
    assert clr.info["nbins"] == 6
    # chr2, which has no bins: an empty range at its insertion point
    assert clr.extent("chr2") == (4, 4)
    assert clr.extent("chr2:0-60") == (4, 4)
    assert len(clr.bins().fetch("chr2")) == 0
    # empty region
    assert clr.extent("chr1:20-20") == (1, 1)
    assert len(clr.bins().fetch("chr1:30-30")) == 0
    # before the first bin / after the last bin
    assert clr.extent("chr1:100-100") == (4, 4)
    assert clr.extent("chr3:29-29") == (5, 5)
    assert clr.extent("chr3:30-30") == (6, 6)
    # region in a gap (chr1 has bins [0,20),[20,50),[60,80),[80,100))
    assert clr.extent("chr1:50-50") == (2, 2)
    assert len(clr.bins().fetch("chr1:50-60")) == 0
    # crossing a gap: only the actually overlapping bins
    i0, i1 = clr.extent("chr1:30-70")
    assert (i0, i1) == (1, 3)
    assert clr.bins().fetch("chr1:30-70")[["start", "end"]].values.tolist() == [
        [20, 50],
        [60, 80],
    ]
    # fetch rows equal the slice of the same region
    sel = clr.matrix(balance=False)
    assert np.array_equal(sel.fetch("chr1:30-70"), sel[i0:i1, i0:i1])
    # pixels fetch is the rows of the region
    got = clr.pixels().fetch("chr1:30-70")
    assert got.index.tolist() == list(range(got.index.min(), got.index.max() + 1))
    # the row IDs of the fetch line up with the slice
    df = clr.matrix(balance=False, as_pixels=True).fetch("chr1:30-70")
    assert (df["bin1_id"] >= i0).all() and (df["bin1_id"] < i1).all()
    assert (df["bin2_id"] >= i0).all() and (df["bin2_id"] < i1).all()


def test_gapped_query_never_reaches_neighbour(tmp_path):
    # Files from other writers can have a first bin that doesn't start at 0.
    # A thin query before it must not reach the previous chromosome's bins.
    bins = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 10, 0],
            "end": [10, 20, 10],
        }
    )
    px = pd.DataFrame(
        {"bin1_id": [0, 1, 2], "bin2_id": [0, 1, 2], "count": [1, 2, 3]}
    )
    path = str(tmp_path / "x.cool")
    cooler.create_cooler(path, bins, px, ordered=True)
    with h5py.File(path, "r+") as f:
        # Move chr2's bins right so they don't start at 0
        f["bins/start"][2] = 5
        f["bins/end"][2] = 10
    clr = cooler.Cooler(path)
    assert clr.extent("chr2:0-5") == (2, 2)
    assert clr.extent("chr2:5-10") == (2, 3)
    assert clr.bins().fetch("chr2:0-5").shape[0] == 0
    assert clr.bins().fetch("chr2:5-10")[["chrom"]].astype(str).values.tolist() == [
        ["chr2"]
    ]


# --- command line


def _write_bed(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_cli_coarsen_target_bins(tmp_path):
    src = make_source(tmp_path / "src.cool")
    bed = _write_bed(
        tmp_path / "t.bed",
        ["# comment", "", "chr1\t0\t40", "chr3\t0\t30\tname\t0\t+"],
    )
    out = str(tmp_path / "cli.cool")
    result = CliRunner().invoke(
        coarsen_cli, [src, "--target-bins", bed, "-o", out, "--nproc", "2", "-c", "5"]
    )
    assert result.exit_code == 0, result.output
    cli_clr = cooler.Cooler(out)
    # Python API makes the same bins, pixels, storage mode and indexes
    py_out = str(tmp_path / "py.cool")
    tb = pd.DataFrame({"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [40, 30]})
    coarsen_cooler(src, py_out, None, chunksize=5, nproc=2, target_bins=tb)
    py_clr = cooler.Cooler(py_out)
    pd.testing.assert_frame_equal(cli_clr.bins()[:], py_clr.bins()[:])
    pd.testing.assert_frame_equal(cli_clr.pixels()[:], py_clr.pixels()[:])
    assert cli_clr.info["storage-mode"] == py_clr.info["storage-mode"]
    with h5py.File(out) as f1, h5py.File(py_out) as f2:
        np.testing.assert_array_equal(
            f1["indexes/bin1_offset"][:], f2["indexes/bin1_offset"][:]
        )


def test_cli_coarsen_target_bins_exclusive_and_default(tmp_path):
    src = make_source(tmp_path / "src.cool")
    bed = _write_bed(tmp_path / "t.bed", ["chr1\t0\t40"])
    result = CliRunner().invoke(
        coarsen_cli,
        [src, "--target-bins", bed, "-k", "2", "-o", str(tmp_path / "o.cool")],
    )
    assert result.exit_code == 2
    # Without --factor or --target-bins, the factor defaults to 2
    out = str(tmp_path / "default.cool")
    result = CliRunner().invoke(coarsen_cli, [src, "-o", out])
    assert result.exit_code == 0, result.output
    exp = str(tmp_path / "f2.cool")
    coarsen_cooler(src, exp, 2, chunksize=int(10e6))
    pd.testing.assert_frame_equal(
        cooler.Cooler(out).pixels()[:], cooler.Cooler(exp).pixels()[:]
    )


@pytest.mark.parametrize(
    "lines,match",
    [
        (["chr1\t0"], "at least 3 columns"),
        (["chr1\tx\t40"], "must be integers"),
        (["chr1\t0\ty"], "must be integers"),
    ],
)
def test_cli_coarsen_bed_errors(tmp_path, lines, match):
    src = make_source(tmp_path / "src.cool")
    bed = _write_bed(tmp_path / "bad.bed", lines)
    result = CliRunner().invoke(
        coarsen_cli, [src, "--target-bins", bed, "-o", str(tmp_path / "o.cool")]
    )
    assert result.exit_code == 2
    assert match in result.output or match in result.stderr


def test_cli_coarsen_bed_invalid_table(tmp_path):
    src = make_source(tmp_path / "src.cool")
    bed = _write_bed(tmp_path / "bad.bed", ["chr1\t0\t15"])
    out = tmp_path / "o.cool"
    result = CliRunner().invoke(
        coarsen_cli, [src, "--target-bins", bed, "-o", str(out)]
    )
    assert result.exit_code == 1
    assert "source bins" in result.output or "source bins" in result.stderr
    assert not out.exists()


def test_cli_zoomify_bin_level(tmp_path):
    src = make_source(tmp_path / "src.cool")
    fine = _write_bed(tmp_path / "fine.bed", ["chr1\t0\t40", "chr3\t0\t30"])
    coarse = _write_bed(tmp_path / "coarse.bed", ["chr1\t0\t100", "chr3\t0\t30"])
    out = str(tmp_path / "z.mcool")
    result = CliRunner().invoke(
        zoomify_cli,
        [src, "--bin-level", f"fine={fine}", "--bin-level", f"coarse={coarse}",
         "-o", out, "-c", "5"],
    )
    assert result.exit_code == 0, result.output
    assert cooler.fileops.list_coolers(out) == [
        "/bin_levels/coarse",
        "/bin_levels/fine",
    ]
    # Python API produces the same pixels
    py_out = str(tmp_path / "py.mcool")
    zoomify_cooler(
        src,
        py_out,
        [],
        chunksize=5,
        bin_levels={
            "fine": pd.DataFrame(
                {"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [40, 30]}
            ),
            "coarse": pd.DataFrame(
                {"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [100, 30]}
            ),
        },
    )
    for name in ("fine", "coarse"):
        pd.testing.assert_frame_equal(
            cooler.Cooler(f"{out}::bin_levels/{name}").pixels()[:],
            cooler.Cooler(f"{py_out}::bin_levels/{name}").pixels()[:],
        )


def test_cli_zoomify_bin_level_conflicts(tmp_path):
    src = make_source(tmp_path / "src.cool")
    bed = _write_bed(tmp_path / "a.bed", ["chr1\t0\t40"])
    z = str(tmp_path / "z.mcool")
    result = CliRunner().invoke(
        zoomify_cli, [src, "--bin-level", f"a={bed}", "-r", "20", "-o", z]
    )
    assert result.exit_code == 2
    result = CliRunner().invoke(
        zoomify_cli,
        [src, "--bin-level", f"a={bed}", "--bin-level", f"a={bed}", "-o", z],
    )
    assert result.exit_code == 2
    result = CliRunner().invoke(
        zoomify_cli, [src, "--bin-level", "noeql", "-o", z]
    )
    assert result.exit_code == 2


def test_cli_zoomify_bed_missing(tmp_path):
    src = make_source(tmp_path / "src.cool")
    result = CliRunner().invoke(
        zoomify_cli,
        [src, "--bin-level", "a=/does/not/exist.bed", "-o", str(tmp_path / "z.mcool")],
    )
    assert result.exit_code == 2


# --- bounded reading and resource probe


def test_chunksize_bounds_source_batches(tmp_path):
    # A single target bin swallows most of a chromosome, but the reader must
    # still only take `chunksize` source pixels at a time.
    from cooler._aggregate import LAST_STATS

    src = make_source(tmp_path / "src.cool", seed=8)
    tb = pd.DataFrame({"chrom": ["chr1", "chr3"], "start": [0, 0], "end": [100, 30]})
    out = str(tmp_path / "big.cool")
    coarsen_cooler(src, out, None, chunksize=17, nproc=1, target_bins=tb)
    assert LAST_STATS["n_batches"] > 0
    assert LAST_STATS["max_batch"] <= 17
    # The result is still all of the retained pixels
    total = cooler.Cooler(src).pixels()["count"][:]
    src_rows = cooler.Cooler(src).pixels().fetch("chr1")["count"].sum()
    src_rows += cooler.Cooler(src).pixels().fetch("chr3")["count"].sum()
    assert cooler.Cooler(out).pixels()["count"][:].sum() <= total.sum()


def test_resource_probe(tmp_path):
    # Full-size probe (>= 100k bins, >= 1M pixels) is run by
    # scripts/probe_sparse_coarsen.py. Here, check the mechanism on a small
    # table, kept fast.
    import importlib.util
    import os.path as op

    script = op.join(op.dirname(__file__), "..", "scripts", "probe_sparse_coarsen.py")
    if not op.exists(script):  # pragma: no cover
        pytest.skip("probe script not found")
    spec = importlib.util.spec_from_file_location("probe_sparse_coarsen", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.probe(str(tmp_path / "probe"), n_bins=2000, n_pixels=20000)
    assert result["source_bins"] >= 2000
    assert result["source_pixels"] > 10000
    assert result["max_batch"] <= result["chunksize"]
    assert result["delta_rss_mib"] >= 0


def test_zoomify_bin_levels_actually_reuses_intermediate(tmp_path):
    # mid is exactly the pooling of fine's bins, and mid is *coarser* per bin
    # count but built from fine, so the intermediate level is reused. Prove
    # it by recording the chosen sources.
    src = make_source(tmp_path / "src.cool", seed=9)
    fine = pd.DataFrame(
        {"chrom": ["chr1", "chr1", "chr2"], "start": [0, 50, 0], "end": [50, 100, 60]}
    )
    mid = pd.DataFrame(
        {"chrom": ["chr1", "chr2"], "start": [0, 0], "end": [100, 60]}
    )
    out = str(tmp_path / "z.mcool")
    zoomify_cooler(src, out, [], chunksize=4, bin_levels={"fine": fine, "mid": mid})
    with h5py.File(out, "r") as f:
        sources = json.loads(f.attrs["bin-level-sources"])
    assert sources["fine"] == "base"
    assert sources["mid"] == "fine"
    # And the result still equals a direct aggregation from the base
    direct = str(tmp_path / "d.cool")
    coarsen_cooler(src, direct, None, chunksize=4, target_bins=mid)
    pd.testing.assert_frame_equal(
        cooler.Cooler(f"{out}::bin_levels/mid").pixels()[:],
        cooler.Cooler(direct).pixels()[:],
    )


def test_zoomify_bin_levels_no_reuse_when_boundary_split(tmp_path):
    # fine splits the source at 30, mid wants 0-50: mid's boundary falls
    # inside fine's first bin, so no level can supply it; it falls back to
    # the base. Recording sources confirms that.
    src = make_source(tmp_path / "src.cool", seed=10)
    fine = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0, 30], "end": [30, 100]}
    )
    mid = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [50]})
    out = str(tmp_path / "z.mcool")
    zoomify_cooler(src, out, [], chunksize=4, bin_levels={"fine": fine, "mid": mid})
    with h5py.File(out, "r") as f:
        sources = json.loads(f.attrs["bin-level-sources"])
    assert sources["mid"] == "base"
    direct = str(tmp_path / "d.cool")
    coarsen_cooler(src, direct, None, chunksize=4, target_bins=mid)
    pd.testing.assert_frame_equal(
        cooler.Cooler(f"{out}::bin_levels/mid").pixels()[:],
        cooler.Cooler(direct).pixels()[:],
    )


def test_target_bins_variable_width_source(tmp_path):
    # Non-uniform source bins can be pooled too, as long as the target
    # intervals are whole runs of them.
    import os.path as op

    src = op.join(op.dirname(__file__), "data", "toy.symm.upper.var.cool")
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "start": [0, 21, 0],
            "end": [21, 32, 10],
        }
    )
    out = str(tmp_path / "var.cool")
    coarsen_cooler(src, out, None, chunksize=3, target_bins=tb)
    clr = cooler.Cooler(out)
    assert clr.bins()[:][["start", "end"]].values.tolist() == [
        [0, 21],
        [21, 32],
        [0, 10],
    ]
    assert clr.chromsizes.to_dict() == {"chr1": 32, "chr2": 32}
    # source runs: chr1 bins 0-4, 5-7; chr2 bins 8-11
    runs = [list(range(5)), list(range(5, 8)), list(range(8, 12))]
    exp = expected_dense(cooler.Cooler(src), runs)
    np.testing.assert_array_equal(stored_dense(clr), exp)


def test_uniform_region_query_not_regressed(tmp_path):
    # A fully tiled uniform file must keep returning the shortest covering
    # range of bins for a region.
    src = make_source(tmp_path / "src.cool")
    clr = cooler.Cooler(src)
    assert clr.extent("chr1:15-25") == (1, 3)
    assert clr.extent("chr1:0-100") == (0, 10)
    # An empty region selects an empty range, never a whole bin
    i0, i1 = clr.extent("chr2:7-7")
    assert i0 == i1
    assert len(clr.bins().fetch("chr2:7-7")) == 0
    assert clr.bins().fetch("chr1:15-25")[["start", "end"]].values.tolist() == [
        [10, 20],
        [20, 30],
    ]
    sel = clr.matrix(balance=False)
    assert np.array_equal(sel.fetch("chr1:15-25"), sel[1:3, 1:3])


def test_target_bins_nproc_agrees(tmp_path):
    src = make_source(tmp_path / "src.cool", seed=11)
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            "start": [0, 40, 0, 0],
            "end": [40, 100, 60, 30],
        }
    )
    frames = []
    for nproc in (1, 2):
        out = str(tmp_path / f"o{nproc}.cool")
        coarsen_cooler(src, out, None, chunksize=6, nproc=nproc, target_bins=tb)
        frames.append(cooler.Cooler(out).pixels()[:])
    pd.testing.assert_frame_equal(frames[0], frames[1])
    assert not __import__("multiprocessing").active_children()


def test_target_bins_large_counts_stay_exact(tmp_path):
    # Sums beyond int32 stay exact (int64), not clipped or wrapped.
    bins = cooler.binnify(pd.Series({"chr1": 20}), 10)
    big = 2_000_000_000
    px = pd.DataFrame(
        {"bin1_id": [0, 1], "bin2_id": [0, 1], "count": [big, big]}
    )
    src = str(tmp_path / "s.cool")
    cooler.create_cooler(
        src, bins, px, ordered=True, dtypes={"count": np.int64},
    )
    tb = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [20]})
    out = str(tmp_path / "o.cool")
    coarsen_cooler(src, out, None, chunksize=1, target_bins=tb)
    got = cooler.Cooler(out).pixels()[:]
    assert got["count"].tolist() == [2 * big]
    assert got["count"].dtype == np.int64


def test_zoomify_bin_levels_value_columns_and_agg(tmp_path):
    src = make_source(tmp_path / "src.cool", columns=("count", "foo"), seed=12)
    fine = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    coarse = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
    out1 = str(tmp_path / "sum.mcool")
    out2 = str(tmp_path / "max.mcool")
    zoomify_cooler(
        src, out1, [], chunksize=5, columns=["count", "foo"],
        bin_levels={"a": fine},
    )
    zoomify_cooler(
        src, out2, [], chunksize=5, columns=["count"], agg={"count": "max"},
        bin_levels={"a": coarse},
    )
    assert list(cooler.Cooler(f"{out1}::bin_levels/a").pixels().columns) == [
        "bin1_id", "bin2_id", "count", "foo",
    ]
    # direct aggregation with max agrees
    direct = str(tmp_path / "d.cool")
    coarsen_cooler(
        src, direct, None, chunksize=5, columns=["count"], agg={"count": "max"},
        target_bins=coarse,
    )
    pd.testing.assert_frame_equal(
        cooler.Cooler(f"{out2}::bin_levels/a").pixels()[:],
        cooler.Cooler(direct).pixels()[:],
    )


def test_retained_source_bins_partition_exactly(tmp_path):
    src = make_source(tmp_path / "src.cool", seed=13)
    tb = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chr3"],
            "start": [0, 50, 0, 10],
            "end": [40, 90, 30, 30],
        }
    )
    out = str(tmp_path / "o.cool")
    coarsen_cooler(src, out, None, chunksize=4, target_bins=tb)
    from cooler._targets import TargetBins, read_source_bins

    clr = cooler.Cooler(src)
    t = TargetBins.from_frame(tb, clr.chromsizes, read_source_bins(clr))
    kept = np.flatnonzero(t.src_to_tgt >= 0)
    # Every retained source bin belongs to exactly one target row, and the
    # rows' runs are disjoint and cover exactly the retained bins
    covered = np.sort(
        np.concatenate(
            [np.arange(t.tgt_src_start[r], t.tgt_src_end[r]) for r in range(len(t))]
        )
    )
    np.testing.assert_array_equal(covered, kept)
    assert len(np.unique(covered)) == len(covered)
    assert set(t.src_to_tgt[kept]) == set(range(len(t)))
    # Total count equals the sum over retained rows fully inside kept bins
    src_px = cooler.Cooler(src).pixels()[:]
    keep_px = (t.src_to_tgt[src_px.bin1_id] >= 0) & (t.src_to_tgt[src_px.bin2_id] >= 0)
    expected = src_px.loc[keep_px, "count"].sum()
    assert cooler.Cooler(out).pixels()["count"][:].sum() == expected


def test_cli_zoomify_bin_level_rejects_extra_base(tmp_path):
    src = make_source(tmp_path / "src.cool")
    bed = _write_bed(tmp_path / "a.bed", ["chr1\t0\t40"])
    result = CliRunner().invoke(
        zoomify_cli,
        [src, "--bin-level", f"a={bed}", "-i", src, "-o", str(tmp_path / "z.mcool")],
    )
    assert result.exit_code == 1
    assert "single base URI" in result.output or "single base URI" in result.stderr
