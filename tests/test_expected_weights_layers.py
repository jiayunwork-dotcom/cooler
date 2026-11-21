"""
Weight interpretation, analysis-layer persistence and matrix-query tests for
the distance-background machinery.
"""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
from _expected_helpers import add_weight_column, make_cooler, reference_stats

import cooler
from cooler.core import CSRReader, DirectRangeQuery2D, FillLowerRangeQuery2D

RTOL = 1e-10
ATOL = 1e-12

BINS = pd.DataFrame(
    {
        "chrom": ["chr1"] * 5 + ["chr2"] * 4,
        "start": [0, 10, 20, 30, 40, 0, 10, 20, 30],
        "end": [10, 20, 30, 40, 50, 10, 20, 30, 40],
    }
)
PIXELS = pd.DataFrame(
    {
        "bin1_id": [0, 0, 1, 2, 3, 5, 5, 6, 7],
        "bin2_id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "count": [3, 6, 2, 8, 4, 5, 1, 7, 2],
    }
)
EDGES = [0, 10, 20, 50]


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "w.cool")


@pytest.fixture
def clr(path):
    return make_cooler(path, BINS, PIXELS)


def test_unweighted_all_bins_valid(clr):
    res = cooler.estimate_expected(clr, EDGES)
    assert res.valid.all()
    assert not res.balance
    assert res.weight_values is None
    assert res.weight_binding is None


def test_multiplicative_weights_match_reference(path, clr):
    weights = np.array([1.0, np.nan, 0.5, 0.0, 2.0, 1.0, 1.5, np.nan, 0.8])
    add_weight_column(path, "weight", weights)
    res = cooler.estimate_expected(clr, EDGES, balance=True)
    assert res.balance and not res.divisive_weights
    assert res.valid.tolist() == [
        True,
        False,
        True,
        False,
        True,
        True,
        True,
        False,
        True,
    ]
    offsets = res.chrom_offsets
    for ci in range(len(res.chrom_names)):
        lo, hi = int(offsets[ci]), int(offsets[ci + 1])
        sub = BINS.iloc[lo:hi].reset_index(drop=True)
        sub_px = PIXELS[
            PIXELS["bin1_id"].between(lo, hi - 1)
            & PIXELS["bin2_id"].between(lo, hi - 1)
        ].copy()
        sub_px["bin1_id"] -= lo
        sub_px["bin2_id"] -= lo
        ref = reference_stats(
            sub, sub_px, EDGES, weights=weights[lo:hi], divisive=False
        )
        np.testing.assert_array_equal(res.n_pairs[ci], ref["n_pairs"])
        np.testing.assert_allclose(
            res.observed_sum[ci], ref["observed_sum"], rtol=RTOL, atol=ATOL
        )
        # Raw counts are retained independently of weighting.
        np.testing.assert_allclose(
            res.raw_sum[ci], ref["raw_sum"], rtol=RTOL, atol=ATOL
        )


def test_divisive_weights_and_alias_default(path, clr):
    weights = np.full(9, 2.0)
    add_weight_column(path, "KR", weights)
    # Alias rule: KR is divisive by default.
    res_default = cooler.estimate_expected(clr, EDGES, balance="KR")
    assert res_default.divisive_weights is True
    # Explicit override to multiplicative.
    res_mult = cooler.estimate_expected(
        clr, EDGES, balance="KR", divisive_weights=False
    )
    assert res_mult.divisive_weights is False
    # Divisive observed values differ from multiplicative ones.
    assert not np.allclose(
        res_default.observed_sum, res_mult.observed_sum, rtol=RTOL, atol=ATOL
    )
    # A generic column name defaults to multiplicative.
    add_weight_column(path, "custom", weights)
    res_custom = cooler.estimate_expected(clr, EDGES, balance="custom")
    assert res_custom.divisive_weights is False


def test_invalid_weights_rejected(path, clr):
    add_weight_column(path, "weight", np.array([1.0, -1.0] + [1.0] * 7))
    with pytest.raises(ValueError, match="negative"):
        cooler.estimate_expected(clr, EDGES, balance=True)
    add_weight_column(path, "weight", np.array([1.0, np.inf] + [1.0] * 7))
    with pytest.raises(ValueError, match="finite"):
        cooler.estimate_expected(clr, EDGES, balance=True)
    with pytest.raises(ValueError, match="weight"):
        cooler.estimate_expected(clr, EDGES, balance="missing_col")


def test_layer_roundtrip_and_listing(path, clr):
    res = cooler.estimate_expected(clr, EDGES)
    assert cooler.list_expected(clr) == []
    cooler.save_expected(clr, res, "bg")
    assert cooler.list_expected(clr) == ["bg"]
    loaded = cooler.load_expected(clr, "bg")
    np.testing.assert_array_equal(loaded.distance_edges, res.distance_edges)
    np.testing.assert_array_equal(loaded.n_pairs, res.n_pairs)
    np.testing.assert_allclose(loaded.expected, res.expected, equal_nan=True)
    np.testing.assert_array_equal(loaded.usable, res.usable)
    np.testing.assert_array_equal(loaded.valid, res.valid)
    assert loaded.policy == res.policy
    assert loaded.layer_name == "bg"
    # Querying via the stored layer name matches the in-memory result.
    a = clr.matrix(balance=False, expected=res)[:]
    b = clr.matrix(expected="bg")[:]
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    assert np.array_equal(nan_a, nan_b)
    np.testing.assert_allclose(a[~nan_a], b[~nan_b], rtol=RTOL, atol=ATOL)


def test_weighted_layer_roundtrip(path, clr):
    weights = np.array([1.0, np.nan, 0.5, 1.0, 2.0, 1.0, 1.5, 1.0, 0.8])
    add_weight_column(path, "weight", weights)
    res = cooler.estimate_expected(clr, EDGES, balance=True)
    cooler.save_expected(clr, res, "wbg")
    loaded = cooler.load_expected(clr, "wbg")
    assert loaded.balance and loaded.weight_name == "weight"
    np.testing.assert_allclose(loaded.weight_values, res.weight_values)
    # The original weight column is untouched and still the balancing weight.
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["bins/weight"][:], weights)
    # Matrix reads use the bound weights automatically.
    oe = clr.matrix(expected="wbg")[:]
    assert np.isnan(oe[:, 1]).all()  # excluded bin row/col
    assert np.isnan(oe[1, :]).all()


def test_layer_rejects_changed_weights(path, clr):
    weights = np.ones(9)
    add_weight_column(path, "weight", weights)
    res = cooler.estimate_expected(clr, EDGES, balance=True)
    cooler.save_expected(clr, res, "wbg")
    with h5py.File(path, "r+") as f:
        f["bins/weight"][:] = np.full(9, 1.5)
    with pytest.raises(ValueError, match="weights"):
        cooler.load_expected(clr, "wbg")
    with pytest.raises(ValueError, match="weights"):
        clr.matrix(expected="wbg")[:]


def test_layer_rejects_changed_bins(path, clr):
    res = cooler.estimate_expected(clr, EDGES)
    cooler.save_expected(clr, res, "bg")
    with h5py.File(path, "r+") as f:
        starts = f["bins/start"][:]
        starts[0] = 1
        f["bins/start"][:] = starts
    with pytest.raises(ValueError, match="bin layout"):
        cooler.load_expected(clr, "bg")
    with pytest.raises(ValueError, match="bin layout"):
        clr.matrix(expected=res)[:]


def test_layer_survives_file_relocation(tmp_path, path, clr):
    res = cooler.estimate_expected(clr, EDGES)
    cooler.save_expected(clr, res, "bg")
    new_dir = tmp_path / "moved"
    new_dir.mkdir()
    new_path = new_dir / "copy.cool"
    shutil.copy(path, new_path)
    moved = cooler.Cooler(str(new_path))
    loaded = cooler.load_expected(moved, "bg")
    np.testing.assert_allclose(loaded.expected, res.expected, equal_nan=True)
    moved.matrix(expected="bg")[:]


def test_overwrite_semantics(path, clr):
    res = cooler.estimate_expected(clr, EDGES)
    cooler.save_expected(clr, res, "bg")
    with pytest.raises(ValueError, match="already exists"):
        cooler.save_expected(clr, res, "bg")
    # A failed overwrite leaves the original layer intact.
    res2 = cooler.estimate_expected(clr, [0, 10, 50])
    with pytest.raises((ValueError, OSError)):
        cooler.save_expected(clr, res2, "bg")
    loaded = cooler.load_expected(clr, "bg")
    np.testing.assert_array_equal(loaded.distance_edges, res.distance_edges)
    # Explicit overwrite works.
    cooler.save_expected(clr, res2, "bg", overwrite=True)
    loaded = cooler.load_expected(clr, "bg")
    np.testing.assert_array_equal(loaded.distance_edges, [0, 10, 50])
    cooler.remove_expected(clr, "bg")
    assert cooler.list_expected(clr) == []
    with pytest.raises(KeyError):
        cooler.remove_expected(clr, "bg")


def test_balance_config_compatibility(path, clr):
    res = cooler.estimate_expected(clr, EDGES)  # unweighted
    # Omitting balance must not be treated as an explicit balance=True.
    clr.matrix(expected=res)[:]
    clr.matrix(balance=False, expected=res)[:]
    with pytest.raises(ValueError, match="incompatible"):
        clr.matrix(balance=True, expected=res)
    weights = np.ones(9)
    add_weight_column(path, "weight", weights)
    wres = cooler.estimate_expected(clr, EDGES, balance=True)
    clr.matrix(expected=wres)[:]
    clr.matrix(balance="weight", expected=wres)[:]
    with pytest.raises(ValueError, match="incompatible"):
        clr.matrix(balance=False, expected=wres)
    with pytest.raises(ValueError, match="incompatible"):
        clr.matrix(balance="other_name", expected=wres)
    with pytest.raises(ValueError, match="incompatible"):
        clr.matrix(balance="weight", divisive_weights=True, expected=wres)
    # Non-count fields cannot combine with expected.
    with pytest.raises(ValueError):
        clr.matrix(field="other", expected=res)


def test_windowed_reads_match_full_slice(clr):
    res = cooler.estimate_expected(clr, EDGES)
    full = clr.matrix(balance=False, expected=res)[:]
    for box in [(1, 4, 2, 5), (0, 5, 5, 9), (3, 7, 3, 7), (0, 0, 0, 9)]:
        i0, i1, j0, j1 = box
        got = clr.matrix(balance=False, expected=res)[i0:i1, j0:j1]
        np.testing.assert_array_equal(np.isnan(got), np.isnan(full[i0:i1, j0:j1]))
        np.testing.assert_allclose(got, full[i0:i1, j0:j1], rtol=RTOL, atol=ATOL)
    # fetch(region) path agrees too.
    region = clr.matrix(balance=False, expected=res).fetch("chr1")
    np.testing.assert_allclose(region, full[0:5, 0:5], rtol=RTOL, atol=ATOL)


def test_dense_sparse_as_pixels_agree(clr):
    res = cooler.estimate_expected(clr, EDGES)
    dense = clr.matrix(balance=False, expected=res)[:]
    sparse = clr.matrix(balance=False, expected=res, sparse=True)[:].toarray()
    np.testing.assert_array_equal(np.isnan(dense), np.isnan(sparse))
    np.testing.assert_allclose(dense, sparse, rtol=RTOL, atol=ATOL)

    px = clr.matrix(balance=False, expected=res, as_pixels=True)[:]
    # Only originally stored rows; raw count preserved; new columns present.
    assert len(px) == len(PIXELS)
    np.testing.assert_array_equal(px["count"].to_numpy(), PIXELS["count"].to_numpy())
    assert {"expected", "oe"} <= set(px.columns)
    # Stored-pixel OE values agree with the dense matrix at those positions.
    for _, row in px.iterrows():
        i, j = int(row["bin1_id"]), int(row["bin2_id"])
        if np.isnan(dense[i, j]):
            assert np.isnan(row["oe"])
        else:
            np.testing.assert_allclose(row["oe"], dense[i, j], rtol=RTOL, atol=ATOL)


def test_low_level_engines_with_background(clr):
    res = cooler.estimate_expected(clr, EDGES)
    bg = res.as_background()
    with clr.open() as h5:
        reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
        bbox = (0, 9, 0, 9)
        direct = DirectRangeQuery2D(
            reader, ["count", "expected", "oe"], bbox, 3, background=bg
        )
        fill = FillLowerRangeQuery2D(
            reader, ["count", "expected", "oe"], bbox, 3, background=bg
        )
        # Frame / array / sparse interpretations are consistent.
        d_frame = direct.to_frame().sort_values(["bin1_id", "bin2_id"])
        f_frame = fill.to_frame()
        # FillLower adds reflected lower-triangle rows; Direct does not.
        assert len(f_frame) >= len(d_frame)
        # Reflected rows carry the same virtual values as their upper source.
        upper = f_frame.set_index(["bin1_id", "bin2_id"])
        for _, row in d_frame.iterrows():
            i, j = int(row["bin1_id"]), int(row["bin2_id"])
            if i != j:
                reflected = upper.loc[(j, i)]
                np.testing.assert_allclose(reflected["oe"], row["oe"])
                np.testing.assert_allclose(reflected["expected"], row["expected"])
        # A multi-field engine is frame-only; matrices need one value column.
        with pytest.raises(ValueError):
            direct.to_array()
        # Chunked consumption matches one-shot consumption.
        chunks = [direct.get_chunk(k) for k in range(direct.n_chunks)]
        cat = pd.concat([pd.DataFrame(c) for c in chunks]).sort_values(
            ["bin1_id", "bin2_id"]
        )
        pd.testing.assert_frame_equal(
            cat.reset_index(drop=True), d_frame.reset_index(drop=True)
        )
        # Virtual-only field requests work and keep implicit zeros implicit.
        oe_only = DirectRangeQuery2D(reader, "oe", bbox, 2, background=bg)
        oe_arr = oe_only.to_array()
        assert oe_arr.shape == (9, 9)
        np.testing.assert_array_equal(
            np.isnan(oe_only.to_sparse_matrix().toarray()), np.isnan(oe_arr)
        )
        # Default field requests (no background) are unchanged in shape.
        plain = DirectRangeQuery2D(reader, "count", bbox, 3)
        assert plain.to_frame().columns.tolist() == ["bin1_id", "bin2_id", "count"]
