"""
Stabilization policy tests and ``cooler dump --expected`` integration.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from _expected_helpers import make_cooler

import cooler


def _decay_cooler(path, n=60, seed=0, enhancement=True, sparse_tail=True):
    rng = np.random.default_rng(seed)
    bins = pd.DataFrame(
        {
            "chrom": ["c"] * n,
            "start": np.arange(n) * 10,
            "end": np.arange(1, n + 1) * 10,
        }
    )
    rows, cols, counts = [], [], []
    for i in range(n):
        for j in range(i, n):
            d = (j - i) * 10
            lam = 100.0 / (1 + d / 50.0)
            if enhancement and 300 <= d <= 350:
                lam *= 3.0
            drop_pixel = sparse_tail and d > 450 and rng.random() < 0.7
            if drop_pixel and rng.random() < 0.5:
                continue
            value = rng.poisson(lam)
            if value > 0:
                rows.append(i)
                cols.append(j)
                counts.append(value)
    pixels = pd.DataFrame({"bin1_id": rows, "bin2_id": cols, "count": counts})
    return make_cooler(path, bins, pixels), bins, pixels


def test_stabilize_false_uses_exact_raw_mean(tmp_path):
    clr, _, _ = _decay_cooler(str(tmp_path / "d.cool"))
    edges = np.arange(0, 601, 10)
    raw = cooler.estimate_expected(clr, edges)
    assert raw.policy["stabilize"] is False
    finite = np.isfinite(raw.expected[0])
    np.testing.assert_allclose(
        raw.expected[0][finite], raw.raw_mean[0][finite], rtol=1e-12
    )
    assert set(np.unique(raw.status[0])) <= {"raw", "unavailable"}


def test_stabilize_outputs_are_finite_non_negative(tmp_path):
    clr, _, _ = _decay_cooler(str(tmp_path / "d.cool"))
    edges = np.arange(0, 601, 10)
    stab = cooler.estimate_expected(
        clr, edges, stabilize=True, min_support=20, window=5
    )
    vals = stab.expected[np.isfinite(stab.expected)]
    assert np.all(vals >= 0)
    assert set(np.unique(stab.status)) <= {
        "raw",
        "despiked",
        "smoothed",
        "unavailable",
    }
    # Unsupported buckets stay explicitly unavailable and are never filled.
    unsupported = stab.n_pairs == 0
    assert np.all(np.isnan(stab.expected[unsupported]))
    assert np.all(stab.status[unsupported] == "unavailable")


def test_stabilize_retains_sustained_enhancement(tmp_path):
    clr, _, _ = _decay_cooler(str(tmp_path / "d.cool"))
    edges = np.arange(0, 601, 10)
    raw = cooler.estimate_expected(clr, edges)
    stab = cooler.estimate_expected(
        clr, edges, stabilize=True, min_support=20, window=5
    )
    # Buckets covering the 300-350 bp enhancement band (indices 30..35) must
    # retain their elevated signal rather than being smoothed to the trend.
    band = slice(30, 36)
    raw_band = raw.expected[0][band]
    stab_band = stab.expected[0][band]
    np.testing.assert_allclose(stab_band, raw_band, rtol=0.05)
    assert set(stab.status[0][band]) == {"raw"}


def test_stabilize_reduces_tail_noise_and_keeps_support_stats(tmp_path):
    clr, _, _ = _decay_cooler(str(tmp_path / "d.cool"))
    edges = np.arange(0, 601, 10)
    raw = cooler.estimate_expected(clr, edges)
    stab = cooler.estimate_expected(
        clr, edges, stabilize=True, min_support=20, window=5
    )
    # Raw statistics are preserved verbatim for auditing.
    np.testing.assert_array_equal(stab.n_pairs, raw.n_pairs)
    np.testing.assert_allclose(stab.raw_sum, raw.raw_sum)
    np.testing.assert_allclose(stab.raw_mean, raw.raw_mean, equal_nan=True)
    # The low-support tail is smoothed and monotone non-increasing.
    tail_idx = np.flatnonzero((stab.n_pairs[0] > 0) & (stab.n_pairs[0] < 20))
    tail_vals = stab.expected[0][tail_idx]
    tail_vals = tail_vals[np.isfinite(tail_vals)]
    assert np.all(np.diff(tail_vals) <= 1e-12)
    # Smoothing reduces spread relative to the raw tail.
    raw_tail = raw.raw_mean[0][tail_idx]
    raw_tail = raw_tail[np.isfinite(raw_tail)]
    assert np.nanstd(tail_vals) <= np.nanstd(raw_tail) + 1e-12


def test_constant_positive_background_is_preserved(tmp_path):
    n = 30
    bins = pd.DataFrame(
        {
            "chrom": ["c"] * n,
            "start": np.arange(n) * 10,
            "end": np.arange(1, n + 1) * 10,
        }
    )
    rows, cols = np.triu_indices(n)
    pixels = pd.DataFrame(
        {"bin1_id": rows, "bin2_id": cols, "count": np.full(rows.size, 5)}
    )
    clr = make_cooler(str(tmp_path / "const.cool"), bins, pixels)
    edges = np.arange(0, 301, 10)
    stab = cooler.estimate_expected(clr, edges, stabilize=True)
    vals = stab.expected[0][np.isfinite(stab.expected[0])]
    np.testing.assert_allclose(vals, np.full_like(vals, 5.0), rtol=1e-12)


def test_stabilize_does_not_borrow_across_chromosomes(tmp_path):
    # Two chromosomes with very different levels; stabilization must keep
    # each chromosome's estimates independent.
    bins = pd.DataFrame(
        {
            "chrom": ["a"] * 20 + ["b"] * 20,
            "start": list(np.arange(20) * 10) * 2,
            "end": list(np.arange(1, 21) * 10) * 2,
        }
    )
    rng = np.random.default_rng(1)
    rows, cols, counts = [], [], []
    for i in range(40):
        for j in range(i, 40):
            if i // 20 != j // 20:
                continue
            level = 100.0 if i < 20 else 1.0
            v = rng.poisson(level)
            if v:
                rows.append(i)
                cols.append(j)
                counts.append(v)
    pixels = pd.DataFrame({"bin1_id": rows, "bin2_id": cols, "count": counts})
    clr = make_cooler(str(tmp_path / "two.cool"), bins, pixels)
    edges = [0, 10, 50, 200]
    stab = cooler.estimate_expected(clr, edges, stabilize=True)
    a = stab.expected[0][np.isfinite(stab.expected[0])]
    b = stab.expected[1][np.isfinite(stab.expected[1])]
    assert np.median(a) > 10 * np.median(b)


def test_policy_parameters_are_stored_and_auditable(tmp_path):
    clr, _, _ = _decay_cooler(str(tmp_path / "d.cool"))
    edges = np.arange(0, 601, 10)
    stab = cooler.estimate_expected(
        clr,
        edges,
        stabilize=True,
        min_support=12,
        window=4,
        huber_delta=2.0,
        spike_z=4.5,
        tail_monotone=False,
    )
    policy = stab.policy
    assert policy["stabilize"] is True
    assert policy["min_support"] == 12
    assert policy["window"] == 4
    assert policy["huber_delta"] == 2.0
    assert policy["spike_z"] == 4.5
    assert policy["tail_monotone"] is False
    cooler.save_expected(clr, stab, "stab")
    loaded = cooler.load_expected(clr, "stab")
    assert loaded.policy == policy


def test_dump_expected_matches_python_reads(tmp_path, run_cli):
    path = str(tmp_path / "dump.cool")
    bins = pd.DataFrame(
        {
            "chrom": ["chr1"] * 4 + ["chr2"] * 3,
            "start": [0, 10, 20, 30, 0, 10, 20],
            "end": [10, 20, 30, 40, 10, 20, 30],
        }
    )
    pixels = pd.DataFrame(
        {
            "bin1_id": [0, 0, 1, 2, 4, 4, 5],
            "bin2_id": [0, 1, 2, 3, 4, 5, 6],
            "count": [5, 2, 3, 1, 4, 0, 2],
        }
    )
    clr = make_cooler(path, bins, pixels)
    res = cooler.estimate_expected(clr, [0, 10, 20, 100])
    cooler.save_expected(clr, res, "bg")

    result = run_cli("dump", path, "-H", "--expected", "bg")
    dumped = pd.read_csv(io.StringIO(result.stdout), sep="\t")
    assert list(dumped.columns) == [
        "bin1_id",
        "bin2_id",
        "count",
        "expected",
        "oe",
    ]
    py = clr.matrix(balance=False, expected=res, as_pixels=True)[:]
    # CSV round-trip normalizes integer dtypes; compare values and columns.
    assert list(dumped.columns) == list(py.columns)
    pd.testing.assert_frame_equal(
        dumped.reset_index(drop=True),
        py.reset_index(drop=True),
        check_dtype=False,
    )


def test_dump_expected_fill_lower_and_range(tmp_path, run_cli):
    path = str(tmp_path / "dump2.cool")
    bins = pd.DataFrame(
        {
            "chrom": ["chr1"] * 4,
            "start": [0, 10, 20, 30],
            "end": [10, 20, 30, 40],
        }
    )
    pixels = pd.DataFrame(
        {"bin1_id": [0, 0, 1, 2], "bin2_id": [0, 1, 2, 3], "count": [5, 2, 3, 1]}
    )
    clr = make_cooler(path, bins, pixels)
    res = cooler.estimate_expected(clr, [0, 10, 100])
    cooler.save_expected(clr, res, "bg")
    result = run_cli(
        "dump", path, "-H", "--expected", "bg", "--fill-lower", "-r", "chr1"
    )
    dumped = pd.read_csv(io.StringIO(result.stdout), sep="\t")
    # Fill-lower reflects the upper triangle: expect symmetric pairs.
    assert len(dumped) == 7  # 4 stored + 3 reflected
    # Default dump without the flag is unchanged.
    default = run_cli("dump", path, "-H")
    default_df = pd.read_csv(io.StringIO(default.stdout), sep="\t")
    assert list(default_df.columns) == ["bin1_id", "bin2_id", "count"]


def test_dump_expected_rejects_balanced_combo(tmp_path, run_cli):
    path = str(tmp_path / "dump3.cool")
    bins = pd.DataFrame({"chrom": ["c"] * 2, "start": [0, 10], "end": [10, 20]})
    pixels = pd.DataFrame({"bin1_id": [0, 0], "bin2_id": [0, 1], "count": [1, 2]})
    clr = make_cooler(path, bins, pixels)
    res = cooler.estimate_expected(clr, [0, 10, 100])
    cooler.save_expected(clr, res, "bg")
    run_cli("dump", path, "--expected", "bg", "--balanced", exit_code=2)
    run_cli("dump", path, "--expected", "missing", exit_code=1)


def test_dump_expected_column_subsets(tmp_path, run_cli):
    path = str(tmp_path / "dump4.cool")
    bins = pd.DataFrame({"chrom": ["c"] * 3, "start": [0, 10, 20], "end": [10, 20, 30]})
    pixels = pd.DataFrame(
        {"bin1_id": [0, 0, 1], "bin2_id": [0, 1, 2], "count": [1, 2, 4]}
    )
    clr = make_cooler(path, bins, pixels)
    res = cooler.estimate_expected(clr, [0, 10, 100])
    cooler.save_expected(clr, res, "bg")

    default = pd.read_csv(
        io.StringIO(run_cli("dump", path, "-H", "--expected", "bg").stdout),
        sep="\t",
    )
    assert list(default.columns) == ["bin1_id", "bin2_id", "count", "expected", "oe"]

    oe_only = pd.read_csv(
        io.StringIO(
            run_cli(
                "dump", path, "-H", "--expected", "bg", "-c", "bin1_id,bin2_id,oe"
            ).stdout
        ),
        sep="\t",
    )
    assert list(oe_only.columns) == ["bin1_id", "bin2_id", "oe"]

    count_only = pd.read_csv(
        io.StringIO(
            run_cli(
                "dump",
                path,
                "-H",
                "--expected",
                "bg",
                "-c",
                "bin1_id,bin2_id,count",
            ).stdout
        ),
        sep="\t",
    )
    assert list(count_only.columns) == ["bin1_id", "bin2_id", "count"]

    # A field that does not exist is rejected.
    run_cli(
        "dump",
        path,
        "--expected",
        "bg",
        "-c",
        "bin1_id,bin2_id,nonexistent",
        exit_code=2,
    )

    # A real non-count stored field is rejected in combination with expected.
    path2 = str(tmp_path / "dump5.cool")
    clr2 = make_cooler(
        path2,
        bins,
        pixels,
        extra_pixel_cols={"signal": np.arange(len(pixels), dtype=float)},
    )
    cooler.save_expected(clr2, cooler.estimate_expected(clr2, [0, 10, 100]), "bg")
    run_cli(
        "dump",
        path2,
        "--expected",
        "bg",
        "-c",
        "bin1_id,bin2_id,signal",
        exit_code=2,
    )
