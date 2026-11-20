#!/usr/bin/env python
"""
Resource probe for sparse-target coarsening.

Builds a synthetic sparse cooler with >= 100,000 source bins and >= 1,000,000
source pixels, in which one target bin swallows most of the source bins of a
chromosome, then archives it with ``coarsen_cooler(target_bins=...)`` at
``chunksize=10000``, ``nproc=1``.

Measured in this process, with the same interpreter that imports the
dependencies:

* baseline RSS after importing cooler (and its dependencies);
* peak RSS sampled by a monitor thread while coarsening runs;
* the additional peak RSS = peak - baseline;
* the wall-clock time of the coarsening call;
* the sizes of the source-pixel batches actually read (the aggregator records
  them), to confirm ``chunksize`` bounds them.

Run it as a script to print the numbers, or import ``probe`` and pass a
``coarsen`` callable.

    python scripts/probe_sparse_coarsen.py [--bins 120000] [--pixels 1200000]
"""

from __future__ import annotations

import argparse
import os
import threading

import numpy as np
import pandas as pd

import cooler


def rss_bytes() -> int:
    """Current resident set size of this process, in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS not found in /proc/self/status")


class _PeakMonitor(threading.Thread):
    """Sample RSS in the background and keep the maximum."""

    def __init__(self, interval=0.001):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            self.peak = max(self.peak, rss_bytes())
            self._stop_event.wait(self.interval)

    def finish(self):
        self._stop_event.set()
        self.join()


def build_source(path, n_bins, n_pixels, seed=0):
    """A sparse symmetric-upper cooler with ``n_bins`` bins and ``n_pixels``."""
    lengths = {
        "chr1": n_bins * 1000,
        "chr2": max(n_bins // 20, 10) * 1000,
    }
    bins = cooler.binnify(pd.Series(lengths), 1000)
    n = len(bins)
    rng = np.random.default_rng(seed)
    # Dense in the first chromosome, sparse elsewhere
    chrom_codes = bins["chrom"].cat.codes.to_numpy()
    n_chr1 = int(np.sum(chrom_codes == 0))
    p_chr1 = 0.9
    which = rng.random(n_pixels) < p_chr1
    i = np.empty(n_pixels, dtype=np.int64)
    j = np.empty(n_pixels, dtype=np.int64)
    i[which] = rng.integers(0, n_chr1, int(which.sum()))
    j[which] = rng.integers(0, n_chr1, int(which.sum()))
    i[~which] = rng.integers(0, n, int((~which).sum()))
    j[~which] = rng.integers(0, n, int((~which).sum()))
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    keys = lo * n + hi
    keys = np.unique(keys)
    lo, hi = keys // n, keys % n
    px = pd.DataFrame(
        {"bin1_id": lo, "bin2_id": hi, "count": rng.integers(1, 10, len(lo))}
    )
    cooler.create_cooler(str(path), bins, px, ordered=True)
    return str(path)


def target_table(n_bins, giant_fraction=0.9):
    """One giant bin swallowing most of chr1, plus a few other bins."""
    giant = int(n_bins * giant_fraction) * 1000
    rows = [("chr1", 0, giant), ("chr1", giant, n_bins * 1000)]
    rows += [("chr2", s, s + 10000) for s in range(0, (n_bins // 20) * 1000, 10000)]
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def probe(workdir, n_bins=120_000, n_pixels=1_200_000, chunksize=10_000):
    """
    Measure the aggregation step in a fresh interpreter.

    The fixture is built here (it is large), but the coarsening runs in a
    child process that imports cooler and then takes its baseline *after*
    import and before reading the fixture. The child's peak is sampled by a
    monitor thread; the difference is the aggregation's extra memory.
    """
    import json
    import subprocess
    import sys

    os.makedirs(workdir, exist_ok=True)
    src = build_source(os.path.join(workdir, "source.cool"), n_bins, n_pixels)
    tb_path = os.path.join(workdir, "targets.csv")
    target_table(n_bins).to_csv(tb_path, index=False)
    out = os.path.join(workdir, "archived.cool")

    code = (
        "import sys, time\n"
        "import numpy as np, pandas as pd\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from probe_sparse_coarsen import rss_bytes, _PeakMonitor\n"
        "import cooler\n"
        "from cooler._reduce import coarsen_cooler\n"
        "import cooler._aggregate as AGG\n"
        "import json\n"
        "baseline = rss_bytes()\n"
        "tb = pd.read_csv(sys.argv[3])\n"
        "mon = _PeakMonitor(); mon.start()\n"
        "t0 = time.perf_counter()\n"
        "coarsen_cooler(sys.argv[2], sys.argv[4], None, chunksize=int(sys.argv[5]),\n"
        "               nproc=1, target_bins=tb)\n"
        "elapsed = time.perf_counter() - t0\n"
        "mon.finish()\n"
        "print('RESULT ' + json.dumps({\n"
        "  'baseline_rss_mib': baseline / 2**20,\n"
        "  'peak_rss_mib': mon.peak / 2**20,\n"
        "  'delta_rss_mib': (mon.peak - baseline) / 2**20,\n"
        "  'elapsed_s': elapsed,\n"
        "  'n_batches': AGG.LAST_STATS.get('n_batches'),\n"
        "  'max_batch': AGG.LAST_STATS.get('max_batch'),\n"
        "}))\n"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            script_dir,
            src,
            tb_path,
            out,
            str(chunksize),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    child = json.loads(line[len("RESULT ") :])

    clr = cooler.Cooler(src)
    result = {
        "source_bins": int(clr.info["nbins"]),
        "source_pixels": int(clr.info["nnz"]),
        "target_bins": int(cooler.Cooler(out).info["nbins"]),
        "chunksize": chunksize,
        **child,
        "output_uri": out,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--bins", type=int, default=120_000)
    parser.add_argument("--pixels", type=int, default=1_200_000)
    parser.add_argument("--chunksize", type=int, default=10_000)
    parser.add_argument("--limit-mib", type=float, default=256.0)
    args = parser.parse_args()
    import tempfile

    workdir = args.workdir or tempfile.mkdtemp(prefix="probe_sparse_")
    result = probe(workdir, args.bins, args.pixels, args.chunksize)
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"within {args.limit_mib} MiB: {result['delta_rss_mib'] <= args.limit_mib}")


if __name__ == "__main__":
    main()
