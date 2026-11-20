"""
Out-of-core aggregation of sparse coolers onto explicit target bin tables.

The source pixel table is processed one *target row* at a time.  Because the
source pixels are sorted by ``bin1_id``, and a target row covers a contiguous
run of source bins, each target row corresponds to one contiguous run of the
source pixel table: no source row is visited twice and no unfinished state is
carried across target rows.

Within a target row the source pixels are read in batches of at most
``chunksize`` records.  A target bin may be arbitrarily wide -- a single one
can pool most of a chromosome -- so batches are bounded by *source pixels* and
are never widened to cover a whole target row.  The state kept between batches
is just the running aggregate of the target row: one value per output pixel of
that row, never the source pixels themselves.

Workers process disjoint groups of target rows, so their outputs concatenate
without any cross-worker merging.  Groups are bounded by an approximate
source-pixel budget, which also bounds the result held in memory per group.
"""

from __future__ import annotations

import math

import numpy as np

from ._logging import get_logger

__all__ = ["LAST_STATS", "TargetAggregator"]

logger = get_logger("cooler.reduce")

# Diagnostics of the most recent iteration in this process: how many source
# pixel batches were read and the largest one. Useful to confirm that
# ``chunksize`` bounds the reads. Workers keep their own stats; with the
# default single process these reflect the whole run.
LAST_STATS: dict[str, int] = {}

# Worker-local aggregator, set once per process by the pool initializer.
_WORKER_AGGREGATOR = None


def _init_worker(aggregator) -> None:  # pragma: no cover (runs in workers)
    global _WORKER_AGGREGATOR
    _WORKER_AGGREGATOR = aggregator


def _run_group(specs):  # pragma: no cover (runs in workers)
    return _WORKER_AGGREGATOR.run(specs)


def _combine_scalar(agg, old, new):
    if agg == "sum":
        return old + new
    if agg == "min":
        return min(old, new)
    return max(old, new)


def _reduce_batch(t2, vd, pos, stop, keep, aggs):
    """
    Combine a batch's source pixels by target column.

    Returns the unique target column IDs, in ascending order, and, for each
    value column, the combined values. Integer columns accumulate in int64,
    so sums stay exact; float columns keep their precision.
    """
    keys = t2.astype(np.int64, copy=False)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    uniq = sk[starts]
    out = []
    for i, dset in enumerate(vd):
        v = np.asarray(dset[pos:stop])
        if not keep.all():
            v = v[keep]
        v = v[order]
        if np.issubdtype(v.dtype, np.integer):
            acc = v.astype(np.int64, copy=False)
        elif not np.issubdtype(v.dtype, np.floating):
            acc = v.astype(np.float64, copy=False)
        else:
            acc = v
        agg = aggs[i]
        if agg == "sum":
            out.append(np.add.reduceat(acc, starts))
        elif agg == "min":
            out.append(np.minimum.reduceat(acc, starts))
        else:
            out.append(np.maximum.reduceat(acc, starts))
    return uniq, out


class TargetAggregator:
    """
    Aggregate source pixels onto an explicit target bin table.

    Parameters
    ----------
    source_uri : str
        Input cooler URI.
    target : cooler._targets.TargetBins
        Validated target bin table, carrying the source bin mapping.
    columns : list of str
        Value columns to aggregate.
    agg : dict of str to str
        Aggregation per value column: "sum", "min" or "max".
    chunksize : int
        Maximum number of source pixels read at a time.
    nproc : int, optional
        Number of workers. Default is 1.
    """

    def __init__(self, source_uri, target, columns, agg, chunksize, nproc=1):
        self.source_uri = source_uri
        self.target = target
        self.columns = list(columns)
        self.aggs = {c: agg.get(c, "sum") for c in self.columns}
        self.chunksize = int(chunksize)
        if self.chunksize < 1:
            raise ValueError("chunksize must be a positive integer.")
        self.nproc = int(nproc)
        self.out_dtypes = self._target_dtypes()
        self._stats = None

    def _target_dtypes(self) -> dict[str, np.dtype]:
        from .api import Cooler

        src = Cooler(self.source_uri).pixels().dtypes
        out = {}
        for c in self.columns:
            kind = np.dtype(src[c]).kind
            if self.aggs[c] == "sum" and kind in "iub":
                out[c] = np.dtype(np.int64)
            elif kind in "iuf":
                out[c] = np.dtype(np.float64)
            else:
                out[c] = np.dtype(src[c])
        return out

    # -- planning: which source rows feed each target row

    def row_specs(self) -> list[tuple[int, int, int]]:
        """``(target_row, lo, hi)`` for each target row with source pixels."""
        from .api import Cooler

        target = self.target
        n = target.nbins
        if n == 0:
            return []
        starts = target.tgt_src_start
        stops = target.tgt_src_end
        needed = np.unique(np.r_[starts, stops])
        with Cooler(self.source_uri).open("r") as grp:
            dset = grp["indexes/bin1_offset"]
            offsets = _read_indices(dset, needed)
        pos = {int(k): int(v) for k, v in zip(needed.tolist(), offsets.tolist())}
        specs = []
        for row in range(n):
            lo = pos[int(starts[row])]
            hi = pos[int(stops[row])]
            if hi > lo:
                specs.append((row, lo, hi))
        return specs

    def tasks(self, specs=None) -> list[list[tuple[int, int, int]]]:
        """
        Group row specs into work units.

        A group holds roughly ``nproc`` times the per-worker share of the
        source pixels, but never more than a few ``chunksize`` batches, so the
        result of a group stays small however large the source is. A single
        target row wider than the budget still forms its own group: a row is
        the unit of the aggregation, and its output is bounded by the number
        of target columns, not by its source pixels.
        """
        if specs is None:
            specs = self.row_specs()
        if not specs:
            return []
        total = sum(hi - lo for _, lo, hi in specs)
        n_groups = max(1, self.nproc * 4)
        share = math.ceil(total / n_groups)
        cap = 8 * self.chunksize
        budget = max(self.chunksize, min(share, cap))
        groups: list[list[tuple[int, int, int]]] = []
        cur: list[tuple[int, int, int]] = []
        cur_n = 0
        for spec in specs:
            n = spec[2] - spec[1]
            if cur and cur_n + n > budget:
                groups.append(cur)
                cur, cur_n = [], 0
            cur.append(spec)
            cur_n += n
        if cur:
            groups.append(cur)
        return groups

    # -- aggregation

    def run(self, specs: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
        """Aggregate a group of target-row specs into one concatenated dict."""
        from .api import Cooler

        cols = self.columns
        dtypes = self.out_dtypes
        aggs = [self.aggs[c] for c in cols]
        src_to_tgt = self.target.src_to_tgt
        parts = []
        clr = Cooler(self.source_uri)
        # One file open for the whole group of rows
        with clr.open("r") as grp:
            pixels = grp["pixels"]
            d2 = pixels["bin2_id"]
            vd = [pixels[c] for c in cols]
            for row, lo, hi in specs:
                # A target row pools several source rows, whose bin2 IDs each
                # restart, so target column IDs are not globally sorted within
                # the row. Accumulate one value per target column of this row:
                # at most one aggregate per target bin, never per source pixel.
                acc: dict[int, list] = {}
                get = acc.get
                pos = lo
                while pos < hi:
                    stop = min(pos + self.chunksize, hi)
                    if self._stats is not None:
                        self._stats["n_batches"] += 1
                        self._stats["max_batch"] = max(
                            self._stats["max_batch"], stop - pos
                        )
                    b2 = np.asarray(d2[pos:stop])
                    t2 = src_to_tgt[b2]
                    keep = t2 >= 0
                    if not keep.all():
                        t2 = t2[keep]
                    if len(t2):
                        ukeys, uvals = _reduce_batch(
                            t2, vd, pos, stop, keep, aggs
                        )
                        for j, k in enumerate(ukeys.tolist()):
                            old = get(k)
                            if old is None:
                                acc[k] = [uvals[i][j] for i in range(len(cols))]
                            else:
                                for i in range(len(cols)):
                                    old[i] = _combine_scalar(
                                        aggs[i], old[i], uvals[i][j]
                                    )
                    pos = stop
                if not acc:
                    continue
                keys = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
                keys.sort()
                part = {
                    "bin1_id": np.full(len(keys), row, dtype=np.int64),
                    "bin2_id": keys,
                }
                ordered = [acc[int(k)] for k in keys]
                for i, c in enumerate(cols):
                    part[c] = np.array([v[i] for v in ordered], dtype=dtypes[c])
                parts.append(part)
        return _concat(parts, cols, dtypes)

    # -- iteration

    def __iter__(self):
        global LAST_STATS
        LAST_STATS = {"n_batches": 0, "max_batch": 0}
        self._stats = LAST_STATS
        tasks = self.tasks()
        if not tasks:
            return
        if self.nproc > 1:
            yield from self._iter_parallel(tasks)
        else:
            for specs in tasks:
                yield self.run(specs)

    def _iter_parallel(self, tasks):
        from .parallel import _ProcessPool

        pool = _ProcessPool(self.nproc, initializer=_init_worker, initargs=(self,))
        try:
            yield from pool.map(_run_group, tasks)
        finally:
            pool.close()


def _concat(parts, columns, dtypes) -> dict[str, np.ndarray]:
    if not parts:
        out = {
            "bin1_id": np.array([], dtype=np.int64),
            "bin2_id": np.array([], dtype=np.int64),
        }
        for c in columns:
            out[c] = np.array([], dtype=dtypes[c])
        return out
    result = {
        "bin1_id": np.concatenate([d["bin1_id"] for d in parts]),
        "bin2_id": np.concatenate([d["bin2_id"] for d in parts]),
    }
    for c in columns:
        result[c] = np.concatenate([d[c] for d in parts])
    return result


def _read_indices(dset, indices: np.ndarray) -> np.ndarray:
    """Read the values of sorted unique indices from a 1D dataset."""
    indices = np.asarray(indices, dtype=np.int64)
    out = np.empty(len(indices), dtype=np.int64)
    step = 65536
    for i in range(0, len(indices), step):
        block = indices[i : i + step]
        out[i : i + len(block)] = dset[block]
    return out
