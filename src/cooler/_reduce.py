from __future__ import annotations

import json
import os.path as op
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

from ._aggregate import TargetAggregator
from ._logging import get_logger
from ._targets import TargetBins, check_level_name, read_source_bins
from ._typing import MapFunctor
from ._version import __format_version_mcool__
from .api import Cooler
from .create import ContactBinner, create
from .create._ingest import exact_column_sum, groupby_agg_exact
from .create._writer import _COMBINE_AGGS, _check_fits_dtype, _value_aggs
from .parallel import _ProcessPool, get_lock
from .util import GenomeSegmentation, parse_cooler_uri

__all__ = ["coarsen_cooler", "merge_coolers", "zoomify_cooler"]


logger = get_logger("cooler.reduce")

HIGLASS_TILE_DIM = 256


def merge_breakpoints(
    indexes: list[np.ndarray | h5py.Dataset], bufsize: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given ``k`` bin1_offset indexes, determine how to partition the data from
    ``k`` corresponding pixel tables for a k-way merge.

    The paritition is a subsequence of bin1 IDs, defining the bounds of chunks
    of data that will be loaded into memory from each table in a single "epoch"
    of merging data. The bounds are calculated such that no single epoch will
    load more than ``bufsize`` records into memory.

    However, the ``bufsize`` condition is not guaranteed (see Notes): check
    ``cum_nrecords``.

    .. note::
        Kept for backward compatibility. :class:`CoolerMerger` no longer
        partitions merges along whole bin1 rows: it streams bounded slices of
        the pixel tables, so the buffer limit always holds.

    Parameters
    ----------
    indexes : sequence of 1D array-like of equal length
        Offset arrays that map bin1 IDs to their offset locations in a
        corresponding pixel table.
    bufsize : int
        Maximum number of pixel records loaded into memory in a single merge
        epoch.

    Returns
    -------
    bin1_partition : 1D array
        Bin1 IDs defining where to partition all the tables for merging.
    cum_nrecords : 1D array
        Cumulative number of records (from all pixel tables combined) that will
        be processed at each epoch.

    Notes
    -----
    The one exception to the post-condition is when a single bin1 increment in
    a table contains more than ``bufsize`` records.
    """
    # This is a "virtual" cumulative index if all the tables were concatenated
    # and sorted but no pixel records were aggregated. It helps us track how
    # many records would be processed at each merge epoch.
    # NOTE: We sum these incrementally in case the indexes are lazy to avoid
    # loading all indexes into memory at once.
    combined_index = np.zeros(indexes[0].shape, dtype=np.int64)
    for i in range(len(indexes)):
        combined_index += indexes[i]
    combined_start = 0
    combined_nnz = combined_index[-1]

    bin1_partition = [0]
    cum_nrecords = [0]
    lo = 0
    while True:
        # Find the last bin1 ID whose records still fit in the buffer
        target = min(combined_start + bufsize, combined_nnz)
        hi = bisect_right(combined_index, target, lo=lo) - 1
        if combined_index[hi] <= combined_start:
            # A single bin1 has more than bufsize records: take it anyway
            hi = bisect_right(combined_index, combined_start, lo=lo)

        hi = min(hi, len(combined_index) - 1)
        bin1_partition.append(hi)
        cum_nrecords.append(combined_index[hi])

        if combined_index[hi] == combined_nnz:
            break

        lo = hi
        combined_start = combined_index[hi]

    bin1_partition = np.array(bin1_partition)
    cum_nrecords = np.array(cum_nrecords)
    return bin1_partition, cum_nrecords


# Aggregations of value columns that give the same result however the pixels
# are grouped: in one step or level by level when coarsening, and input by
# input when merging
_AGGS = _COMBINE_AGGS


def _chrom_catalog(clr: Cooler) -> tuple[list[str], np.ndarray]:
    """A cooler's ordered chromosome names and lengths, as values."""
    sizes = clr.chromsizes
    return [str(x) for x in sizes.index], np.asarray(sizes)


def _bin_axis(clr: Cooler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cooler's bin table as value arrays: chrom names, starts, ends."""
    bins = clr.bins()[["chrom", "start", "end"]][:]
    chrom = bins["chrom"]
    if chrom.dtype.kind in "iu":
        # Raw integer codes (stored without an HDF5 enum): map them back to
        # names through the chromosome catalog.
        names = np.asarray(list(clr.chromnames), dtype=object)[chrom.to_numpy()]
    else:
        names = chrom.astype(str).to_numpy()
    return names, bins["start"].to_numpy(), bins["end"].to_numpy()


def check_same_axes(coolers: list[Cooler]) -> None:
    """
    Verify that all the coolers share the same axis: the chromosome catalog
    (names in order, and lengths, including chromosomes that have no bins)
    and every row of the bin table. Integer storage width is ignored: the
    same coordinates stored as int32 or int64 compare equal.
    """
    ref = coolers[0]
    ref_names, ref_lengths = _chrom_catalog(ref)
    ref_bins = _bin_axis(ref)
    for clr in coolers[1:]:
        names, lengths = _chrom_catalog(clr)
        if names != ref_names:
            raise ValueError(
                "Coolers must have the same chromosomes in the same order: "
                f"'{clr.filename}' differs from '{ref.filename}'."
            )
        if not np.array_equal(lengths, ref_lengths):
            raise ValueError(
                "Coolers must have the same chromosome lengths: "
                f"'{clr.filename}' differs from '{ref.filename}'."
            )
        bins = _bin_axis(clr)
        if len(bins[0]) != len(ref_bins[0]) or not all(
            np.array_equal(stored, ref) for stored, ref in zip(bins, ref_bins)
        ):
            raise ValueError(
                "Coolers must have the same bin table: "
                f"'{clr.filename}' differs from '{ref.filename}'."
            )


def _normalize_int(arr: np.ndarray) -> np.ndarray:
    """
    Canonicalize an integer array for concatenation: int64 when the values
    fit, else an object array of exact Python ints. This prevents numpy from
    silently coercing mixed int64/uint64 data to lossy float64.
    """
    arr = np.asarray(arr)
    kind = arr.dtype.kind
    if kind == "i":
        return arr if arr.dtype == np.int64 else arr.astype(np.int64)
    if kind == "u":
        if not len(arr) or arr.max() <= np.iinfo(np.int64).max:
            return arr.astype(np.int64)
        return arr.astype(object)
    return arr


def _scalar_row(value: Any) -> np.ndarray:
    """A 1-row array for an aggregated value, kept exact at any magnitude."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = int(value)
        if np.iinfo(np.int64).min <= value <= np.iinfo(np.int64).max:
            return np.array([value], dtype=np.int64)
        return np.array([value], dtype=object)
    return np.asarray([value])


class _PixelStream:
    """
    A sequential cursor over one cooler's pixel table.

    Holds at most one loaded slice at a time; ``load`` replaces the buffer,
    so the number of buffered (loaded but unconsumed) pixels is always
    known and bounded by the caller.
    """

    __slots__ = ("buf", "clr", "fields", "i", "nnz", "pos")

    def __init__(self, clr: Cooler, fields: list[str]):
        self.clr = clr
        self.fields = fields
        self.nnz = clr.info["nnz"]
        self.pos = 0  # next row to load from disk
        self.buf: dict[str, np.ndarray] | None = None
        self.i = 0  # next unconsumed row within the buffer

    @property
    def eof(self) -> bool:
        return self.pos >= self.nnz

    @property
    def buffered(self) -> int:
        return 0 if self.buf is None else len(self.buf["bin1_id"]) - self.i

    def load(self, n: int) -> int:
        """Load up to ``n`` more rows; returns how many were loaded."""
        hi = min(self.pos + n, self.nnz)
        if hi <= self.pos:
            return 0
        df = self.clr.pixels()[self.fields][self.pos : hi]
        self.buf = {k: _normalize_int(v.to_numpy()) for k, v in df.items()}
        self.i = 0
        n = hi - self.pos
        self.pos = hi
        return n

    @property
    def last_key(self) -> tuple[int, int]:
        return (self.buf["bin1_id"][-1], self.buf["bin2_id"][-1])

    def _take(self, n: int) -> dict[str, np.ndarray]:
        out = {col: values[self.i : self.i + n] for col, values in self.buf.items()}
        self.i += n
        return out

    def take_lt(self, boundary: tuple[int, int] | None) -> dict[str, np.ndarray]:
        """Consume the buffered prefix with keys strictly below ``boundary``."""
        b1 = self.buf["bin1_id"][self.i :]
        b2 = self.buf["bin2_id"][self.i :]
        if boundary is None:
            n = len(b1)
        else:
            # The buffer is sorted, so this mask is a prefix
            mask = (b1 < boundary[0]) | ((b1 == boundary[0]) & (b2 < boundary[1]))
            n = int(mask.sum())
        return self._take(n)

    def take_eq(self, key: tuple[int, int]) -> dict[str, np.ndarray]:
        """Consume the buffered prefix of records with this exact key."""
        b1 = self.buf["bin1_id"][self.i :]
        b2 = self.buf["bin2_id"][self.i :]
        mask = (b1 == key[0]) & (b2 == key[1])
        return self._take(int(mask.sum()))


class CoolerMerger(ContactBinner):
    """
    Implementation of cooler merging.

    Streams the inputs in a bounded k-way merge: at most ``mergebuf`` source
    pixels are held in memory at any time, however wide a matrix row is, and
    output is produced in batches of at most ``mergebuf`` pixels. Integer
    sums are exact at any magnitude; a total that doesn't fit the output
    column's dtype raises :class:`IntegerOverflowError` rather than wrapping.

    Attributes
    ----------
    read_sizes : list of int
        Diagnostic: the size of every source slice loaded, in order.
    max_buffered : int
        Diagnostic: the peak number of source pixels held at once.

    """

    def __init__(
        self,
        coolers: list[Cooler],
        mergebuf: int,
        columns: list[str] | None = None,
        agg: dict[str, Any] | None = None,
        dtypes: dict[str, Any] | None = None,
    ):
        self.coolers = list(coolers)
        if not self.coolers:
            raise ValueError("No coolers to merge.")
        self.mergebuf = int(mergebuf)
        if self.mergebuf < 1:
            raise ValueError(f"mergebuf must be a positive integer, not {mergebuf}.")
        self.columns = ["count"] if columns is None else list(columns)
        self.agg = _value_aggs(self.columns, agg)
        if dtypes is None:
            # Check totals against the common dtype of the inputs
            dtypes = {}
            for col in self.columns:
                dtypes[col] = np.result_type(
                    *[clr.pixels().dtypes[col] for clr in self.coolers]
                )
        self.dtypes = {col: np.dtype(dtypes[col]) for col in self.columns}

        # All inputs must share the same axis
        check_same_axes(self.coolers)

        # The merge needs one buffer slot per non-empty input
        n_live = sum(1 for clr in self.coolers if clr.info["nnz"] > 0)
        if n_live > self.mergebuf:
            raise ValueError(
                f"Cannot merge {n_live} non-empty inputs with "
                f"mergebuf={self.mergebuf}: the buffer must hold at least one "
                "pixel per input. Increase mergebuf."
            )

        self._fields = ["bin1_id", "bin2_id", *self.columns]
        self.read_sizes: list[int] = []
        self.max_buffered = 0

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        streams = [
            _PixelStream(clr, self._fields)
            for clr in self.coolers
            if clr.info["nnz"] > 0
        ]
        if not streams:
            return
        mergebuf = self.mergebuf
        key_cols = ["bin1_id", "bin2_id"]
        out_batches: list[dict[str, np.ndarray]] = []
        out_n = 0
        pend_key: tuple[int, int] | None = None
        pend_vals: dict[str, Any] = {}

        def flush():
            nonlocal out_batches, out_n
            if not out_n:
                return
            chunk = self._finalize(out_batches)
            out_batches = []
            out_n = 0
            n = len(chunk["bin1_id"])
            for lo in range(0, n, mergebuf):
                yield {col: values[lo : lo + mergebuf] for col, values in chunk.items()}

        while True:
            # Refill empty buffers. Invariant: the number of pixels loaded
            # but not yet consumed never exceeds mergebuf.
            empties = [s for s in streams if s.buffered == 0 and not s.eof]
            if empties:
                avail = mergebuf - sum(s.buffered for s in streams)
                # Distribute the available slots as evenly as possible
                base, extra = divmod(avail, len(empties))
                for j, s in enumerate(empties):
                    self.read_sizes.append(s.load(base + (j < extra)))
                self.max_buffered = max(
                    self.max_buffered, sum(s.buffered for s in streams)
                )

            active = [s for s in streams if s.buffered > 0]
            if not active:
                if pend_key is not None:
                    out_batches.append(self._pend_chunk(pend_key, pend_vals))
                    out_n += 1
                yield from flush()
                break

            # Keys strictly below the boundary are complete in every input:
            # each non-EOF input's unread records start at its last buffered
            # key or beyond, and EOF inputs have nothing left on disk.
            boundary = None
            for s in active:
                if s.eof:
                    continue
                key = s.last_key
                if boundary is None or key < boundary:
                    boundary = key

            if pend_key is not None and (boundary is None or pend_key < boundary):
                # Every input has moved past the pending key: emit its total
                out_batches.append(self._pend_chunk(pend_key, pend_vals))
                out_n += 1
                pend_key, pend_vals = None, {}
                if out_n >= mergebuf:
                    yield from flush()

            # Combine all buffered records below the boundary in one
            # vectorized, exactly-summed aggregation
            slices = [s.take_lt(boundary) for s in active]
            slices = [sl for sl in slices if len(sl["bin1_id"])]
            if slices:
                data = {
                    col: np.concatenate([sl[col] for sl in slices])
                    for col in [*key_cols, *self.columns]
                }
                combined = groupby_agg_exact(pd.DataFrame(data), key_cols, self.agg)
                out_batches.append(
                    {col: _normalize_int(combined[col].to_numpy()) for col in data}
                )
                out_n += len(combined)
                if out_n >= mergebuf:
                    yield from flush()

            if boundary is not None:
                # The boundary key itself may continue on disk: accumulate it
                # as the single pending key and free its buffer space
                if pend_key is None:
                    pend_vals = {col: None for col in self.columns}
                for s in active:
                    sl = s.take_eq(boundary)
                    if not len(sl["bin1_id"]):
                        continue
                    for col, f in self.agg.items():
                        v = sl[col]
                        if f == "sum":
                            v = exact_column_sum(v)
                        elif f == "min":
                            v = v.min()
                        else:
                            v = v.max()
                        cur = pend_vals[col]
                        if cur is None:
                            pend_vals[col] = v
                        elif f == "sum":
                            pend_vals[col] = cur + v
                        elif f == "min":
                            pend_vals[col] = min(cur, v)
                        else:
                            pend_vals[col] = max(cur, v)
                pend_key = boundary

    def _pend_chunk(self, key, vals) -> dict[str, np.ndarray]:
        chunk = {
            "bin1_id": np.array([key[0]], dtype=np.int64),
            "bin2_id": np.array([key[1]], dtype=np.int64),
        }
        for col in self.columns:
            chunk[col] = _scalar_row(vals[col])
        return chunk

    def _finalize(self, batches) -> dict[str, np.ndarray]:
        """Concatenate queued rows, check integer ranges, cast to dtypes."""
        out = {}
        for col in ["bin1_id", "bin2_id", *self.columns]:
            arr = np.concatenate([b[col] for b in batches])
            if col in ("bin1_id", "bin2_id"):
                out[col] = arr.astype(np.int64, copy=False)
                continue
            dtype = self.dtypes[col]
            if arr.dtype.kind == "O" and not len(arr):
                arr = np.empty(0, dtype=dtype)
            else:
                _check_fits_dtype(col, arr, dtype)
                arr = np.asarray(arr, dtype=dtype)
            out[col] = arr
        return out


def merge_coolers(
    output_uri: str,
    input_uris: list[str],
    mergebuf: int,
    columns: list[str] | None = None,
    dtypes: dict[str, Any] | None = None,
    agg: dict[str, Any] | None = None,
    **kwargs,
) -> None:
    """
    Merge multiple coolers with identical axes.

    The merged cooler is stored at ``output_uri``.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    output_uri : str
        Output cooler file path or URI.
    input_uris : list of str
        List of input file path or URIs of coolers to combine.
    mergebuf : int
        Maximum number of source pixels held in memory at a time. This is a
        hard bound: wide matrix rows are split across read batches, and the
        output is produced in batches of at most this size. It must be at
        least the number of non-empty inputs.
    columns : list of str, optional
        Specify which pixel value columns to include in the aggregation.
        Default is to use all available value columns.
    dtypes : dict, optional
        Specific dtypes to use for value columns. Default is to propagate
        the current dtypes of the value columns.
    agg : dict, optional
        How to combine the values of a pixel across the input coolers: a
        mapping of value column names to "sum" (default), "min" or "max". An
        input without the pixel doesn't contribute a value.
    kwargs
        Passed to ``cooler.create``.

    Notes
    -----
    The default output file mode is 'w'. If appending output to an existing
    file, pass `mode='a'`.

    All inputs must share the same axis: the same chromosomes in the same
    order with the same lengths (including chromosomes without bins), and
    the same bin table. This is checked before the output is created; the
    integer width the axis is stored with doesn't matter. Inputs with a
    sparse bin layout (``bin-layout: sparse``) merge into a sparse output
    that keeps the full chromosome catalog and the gaps.

    Integer sums are exact at any magnitude. A total that doesn't fit the
    output column's dtype raises ``IntegerOverflowError`` (an
    ``OverflowError``) instead of wrapping; request a wider dtype, e.g.
    ``dtypes={"count": "uint64"}``, for large counts. The ``sum`` metadata
    of the output is always exact, even when it exceeds 64 bits. A failure
    never leaves a partial cooler behind: an existing group at the
    destination and any sibling groups are preserved.

    See also
    --------
    cooler.coarsen_cooler
    cooler.zoomify_cooler

    """
    # TODO: combine metadata from inputs
    logger.info(f"Merging {len(input_uris)} coolers into {output_uri}")
    for uri in input_uris:
        logger.debug(f"Input: {uri}")

    clrs = [Cooler(path) for path in input_uris]
    if not clrs:
        raise ValueError("No input coolers were given.")

    is_symm = [clr.storage_mode == "symmetric-upper" for clr in clrs]
    if all(is_symm):
        symmetric_upper = True
    elif not any(is_symm):
        symmetric_upper = False
    else:
        raise ValueError("Cannot merge symmetric and non-symmetric coolers.")

    if columns is None:
        columns = ["count"]

    dtype_map = defaultdict(list)
    for clr in clrs:
        pixel_dtypes = clr.pixels().dtypes
        for col in columns:
            if col not in pixel_dtypes:
                raise ValueError(
                    f"Pixel value column '{col}' not found in input '{clr.filename}'."
                )
            else:
                dtype_map[col].append(pixel_dtypes[col])

    dtypes = {} if dtypes is None else dict(dtypes)
    for col in columns:
        if col not in dtypes:
            # Find a common dtype to accomodate all the inputs being merged.
            dtypes[col] = np.result_type(*dtype_map[col])

    # A sparse bin layout (an extension with gaps in the bin table) is kept
    # sparse; ordinary tiling layouts are delivered as ordinary layouts.
    sparse = any(clr.info.get("bin-layout") == "sparse" for clr in clrs)

    bins = clrs[0].bins()[["chrom", "start", "end"]][:]
    assembly = clrs[0].info.get("genome-assembly", None)
    # Validates the axes before the output is created
    iterator = CoolerMerger(
        clrs,
        mergebuf=mergebuf,
        columns=columns,
        agg=agg,
        dtypes={col: dtypes[col] for col in columns},
    )

    if sparse:
        kwargs.setdefault("sparse", True)
        kwargs.setdefault("chromsizes", clrs[0].chromsizes)
        binsize = None
    else:
        binsize = clrs[0].binsize

    create(
        output_uri,
        bins,
        iterator,
        columns=columns,
        dtypes=dtypes,
        assembly=assembly,
        symmetric_upper=symmetric_upper,
        binsize=binsize,
        **kwargs,
    )


def _greedy_prune_partition(edges: np.ndarray, maxlen: int) -> np.ndarray:
    """Given an integer interval partition ``edges`` from 0..nnz, prune the
    edges to make the new subintervals roughly ``maxlen`` in length.
    """
    edges = np.asarray(edges)
    assert len(edges) >= 2 and edges[0] == 0
    cumlen = np.r_[0, np.cumsum(np.diff(edges))]
    cuts = [maxlen * i for i in range(int(np.ceil(cumlen[-1] / maxlen)))]
    cuts.append(cumlen[-1])
    idx = np.unique(np.searchsorted(cumlen, cuts))
    return edges[idx]


def geomprog(start: int, mul: int) -> Iterator[int]:
    """
    Generate a geometric progression of integers.

    Beginning with integer ``start``, yield an unbounded geometric progression
    with integer ratio ``mul``.

    """
    start, mul = int(start), int(mul)
    yield start
    while True:
        start *= mul
        yield start


def niceprog(start: int) -> Iterator[int]:
    """
    Generate a nice progression of integers.

    Beginning with integer ``start``, yield a sequence of "nicely" spaced
    integers: an unbounded geometric progression with ratio 10, interspersed
    with steps of ratios 2 and 5.

    """
    start = int(start)
    yield start
    while True:
        for mul in (2, 5, 10):
            yield start * mul
        start *= 10


def preferred_sequence(
    start: int, stop: int, style: Literal["binary", "nice"] = "nice"
) -> list[int]:
    """
    Return a sequence of integers with a "preferred" stepping pattern.

    Parameters
    ----------
    start : int
        Starting value in the progression.
    stop : int
        Upper bound of progression, inclusive. Values will not exceed this.
    style : {'nice', 'binary'}
        Style of progression. 'nice' gives geometric steps of 10 with 2 and 5
        in between. 'binary' gives geometric steps of 2.

    Returns
    ------
    list of int

    Examples
    --------
    For certain values of `start` (n * 10^i), nice stepping produces familiar
    "preferred" sequences [1]_:

    Note denominations in Dollars (1-2-5)

        >>> preferred_sequence(1, 100, 'nice')
        [1, 2, 5, 10, 20, 50, 100]


    Coin denominations in Cents

        >>> preferred_sequence(5, 100, 'nice')
        [5, 10, 25, 50, 100]

    .. [1] https://en.wikipedia.org/wiki/Preferred_number#1-2-5_series

    """
    if start > stop:
        return []

    if style == "binary":
        gen = geomprog(start, 2)
    elif style == "nice":
        gen = niceprog(start)
    else:
        raise ValueError(f"Expected style value of 'binary' or 'nice'; got '{style}'.")

    seq = [next(gen)]
    while True:
        n = next(gen)
        if n > stop:
            break
        seq.append(n)

    return seq


def get_multiplier_sequence(
    resolutions: list[int], bases: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From a set of target resolutions and one or more base resolutions
    deduce the most efficient sequence of integer multiple aggregations
    to satisfy all targets starting from the base resolution(s).

    Parameters
    ----------
    resolutions: sequence of int
        The target resolutions
    bases: sequence of int, optional
        The base resolutions for which data already exists.
        If not provided, the smallest resolution is assumed to be the base.

    Returns
    -------
    resn: 1D array
        Resolutions, sorted in ascending order.
    pred: 1D array
        Index of the predecessor resolution in `resn`. A value of -1 implies
        that the resolution is a base resolution.
    mult: 1D array
        Multiplier to go from predecessor to target resolution.

    """
    if bases is None:
        # assume the base resolution is the smallest one
        bases = {min(resolutions)}
    else:
        bases = set(bases)

    resn = np.array(sorted(bases.union(resolutions)))
    pred = -np.ones(len(resn), dtype=int)
    mult = -np.ones(len(resn), dtype=int)

    for i, target in list(enumerate(resn))[::-1]:
        if target in bases:
            continue  # data for base resolutions already exists
        p = i - 1
        while p >= 0:
            if target % resn[p] == 0:
                pred[i] = p
                mult[i] = target // resn[p]
                break
            else:
                p -= 1

    for i, p in enumerate(pred):
        if p == -1 and resn[i] not in bases:
            raise ValueError(
                f"Resolution {resn[i]} cannot be derived from "
                f"the base resolutions: {bases}."
            )

    return resn, pred, mult


class CoolerCoarsener(ContactBinner):
    """
    Implementation of cooler coarsening.

    """

    def __init__(
        self,
        source_uri: str,
        factor: int,
        chunksize: int,
        columns: list[str],
        agg: dict[str, Any] | None,
        batchsize: int,
        map: MapFunctor = map,
    ):
        self._map = map

        self.source_uri = source_uri
        self.batchsize = batchsize

        assert isinstance(factor, int) and factor > 1
        self.factor = factor
        self.chunksize = int(chunksize)

        self.index_columns = ["bin1_id", "bin2_id"]
        self.value_columns = list(columns)
        self.columns = self.index_columns + self.value_columns
        self.agg = _value_aggs(self.value_columns, agg)

        clr = Cooler(source_uri)
        chromsizes = clr.chromsizes

        # Info for the old bin segmentation
        self.old_binsize = clr.binsize
        self.old_chrom_offset = clr._load_dset("indexes/chrom_offset")
        self.old_bin1_offset = clr._load_dset("indexes/bin1_offset")

        # Calculate the new bin segmentation
        if self.old_binsize is None:
            self.new_binsize = None
        else:
            self.new_binsize = self.old_binsize * factor
        old_bins = clr.bins()[["chrom", "start", "end"]][:]
        self.new_bins = self.coarsen_bins(old_bins, chromsizes, factor)
        self.gs = GenomeSegmentation(chromsizes, self.new_bins)

        # Pre-compute the partition of bin1 offsets that groups the pixels into
        # coarsened bins along the i-axis. Then remove some of the internal
        # edges of this partition to make bigger groups of pixels. This way
        # we ensure that none of the original groups gets split.
        edges = []
        for i in self.gs.idmap.to_numpy():
            # Respect chrom1 boundaries
            c0 = self.old_chrom_offset[i]
            c1 = self.old_chrom_offset[i + 1]
            edges.extend(self.old_bin1_offset[c0:c1:factor])
        edges.append(self.old_bin1_offset[-1])
        self.edges = _greedy_prune_partition(edges, self.chunksize)

    @staticmethod
    def coarsen_bins(
        old_bins: pd.DataFrame, chromsizes: pd.Series, factor: int
    ) -> pd.DataFrame:
        def _each(group):
            out = group[["chrom", "start"]].copy().iloc[::factor]
            end = group["end"].iloc[factor - 1 :: factor].values
            if len(end) < len(out):
                end = np.r_[end, chromsizes[group.name]]
            out["end"] = end
            return out

        return (
            old_bins.groupby("chrom", observed=True)[["chrom", "start", "end"]]
            .apply(_each)
            .reset_index(drop=True)
        )

    def aggregate(self, span: tuple[int, int]) -> pd.DataFrame:
        lo, hi = span

        clr = Cooler(self.source_uri)
        # convert_enum=False returns chroms as raw ints
        table = clr.pixels(join=True, convert_enum=False)[self.columns]
        chunk = table[lo:hi]
        logger.debug(f"Coarsening pixels {lo:,}-{hi:,}")

        # use the "start" point as anchor for re-binning
        binsize = self.gs.binsize
        chrom_binoffset = self.gs.chrom_binoffset
        chrom_abspos = self.gs.chrom_abspos
        start_abspos = self.gs.start_abspos

        chrom_id1 = chunk["chrom1"].values
        chrom_id2 = chunk["chrom2"].values
        start1 = chunk["start1"].values
        start2 = chunk["start2"].values
        if binsize is None:
            abs_start1 = chrom_abspos[chrom_id1] + start1
            abs_start2 = chrom_abspos[chrom_id2] + start2
            chunk["bin1_id"] = (
                np.searchsorted(start_abspos, abs_start1, side="right") - 1
            )
            chunk["bin2_id"] = (
                np.searchsorted(start_abspos, abs_start2, side="right") - 1
            )
        else:
            rel_bin1 = np.floor(start1 / binsize).astype(int)
            rel_bin2 = np.floor(start2 / binsize).astype(int)
            chunk["bin1_id"] = chrom_binoffset[chrom_id1] + rel_bin1
            chunk["bin2_id"] = chrom_binoffset[chrom_id2] + rel_bin2

        return (
            chunk.groupby(self.index_columns, sort=True)
            .aggregate(self.agg)
            .reset_index()
        )

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        # Distribute batches of `batchsize` pixel spans at once.
        batchsize = self.batchsize
        spans = list(zip(self.edges[:-1], self.edges[1:]))
        for i in range(0, len(spans), batchsize):
            try:
                if batchsize > 1:
                    get_lock().acquire()
                # Collect the batch before releasing the lock, so workers
                # aren't reading while the output is written
                results = list(self._map(self.aggregate, spans[i : i + batchsize]))
            finally:
                if batchsize > 1:
                    get_lock().release()
            for df in results:
                yield {k: v.values for k, v in df.items()}


class CoolerTargetCoarsener(ContactBinner):
    """
    Implementation of coarsening onto an explicit target bin table.

    Yields target pixels sorted by (bin1_id, bin2_id) with no duplicates,
    aggregating only source pixels whose both bins are retained.
    """

    def __init__(
        self,
        source_uri: str,
        target: TargetBins,
        chunksize: int,
        columns: list[str],
        agg: dict[str, Any] | None,
        nproc: int = 1,
    ):
        self.source_uri = source_uri
        self.target = target
        self.chunksize = int(chunksize)
        self.columns = list(columns)
        self.aggregator = TargetAggregator(
            source_uri,
            target,
            columns=self.columns,
            agg=_value_aggs(self.columns, agg),
            chunksize=self.chunksize,
            nproc=nproc,
        )

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        yield from self.aggregator


def coarsen_cooler(
    base_uri: str,
    output_uri: str,
    factor: int | None,
    chunksize: int,
    nproc: int = 1,
    columns: list[str] | None = None,
    dtypes: dict[str, Any] | None = None,
    agg: dict[str, Any] | None = None,
    target_bins: pd.DataFrame | None = None,
    **kwargs,
) -> None:
    """
    Coarsen a cooler to a lower resolution by an integer factor *k*.

    This is done by pooling *k*-by-*k* neighborhoods of pixels and aggregating.
    Each chromosomal block is coarsened individually. Result is a coarsened
    cooler stored at ``output_uri``.

    Alternatively, pass ``target_bins`` to pool an arbitrary, possibly sparse
    set of intervals of whole source bins: each target bin becomes an output
    bin, in the row order of the table, and only source pixels whose both
    bins are retained contribute.  The output is marked with the ``bin-layout``
    attribute ``"sparse"``, an extension to the cooler format that other
    implementations may not interpret; see Notes.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    base_uri : str
        Input cooler file path or URI.
    output_uri : str
        Input cooler file path or URI.
    factor : int or None
        Coarsening factor. Must be ``None`` when ``target_bins`` is given.
    chunksize : int
        Number of source pixels processed at a time per worker. It bounds the
        pixels read at once; it is not widened to cover a whole target bin.
    nproc : int, optional
        Number of workers for batch processing of pixels. Default is 1,
        i.e. no process pool.
    columns : list of str, optional
        Specify which pixel value columns to include in the aggregation.
        Default is to use all available value columns.
    dtypes : dict, optional
        Specific dtypes to use for value columns. Default is to propagate
        the current dtypes of the value columns.
    agg : dict, optional
        How to combine the pixels of each *k*-by-*k* block (or target bin): a
        mapping of value column names to "sum" (default), "min" or "max".
        Pixels that aren't stored don't contribute a value.
    target_bins : pandas.DataFrame, optional
        An explicit target bin table with columns ``chrom``, ``start`` and
        ``end``, sorted by input chromosome catalog order then start. Each
        interval must be non-empty, must not overlap another, and must be
        exactly a run of consecutive source bins on one chromosome: target
        bins can't split a source bin or cover coordinates the source lacks.
        Source bins and whole chromosomes may be omitted; a zero-row table is
        allowed. The output keeps the source chromosome catalog and lengths.
        The table's row order defines the output bin IDs; every retained
        source bin belongs to exactly one target bin. Source pixel value
        columns such as ``weight`` in the bin table are not copied.
    kwargs
        Passed to ``cooler.create``.

    Notes
    -----
    The sparse target layout is an extension to the cooler format. Output
    files carry ``bin-layout: sparse`` and remain readable by this version of
    cooler, but other cooler implementations may not interpret them
    correctly.

    Only source pixels with both bins retained take part; the rest are
    dropped. ``count`` stays an exact integer. A pixel landing on a target
    pair is counted once in symmetric-upper storage, including contacts that
    were off the diagonal but merge onto it; square storage keeps direction
    and doesn't add transposes or combine reverse records.

    See also
    --------
    cooler.zoomify_cooler
    cooler.merge_coolers

    """
    # TODO: decide whether to default to 'count' or whatever is there besides
    # bin1_id, bin2_id dtypes = dict(clr.pixels().dtypes.drop(['bin1_id', 'bin2_id']))
    if target_bins is not None and factor is not None:
        raise ValueError(
            "Pass either an integer factor or an explicit target_bins table, not both."
        )
    if target_bins is None and factor is None:
        raise ValueError("Either factor or target_bins must be given.")

    clr = Cooler(base_uri)

    if factor is not None:
        factor = int(factor)
        if factor < 2:
            raise ValueError(f"The coarsening factor must be at least 2, not {factor}.")

    if columns is None:
        columns = ["count"]

    dtypes = {} if dtypes is None else dict(dtypes)
    explicit_dtypes = set(dtypes)

    input_dtypes = clr.pixels().dtypes
    for col in columns:
        if col not in input_dtypes:
            raise ValueError(
                f"Pixel value column '{col}' not found in input '{clr.filename}'."
            )
        else:
            dtypes.setdefault(col, input_dtypes[col])

    # Validate the target table before anything is written
    target = None
    if target_bins is not None:
        target = TargetBins.from_frame(
            target_bins,
            clr.chromsizes,
            read_source_bins(clr),
        )
        # Keep summed integers exact. Unless the caller asked for a specific
        # dtype, pool integer `sum` columns into int64 and float columns into
        # float64 (which also avoids overflowing the source's int32 counts).
        aggs_by_col = _value_aggs(columns, agg)
        for col in columns:
            if col in explicit_dtypes:
                continue
            kind = np.dtype(input_dtypes[col]).kind
            if kind in "iub":
                dtypes[col] = np.dtype(
                    np.int64 if aggs_by_col.get(col, "sum") == "sum" else kind
                )
            elif kind == "f" and aggs_by_col.get(col, "sum") == "sum":
                dtypes[col] = np.dtype(np.float64)

    if target is not None:
        # The target coarsener manages its own workers (and their reader
        # state); no second pool is created here.
        iterator = CoolerTargetCoarsener(
            base_uri,
            target,
            chunksize,
            columns=columns,
            agg=agg,
            nproc=nproc,
        )
        new_bins = target.bins
        create_kwargs = {"chromsizes": target.chromsizes, "sparse": True}
        if nproc > 1:
            kwargs.setdefault("lock", get_lock())
        kwargs.setdefault("mode", "a")
        kwargs.setdefault("assembly", clr.info.get("genome-assembly"))
        kwargs.setdefault("metadata", clr.info.get("metadata"))
        create(
            output_uri,
            new_bins,
            iterator,
            columns=columns,
            dtypes=dtypes,
            symmetric_upper=clr.storage_mode == "symmetric-upper",
            binsize=None,
            **create_kwargs,
            **kwargs,
        )
        return

    if nproc > 1:
        kwargs.setdefault("lock", get_lock())
        pool = _ProcessPool(nproc)
    try:
        iterator = CoolerCoarsener(
            base_uri,
            factor,
            chunksize,
            columns=columns,
            agg=agg,
            batchsize=nproc,
            map=pool.map if nproc > 1 else map,
        )
        new_bins = iterator.new_bins

        kwargs.setdefault("mode", "a")
        kwargs.setdefault("assembly", clr.info.get("genome-assembly"))
        kwargs.setdefault("metadata", clr.info.get("metadata"))

        create(
            output_uri,
            new_bins,
            iterator,
            columns=columns,
            dtypes=dtypes,
            symmetric_upper=clr.storage_mode == "symmetric-upper",
            binsize=iterator.new_binsize,
            **kwargs,
        )

    finally:
        if nproc > 1:
            pool.close()


def zoomify_cooler(
    base_uris: str | list[str],
    outfile: str,
    resolutions: list[int] | None,
    chunksize: int,
    nproc: int = 1,
    columns: list[str] | None = None,
    dtypes: dict[str, Any] | None = None,
    agg: dict[str, Any] | None = None,
    bin_levels: dict[str, pd.DataFrame] | None = None,
    **kwargs,
) -> None:
    """
    Generate multiple cooler resolutions by recursive coarsening.

    Result is a "zoomified" or "multires" cool file stored at ``outfile``
    using the MCOOL v2 layout, where coolers are stored under a hierarchy of
    the form ``resolutions/<r>`` for each resolution ``r``.

    Alternatively, pass ``bin_levels`` to store explicitly binned, possibly
    sparse, levels under ``/bin_levels/<name>`` instead.  This is an extension
    to the MCOOL layout: the levels keep their names rather than pretending to
    be integer resolutions, and ``resolutions`` must be empty.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    base_uris : str or sequence of str
        One or more cooler URIs to use as "base resolutions" for aggregation.
    outfile : str
        Output multires cooler (mcool) file path.
    resolutions : list of int
        A list of target resolutions to generate. Mutually exclusive with
        ``bin_levels``. May be empty only when ``bin_levels`` is given.
    chunksize : int
        Number of pixels processed at a time per worker.
    nproc : int, optional
        Number of workers for batch processing of pixels. Default is 1,
        i.e. no process pool.
    columns : list of str, optional
        Specify which pixel value columns to include in the aggregation.
        Default is to use only the column named 'count' if it exists.
    dtypes : dict, optional
        Specific dtypes to use for value columns. Default is to propagate
        the current dtypes of the value columns.
    agg : dict, optional
        How to combine the pixels of each block when coarsening: a mapping of
        value column names to "sum" (default), "min" or "max". These give the
        same result whether a resolution is coarsened from the base or from
        the previous resolution.
    bin_levels : dict of str to pandas.DataFrame, optional
        Explicit target bin tables, keyed by level name. Keys must start with
        an ASCII letter or digit and contain only ASCII letters, digits,
        underscores and hyphens. Each table follows the same rules as
        ``coarsen_cooler``'s ``target_bins``. Each level is stored under
        ``/bin_levels/<name>``; levels may be unrelated to one another or be
        coarser merges of another level. Every table is validated against the
        original input first.
    kwargs
        Passed to ``cooler.create``.

    Notes
    -----
    The explicit-level mode accepts a single base URI for now. A level may be
    aggregated directly from the base, or, when it can be built purely by
    pooling whole target bins of another level, from that level; in either
    case the result is identical to aggregating from the base. A level whose
    needed bins or bin boundaries are missing from an intermediate level is
    always aggregated from the base. The insertion order of ``bin_levels``
    doesn't change any level's result.

    See also
    --------
    cooler.coarsen_cooler
    cooler.merge_coolers

    """
    if bin_levels is not None:
        _zoomify_bin_levels(
            base_uris,
            outfile,
            bin_levels,
            resolutions,
            chunksize,
            nproc=nproc,
            columns=columns,
            dtypes=dtypes,
            agg=agg,
            **kwargs,
        )
        return

    # TODO: provide presets? {'pow2', '4dn'}
    if isinstance(base_uris, str):
        base_uris = [base_uris]

    parsed_uris = {}
    base_resolutions = set()
    for input_uri in base_uris:
        infile, ingroup = parse_cooler_uri(input_uri)
        if op.exists(outfile) and op.samefile(infile, outfile):
            # The output file is truncated before the inputs are read
            raise ValueError(
                f"The output file '{outfile}' contains the input '{input_uri}'."
            )
        clr = Cooler(infile, ingroup)
        base_binsize = 1 if clr.binsize is None else clr.binsize
        if base_binsize in parsed_uris:
            raise ValueError(
                "Base coolers must have different resolutions, and at most one "
                f"can have variable-size bins: '{input_uri}'."
            )
        parsed_uris[base_binsize] = (infile, ingroup)
        base_resolutions.add(base_binsize)

    # Determine the sequence of reductions.
    resn, pred, mult = get_multiplier_sequence(resolutions, base_resolutions)

    n_zooms = len(resn)
    n_new = int(np.sum(pred != -1))
    bases = ", ".join(str(b) for b in sorted(base_resolutions))
    logger.info(f"Copying base resolutions ({bases}) and producing {n_new} new ones")

    if columns is None:
        columns = ["count"]

    # Copy base matrices into a new output file
    with h5py.File(outfile, "w"):
        pass
    for base_binsize in sorted(base_resolutions):
        logger.info(f"Copying base resolution {base_binsize}")
        infile, ingroup = parsed_uris[base_binsize]
        with h5py.File(infile, "r") as src, \
             h5py.File(outfile, "r+") as dest:  # fmt: skip
            prefix = f"/resolutions/{base_binsize}"

            src.copy(ingroup + "/chroms", dest, prefix + "/chroms")
            src.copy(ingroup + "/bins", dest, prefix + "/bins")
            for col in ["bin1_id", "bin2_id", *list(columns)]:
                src.copy(
                    ingroup + f"/pixels/{col}",
                    dest,
                    prefix + f"/pixels/{col}",
                )
            src.copy(ingroup + "/indexes", dest, prefix + "/indexes")
            dest[prefix].attrs.update(src[ingroup].attrs)

    # Aggregate
    # Use lock to sync read/write ops on same file
    for i in range(n_zooms):
        if pred[i] == -1:
            continue
        prev_binsize = resn[pred[i]]
        binsize = prev_binsize * mult[i]
        logger.info(f"Aggregating from {prev_binsize} to {binsize}")
        coarsen_cooler(
            outfile + f"::resolutions/{prev_binsize}",
            outfile + f"::resolutions/{binsize}",
            mult[i],
            chunksize,
            nproc=nproc,
            columns=columns,
            dtypes=dtypes,
            agg=agg,
            mode="r+",
            **kwargs,
        )

    with h5py.File(outfile, "r+") as fw:
        fw.attrs.update(
            {"format": "HDF5::MCOOL", "format-version": __format_version_mcool__}
        )


def _zoomify_bin_levels(
    base_uris,
    outfile,
    bin_levels,
    resolutions,
    chunksize,
    nproc=1,
    columns=None,
    dtypes=None,
    agg=None,
    **kwargs,
):
    """Write explicit bin levels under ``/bin_levels/<name>``."""
    if resolutions:
        raise ValueError(
            "Pass either integer resolutions or explicit bin_levels, not both."
        )
    if isinstance(base_uris, str):
        base_uris = [base_uris]
    base_uris = list(base_uris)
    if len(base_uris) != 1:
        raise ValueError(
            "Explicit bin levels support a single base URI for now, not "
            f"{len(base_uris)}."
        )
    if not isinstance(bin_levels, dict):
        raise ValueError(  # noqa: TRY004
            "bin_levels must be a dict mapping level names to target bin DataFrames."
        )
    names = []
    for name in bin_levels:
        check_level_name(name)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("bin_levels has duplicate level names.")

    base_uri = base_uris[0]
    infile, ingroup = parse_cooler_uri(base_uri)
    if op.exists(outfile) and op.samefile(infile, outfile):
        raise ValueError(
            f"The output file '{outfile}' contains the input '{base_uri}'."
        )
    base = Cooler(infile, ingroup)
    if columns is None:
        columns = ["count"]
    input_dtypes = base.pixels().dtypes
    for col in columns:
        if col not in input_dtypes:
            raise ValueError(
                f"Pixel value column '{col}' not found in input '{base.filename}'."
            )
    dtypes = {} if dtypes is None else dict(dtypes)
    # Only the caller's explicit dtypes are forwarded; otherwise each level
    # picks exact accumulation dtypes in coarsen_cooler.

    # Validate every table against the original input before writing anything
    src_bins = read_source_bins(base)
    targets = {
        name: TargetBins.from_frame(bin_levels[name], base.chromsizes, src_bins)
        for name in names
    }

    # Choose a source for each level. A level may be built from another level
    # only if its target bins are exactly whole runs of that level's bins and
    # no needed bin was dropped; then pooling is exact for sum, min and max,
    # and reads fewer pixels. Otherwise it is aggregated from the base.
    #
    # To keep the choice independent of dict insertion order and to avoid
    # cycles, process levels from finest to coarsest: a level can only use an
    # already-written finer level. "Finer" means it has more target bins (a
    # derivable level has at least as many), so sort descending by bin count.
    rank = {
        name: (-targets[name].nbins, -_n_retained(targets[name]), name)
        for name in names
    }
    ordered = sorted(names, key=lambda n: rank[n])
    parent: dict[str, str | None] = {}
    for i, name in enumerate(ordered):
        chosen = None
        best_key = None
        # Prefer the fitting level that stores the fewest pixels: it reads
        # less. That is the coarsest fitting level.
        for candidate in ordered[:i]:
            pooled = targets[name].derives_from(targets[candidate])
            if pooled is None:
                continue
            key = (_level_nnz(base, targets, candidate), rank[candidate])
            if best_key is None or key < best_key:
                best_key = key
                chosen = candidate
        parent[name] = chosen

    logger.info(
        f"Writing {len(names)} explicit bin levels: "
        + ", ".join(
            f"{name} (from {parent[name] if parent[name] else 'base'})"
            for name in names
        )
    )

    with h5py.File(outfile, "w"):
        pass

    for name in ordered:
        source = parent[name]
        spec = dict(
            chunksize=chunksize,
            nproc=nproc,
            columns=columns,
            dtypes=dtypes,
            agg=agg,
            mode="a",
            **kwargs,
        )
        spec.setdefault("lock", get_lock())
        if source is None:
            pool_target = targets[name]
            source_uri = base_uri
        else:
            pool_target = targets[name].derives_from(targets[source])
            source_uri = outfile + "::/bin_levels/" + source
        coarsen_cooler(
            source_uri,
            outfile + "::/bin_levels/" + name,
            None,
            target_bins=pool_target.bins,
            **spec,
        )

    with h5py.File(outfile, "r+") as fw:
        fw.attrs.update(
            {
                "format": "HDF5::MCOOL",
                "format-version": __format_version_mcool__,
                # Level names, sorted, so the listing doesn't depend on the
                # insertion order of the mapping
                "bin-levels": json.dumps(sorted(names)),
                # Which source each level was aggregated from ("base" or a
                # level name), for inspection.
                "bin-level-sources": json.dumps(
                    {name: (parent[name] or "base") for name in sorted(names)}
                ),
            }
        )


def _n_retained(target: TargetBins) -> int:
    return int((target.src_to_tgt >= 0).sum())


def _level_nnz(base, targets, candidate) -> int:
    """A cheap upper bound on the pixels a level stores."""
    # A level stores at most as many pixels as the base rows it retains; use
    # its retained fraction of the base's nnz as an estimate.
    nnz = base.info.get("nnz", 0)
    n = len(targets[candidate].src_to_tgt)
    return int(nnz * _n_retained(targets[candidate]) / max(n, 1))
