"""
Explicit, possibly sparse target bin tables for coarsening.

A target bin table is a BED-like DataFrame of ``chrom``, ``start`` and
``end`` columns whose rows define the output bins, in order.  Unlike the
ordinary bin table of a cooler, it need not tile the chromosomes: it may
omit source bins, whole chromosomes, or be empty.  Every target interval
must, however, be exactly a run of consecutive source bins: a target bin
can't split a source bin, cross a chromosome, or cover coordinates that the
source bin table doesn't cover.

The sparse layout is an extension of the cooler format, marked by the
``bin-layout`` attribute (see :class:`TargetBins` and the documentation).
Only cooler implementations that support it can read these files reliably;
ordinary coolers are unaffected.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = ["TargetBins", "check_level_name", "read_source_bins"]

# Level names in a bin_levels mapping: letters or digits first, then ASCII
# letters, digits, underscores and hyphens.
_LEVEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def check_level_name(name: object) -> str:
    """Validate and return a bin level name."""
    if not isinstance(name, str) or not _LEVEL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid bin level name: {name!r}. Level names must start with an "
            "ASCII letter or digit and contain only ASCII letters, digits, "
            "underscores and hyphens."
        )
    return name


def read_source_bins(clr) -> pd.DataFrame:
    """Read the ``chrom``, ``start`` and ``end`` columns of a cooler's bins."""
    bins = clr.bins()[["chrom", "start", "end"]][:]
    # Drop the pandas index so positions are bin IDs
    return bins.reset_index(drop=True)


def _coerce_coords(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"The target bin table has missing '{name}' values.")
    if not pd.api.types.is_integer_dtype(series.dtype):
        raise ValueError(
            f"The target bin table column '{name}' must contain integers, "
            f"not {series.dtype}. Coordinates are zero-based base pairs."
        )
    return series.to_numpy(dtype=np.int64, copy=True)


class TargetBins:
    """
    A validated target bin table for coarsening a cooler.

    Parameters
    ----------
    bins : pandas.DataFrame
        Normalized bin table: ``chrom`` (categorical, in the source
        chromosome order), ``start`` and ``end`` (int64), one row per output
        bin, in output bin ID order.
    src_to_tgt : ndarray of int64
        For each source bin, its target bin ID, or -1 if the source bin is
        not retained.
    tgt_src_start, tgt_src_end : ndarray of int64
        The half-open range of source bin IDs each target bin covers.
    chromsizes : pandas.Series
        The source chromosome lengths, in the source chromosome order.

    Notes
    -----
    The target table is an extension of the cooler bin table: it need not
    tile the chromosomes.  Output coolers made from it carry the metadata
    attribute ``bin-layout`` set to ``"sparse"``; the usual ``bin-type`` is
    ``"variable"`` and ``bin-size`` is ``"null"``, so region queries use the
    variable-bin code path.  Other cooler implementations that don't know
    about the extension may not interpret such files correctly.
    """

    def __init__(
        self,
        bins: pd.DataFrame,
        src_to_tgt: np.ndarray,
        tgt_src_start: np.ndarray,
        tgt_src_end: np.ndarray,
        chromsizes: pd.Series,
    ):
        self.bins = bins
        self.src_to_tgt = src_to_tgt
        self.tgt_src_start = tgt_src_start
        self.tgt_src_end = tgt_src_end
        self.chromsizes = chromsizes

    @property
    def chroms(self) -> list[str]:
        return list(self.chromsizes.index)

    @property
    def nbins(self) -> int:
        return len(self.bins)

    def __len__(self) -> int:
        return len(self.bins)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        chromsizes: pd.Series,
        src_bins: pd.DataFrame,
    ) -> TargetBins:
        """
        Validate a user-provided target bin table against source bins.

        Parameters
        ----------
        frame : pandas.DataFrame
            Target bin table with ``chrom``, ``start`` and ``end`` columns.
            Only the first three columns are used; the input is not modified.
        chromsizes : pandas.Series
            Chromosome lengths of the source, indexed by chromosome name.
        src_bins : pandas.DataFrame
            Source bins with ``chrom``, ``start`` and ``end`` columns, in
            source bin ID order.

        Returns
        -------
        TargetBins

        Raises
        ------
        ValueError
            If the table is missing columns, has unknown chromosomes,
            non-integer or out-of-bounds coordinates, empty or overlapping
            intervals, is not sorted by chromosome and start, or an interval
            is not exactly a run of consecutive source bins.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "target_bins must be a pandas DataFrame with columns 'chrom', "
                "'start' and 'end'."
            )
        for col in ("chrom", "start", "end"):
            if col not in frame.columns:
                raise ValueError(f"Missing column from target bin table: '{col}'.")

        chromsizes = pd.Series(
            np.asarray(chromsizes.to_numpy(), dtype=np.int64),
            index=list(chromsizes.index),
        )
        names = list(chromsizes.index)

        # Normalize without touching the caller's frame
        src_bins = src_bins.reset_index(drop=True)
        if len(frame) == 0:
            # A legal zero-row table. Accept any column dtypes, since there
            # are no coordinates to be non-integer.
            bins = pd.DataFrame(
                {
                    "chrom": pd.Categorical(
                        [], categories=names, ordered=True
                    ),
                    "start": np.array([], dtype=np.int64),
                    "end": np.array([], dtype=np.int64),
                }
            )
            return cls(
                bins,
                np.full(len(src_bins), -1, dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                chromsizes,
            )

        starts = _coerce_coords(frame["start"], "start")
        ends = _coerce_coords(frame["end"], "end")

        chrom = frame["chrom"]
        if chrom.isna().any():
            raise ValueError("The target bin table has missing 'chrom' values.")
        raw_chrom = np.asarray(chrom.astype(object))
        code_arr = pd.Index(names).get_indexer(raw_chrom)
        unknown = code_arr < 0
        if unknown.any():
            bad = sorted({str(x) for x in raw_chrom[unknown]})
            raise ValueError(
                "The target bin table has chromosomes not in the source: "
                f"{bad}."
            )
        codes = code_arr.astype(np.int64)

        if (starts < 0).any():
            i = int(np.argmax(starts < 0))
            raise ValueError(
                f"Target bin {i} has a negative start coordinate: {starts[i]}."
            )
        if (ends <= starts).any():
            i = int(np.argmax(ends <= starts))
            raise ValueError(
                f"Target bin {i} ({names[codes[i]]}:{starts[i]}-{ends[i]}) is "
                "empty or reversed: end must be greater than start."
            )
        lengths = chromsizes.to_numpy()[codes]
        if (ends > lengths).any():
            i = int(np.argmax(ends > lengths))
            raise ValueError(
                f"Target bin {i} ({names[codes[i]]}:{starts[i]}-{ends[i]}) "
                f"extends past the chromosome length ({int(lengths[i])})."
            )

        # Sorted by chromosome (in the source catalog's order), then start.
        if np.any(codes[1:] < codes[:-1]):
            i = int(np.flatnonzero(codes[1:] < codes[:-1])[0]) + 1
            raise ValueError(
                "The target bin table is not sorted by chromosome (in the "
                f"source order): row {i} goes back to {names[codes[i]]}."
            )
        same = codes[1:] == codes[:-1]
        if np.any(same & (starts[1:] < ends[:-1])):
            i = int(np.flatnonzero(same & (starts[1:] < ends[:-1]))[0]) + 1
            raise ValueError(
                f"Target bins {i - 1} and {i} overlap or are duplicated: "
                f"({names[codes[i - 1]]}:{starts[i - 1]}-{ends[i - 1]}) and "
                f"({names[codes[i]]}:{starts[i]}-{ends[i]})."
            )

        # Per-chromosome source bin ranges
        nchrom = len(names)
        src_codes = pd.Categorical(
            np.asarray(src_bins["chrom"].astype(object)),
            categories=names,
            ordered=True,
        ).codes
        if (src_codes < 0).any():
            bad = sorted(
                {str(x) for x in np.asarray(src_bins["chrom"])[src_codes < 0]}
            )
            raise ValueError(
                f"The source bin table has chromosomes not in chromsizes: {bad}."
            )
        counts = np.bincount(src_codes, minlength=nchrom)
        chrom_binoffset = np.r_[0, np.cumsum(counts)]

        src_starts = src_bins["start"].to_numpy(dtype=np.int64, copy=True)
        src_ends = src_bins["end"].to_numpy(dtype=np.int64, copy=True)

        n_tgt = len(frame)
        src_to_tgt = np.full(len(src_bins), -1, dtype=np.int64)
        tgt_src_start = np.full(n_tgt, -1, dtype=np.int64)
        tgt_src_end = np.full(n_tgt, -1, dtype=np.int64)

        # Validate each chromosome's target rows against its source bins.
        # Target rows are grouped by chromosome, since the table is sorted.
        boundaries = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, n_tgt]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            c = int(codes[a])
            s0, s1 = int(chrom_binoffset[c]), int(chrom_binoffset[c + 1])
            m = s1 - s0
            c_starts = src_starts[s0:s1]
            c_ends = src_ends[s0:s1]
            # Gaps between consecutive source bins (none in a complete table)
            gap = np.zeros(m, dtype=np.int64)
            if m > 1:
                gap[1:] = c_starts[1:] != c_ends[:-1]
            cum_gap = np.r_[0, np.cumsum(gap)]

            # The first source bin that starts at the target start, and the
            # source bin that ends at the target end (the last one covered).
            i0 = np.searchsorted(c_starts, starts[a:b], side="left")
            i1 = np.searchsorted(c_ends, ends[a:b], side="left")
            ok = (i0 < m) & (i1 < m)
            i0_safe = np.where(ok, i0, 0)
            i1_safe = np.where(ok, i1, 0)
            ok &= c_starts[i0_safe] == starts[a:b]
            ok &= c_ends[i1_safe] == ends[a:b]
            ok &= i1_safe >= i0_safe
            # No gap between the source bins of the run
            ok &= (cum_gap[i1_safe] - cum_gap[i0_safe]) == 0

            if not ok.all():
                j = int(np.flatnonzero(~ok)[0])
                row = a + j
                if i0[j] >= m or i1[j] >= m:
                    reason = "is not covered by the source bin table"
                else:
                    reason = "is not exactly a run of consecutive source bins"
                raise ValueError(
                    f"Target bin {row} "
                    f"({names[c]}:{starts[row]}-{ends[row]}) {reason}. "
                    "Target intervals must be unions of whole source bins, "
                    "within a chromosome, and can't cover coordinates that the "
                    "source bin table doesn't cover."
                )

            # The run of source bins is i0 .. i1 inclusive; the exclusive
            # stop is i1 + 1.
            tgt_ids = np.arange(a, b, dtype=np.int64)
            tgt_src_start[a:b] = s0 + i0
            tgt_src_end[a:b] = s0 + i1 + 1
            lens = i1 - i0 + 1
            if len(lens):
                total = int(lens.sum())
                if total:
                    ids = np.repeat(tgt_ids, lens)
                    offsets = np.r_[0, np.cumsum(lens)[:-1]]
                    within = np.arange(total) - np.repeat(offsets, lens)
                    src_ids = np.repeat(s0 + i0, lens) + within
                    src_to_tgt[src_ids] = ids

        bins = pd.DataFrame(
            {
                "chrom": pd.Categorical(
                    np.asarray(chrom.astype(object)), categories=names, ordered=True
                ),
                "start": starts,
                "end": ends,
            }
        )
        return cls(
            bins, src_to_tgt, tgt_src_start, tgt_src_end, chromsizes
        )

    def can_build_from(self, finer: TargetBins) -> bool:
        """
        Whether aggregating ``finer`` (which retains a superset of the source
        bins) and pooling its target bins reproduces a direct aggregation of
        this target table.

        This holds when every target bin here is exactly a run of whole
        target bins of ``finer`` and no source bin inside those runs is
        dropped by ``finer``.
        """
        if len(self.chromsizes) != len(finer.chromsizes) or not np.array_equal(
            self.chromsizes.to_numpy(), finer.chromsizes.to_numpy()
        ):
            return False
        if len(finer.src_to_tgt) != len(self.src_to_tgt):
            return False
        lo = self.tgt_src_start
        hi = self.tgt_src_end
        if len(lo) == 0:
            return True
        # Source bins at both ends of every target run must be retained by
        # the finer level, and fall on its target bin boundaries.
        first = finer.src_to_tgt[lo]
        last = finer.src_to_tgt[hi - 1]
        if np.any(first < 0) or np.any(last < 0):
            return False
        if np.any(finer.tgt_src_start[first] != lo):
            return False
        if np.any(finer.tgt_src_end[last] != hi):
            return False
        # No source bin inside a target run may be dropped by the finer level
        retained = np.r_[0, np.cumsum(finer.src_to_tgt >= 0)]
        return not np.any(retained[hi] - retained[lo] != hi - lo)

    def derives_from(self, finer: TargetBins):
        """
        The target over ``finer``'s output bins equivalent to this table, if
        this table can be built by pooling ``finer``'s whole bins.

        The result maps each of ``finer``'s output bin IDs to one of this
        table's bin IDs, using ``finer``'s bin table as the "source" bins.
        Returns ``None`` if the pooling wouldn't reproduce a direct
        aggregation.
        """
        if not self.can_build_from(finer):
            return None
        return TargetBins.from_frame(self.bins, self.chromsizes, finer.bins)
