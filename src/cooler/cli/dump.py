import contextlib
import gzip
import sys

import click
import pandas as pd

from .. import api
from ..core import (
    CSRReader,
    DirectRangeQuery2D,
    FillLowerRangeQuery2D,
    region_to_extent,
)
from ..expected import load_expected
from ..util import parse_region
from . import cli
from ._util import DelimitedTuple


def make_annotator(bins, balanced, join, annotate, one_based_ids, one_based_starts):
    def annotator(chunk):
        if annotate is not None:
            extra = api.annotate(
                chunk[["bin1_id", "bin2_id"]], bins[list(annotate)], replace=True
            )

        if balanced:
            df = api.annotate(chunk, bins[[balanced]])
            chunk["balanced"] = df[f"{balanced}1"] * df[f"{balanced}2"] * chunk["count"]

        if join:
            chunk = api.annotate(chunk, bins[["chrom", "start", "end"]], replace=True)

        if annotate is not None:
            chunk = pd.concat([chunk, extra], axis=1)

        if one_based_ids:
            for col in ["bin1_id", "bin2_id"]:
                if col in chunk.columns:
                    chunk[col] += 1

        if one_based_starts:
            for col in ["start1", "start2"]:
                if col in chunk.columns:
                    chunk[col] += 1

        return chunk

    return annotator


@cli.command()
@click.argument("cool_uri", metavar="COOL_PATH")
@click.option(
    "--table",
    "-t",
    help="Which table to dump. Choosing 'chroms' or 'bins' will cause all "
    "pixel-related options to be ignored. Note that for coolers stored "
    "in symmetric-upper mode, 'pixels' only holds the upper triangle "
    "values of the matrix.",
    type=click.Choice(["chroms", "bins", "pixels"]),
    default="pixels",
    show_default=True,
)
@click.option(
    "--columns",
    "-c",
    help="Restrict output to a subset of columns, provided as a "
    "comma-separated list. For pixels, the default is every column of the "
    "pixel table.",
    type=DelimitedTuple(sep=","),
)
@click.option(
    "--header",
    "-H",
    help="Print the header of column names as the first row.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--na-rep", help="Missing data representation. Default is empty ''.", default=""
)
@click.option(
    "--float-format",
    help="Format string for floating point numbers (e.g. '.12g', '03.2f').",
    default="g",
    show_default=True,
)
@click.option(
    "--range",
    "-r",
    help="The coordinates of a genomic region shown along the row dimension, "
    "as a range string with zero-based, half-open coordinates. "
    "(Example: chr1:10,000,000-11,000,000). "
    "If omitted, the entire contact matrix is printed.",
    type=str,
)
@click.option(
    "--range2",
    "-r2",
    type=str,
    help="The coordinates of a genomic region shown along the column dimension. "
    "If omitted, the column range is the same as the row range.",
)
@click.option(
    "--fill-lower",
    "-f",
    help="For coolers using 'symmetric-upper' storage, populate implicit areas "
    "of the genomic query box by generating lower triangle pixels. If not "
    "specified, only upper triangle pixels are reported. This option has no "
    "effect on coolers stored in 'square' mode.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--balanced/--no-balance",
    "-b",
    help="Apply balancing weights to data. This will print an extra column "
    "called `balanced`",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--expected",
    "expected_layer",
    default=None,
    help="Name of a stored distance-background layer (see "
    "`cooler.estimate_expected`). Adds `expected` and `oe` (observed over "
    "expected) columns using the layer's bound weights and selection. "
    "Incompatible with --balanced/--weight-name and with non-count columns.",
)
@click.option(
    "--weight-name",
    help="Name of the bin table column containing the balancing weights. "
    "Implies --balanced. [default: weight]",
)
@click.option(
    "--join",
    help="Print the full chromosome bin coordinates instead of bin IDs. "
    "This will replace the `bin1_id` column with `chrom1`, `start1`, and "
    "`end1`, and the `bin2_id` column with `chrom2`, `start2` and `end2`.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--annotate",
    help="Join additional columns from the bin table against the pixels. "
    "Provide a comma separated list of column names (no spaces). "
    "The merged columns will be suffixed by '1' and '2' accordingly.",
    type=DelimitedTuple(sep=","),
)
@click.option(
    "--one-based-ids",
    help="Print bin IDs as one-based rather than zero-based.",
    is_flag=True,
    default=False,
)
@click.option(
    "--one-based-starts",
    help="Print start coordinates as one-based rather than zero-based. Only "
    "applies with --join, which adds the start columns.",
    is_flag=True,
    default=False,
)
@click.option(
    "--chunksize",
    "-k",
    help="Sets the number of pixel records loaded from disk at one time. "
    "Can affect the performance of joins on high resolution datasets. ",
    type=int,
    default=1_000_000,
    show_default=True,
)
@click.option(
    "--out",
    "-o",
    help="Output text file. If .gz extension is detected, file is written "
    "using zlib. Default behavior is to stream to stdout.",
)
def dump(
    cool_uri,
    table,
    columns,
    header,
    na_rep,
    float_format,
    range,
    range2,
    fill_lower,
    balanced,
    expected_layer,
    weight_name,
    join,
    annotate,
    one_based_ids,
    one_based_starts,
    chunksize,
    out,
):
    """
    Dump a cooler's data to a text stream.

    COOL_PATH : Path to COOL file or cooler URI.

    """
    if range2 is not None and range is None:
        raise click.BadParameter(
            "Give the rows' range with --range too.", param_hint="'--range2'"
        )
    clr = api.Cooler(cool_uri)
    if table != "pixels" and columns is not None:
        available = (clr.chroms() if table == "chroms" else clr.bins()).columns
        unknown = [col for col in columns if col not in available]
        if unknown:
            raise click.BadParameter(
                f"Not in the {table} table: {', '.join(unknown)}",
                param_hint="'--columns'",
            )

    with contextlib.ExitStack() as stack:
        # Choose the output stream
        if out is None or out == "-":
            f = sys.stdout
        elif out.endswith(".gz"):
            f = stack.enter_context(gzip.open(out, "wt"))
        else:
            f = stack.enter_context(open(out, "w"))

        # Choose the source table
        if table == "chroms":
            selector = clr.chroms()
            if columns is not None:
                selector = selector[list(columns)]
            chunks = (selector[:],)

        elif table == "bins":
            selector = clr.bins()
            if columns is not None:
                selector = selector[list(columns)]
            chunks = (selector[:],)

        else:  # Pixel table
            chunks = _pixel_chunks(
                clr,
                stack,
                columns,
                range,
                range2,
                fill_lower,
                balanced,
                expected_layer,
                weight_name,
                join,
                annotate,
                one_based_ids,
                one_based_starts,
                chunksize,
            )

        if float_format is not None:
            float_format = "%" + float_format

        is_first_chunk = True
        for chunk in chunks:
            if is_first_chunk:
                if header:
                    chunk[0:0].to_csv(
                        f,
                        sep="\t",
                        lineterminator="\n",
                        index=False,
                        header=True,
                        float_format=float_format,
                        na_rep=na_rep,
                    )
                is_first_chunk = False

            chunk.to_csv(
                f,
                sep="\t",
                lineterminator="\n",
                index=False,
                header=False,
                float_format=float_format,
                na_rep=na_rep,
            )

        f.flush()


def _pixel_chunks(
    clr,
    stack,
    columns,
    range,
    range2,
    fill_lower,
    balanced,
    expected_layer,
    weight_name,
    join,
    annotate,
    one_based_ids,
    one_based_starts,
    chunksize,
):
    # Load all the bins
    bins = clr.bins()[:]
    n_bins = len(bins)
    if chunksize is None:
        chunksize = len(bins)

    if annotate is not None:
        unknown = [col for col in annotate if col not in bins.columns]
        if unknown:
            raise click.BadParameter(
                f"Not in the bin table: {', '.join(unknown)}",
                param_hint="'--annotate'",
            )

    background = None
    if expected_layer is not None:
        if balanced or weight_name is not None:
            raise click.BadParameter(
                "--expected uses the balancing weights bound to the layer and "
                "cannot be combined with --balanced/--weight-name",
                param_hint="'--expected'",
            )
        try:
            result = load_expected(clr, expected_layer)
        except (KeyError, ValueError) as exc:
            raise click.ClickException(str(exc)) from None
        background = result.as_background()

    if weight_name is not None or balanced:
        balanced = weight_name or "weight"
        if balanced not in bins.columns:
            raise click.ClickException(f"Balancing weights '{balanced}' not found")

    # Value columns to write: every one by default, or those requested
    pixel_columns = list(clr.pixels().columns)
    explicit_columns = columns is not None
    if columns is None:
        columns = pixel_columns
    allowed = set(pixel_columns)
    if background is not None:
        allowed.update(("expected", "oe"))
    unknown = [col for col in columns if col not in allowed]
    if unknown:
        raise click.BadParameter(
            f"Not in the pixel table: {', '.join(unknown)}",
            param_hint="'--columns'",
        )
    fields = [col for col in columns if col not in ("bin1_id", "bin2_id")]
    if background is not None:
        virtual_requested = [f for f in ("expected", "oe") if f in fields]
        if not virtual_requested:
            if explicit_columns:
                # The caller asked for stored columns only (e.g. just count);
                # do not add virtual columns they did not request.
                virtual_requested = []
            else:
                # Default output: emit both virtual columns.
                virtual_requested = ["expected", "oe"]
        stored_requested = [f for f in fields if f not in ("expected", "oe")]
        bad = [f for f in stored_requested if f != "count"]
        if bad:
            raise click.BadParameter(
                "--expected can only be combined with the 'count' field; "
                f"got {', '.join(bad)}",
                param_hint="'--columns'",
            )
        # The transform always needs count; virtual columns are annotated by
        # the engine and unrequested ones are dropped before writing.
        engine_fields = ["count", *virtual_requested]
        output_value_fields = [
            *(["count"] if "count" in fields else []),
            *virtual_requested,
        ]
        if not explicit_columns:
            output_value_fields = ["count", *virtual_requested]
        drop = [f for f in ("count", "expected", "oe") if f not in output_value_fields]
    else:
        engine_fields = list(fields)
        if not engine_fields:
            engine_fields = (
                ["count"] if "count" in pixel_columns else pixel_columns[2:3]
            )
        if balanced and "count" not in engine_fields:
            if "count" not in pixel_columns:
                raise click.ClickException("Balancing needs a 'count' column")
            engine_fields.append("count")
        output_value_fields = fields
        drop = [col for col in engine_fields if col not in fields]
    drop += [col for col in ("bin1_id", "bin2_id") if col not in columns]

    h5 = stack.enter_context(clr.open("r"))
    reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])

    if range:
        # User-specified bbox provided
        i0, i1 = region_to_extent(
            h5,
            clr._chromids,
            parse_region(range, clr.chromsizes),
            binsize=clr.binsize,
        )
        if range2 is not None:
            j0, j1 = region_to_extent(
                h5,
                clr._chromids,
                parse_region(range2, clr.chromsizes),
                binsize=clr.binsize,
            )
        else:
            j0, j1 = i0, i1
        bbox = (i0, i1, j0, j1)
    else:
        # Dump everything
        bbox = (0, n_bins, 0, n_bins)

    if fill_lower and clr.storage_mode == "symmetric-upper":
        engine = FillLowerRangeQuery2D(
            reader, engine_fields, bbox, chunksize, background=background
        )
    else:
        engine = DirectRangeQuery2D(
            reader, engine_fields, bbox, chunksize, background=background
        )

    output_columns = ["bin1_id", "bin2_id", *engine_fields]
    chunks = (pd.DataFrame(dct, columns=output_columns) for dct in engine)

    if balanced or join or annotate or one_based_ids:
        annotator = make_annotator(
            bins, balanced, join, annotate, one_based_ids, one_based_starts
        )
        chunks = map(annotator, chunks)

    if drop:
        chunks = (chunk.drop(columns=drop, errors="ignore") for chunk in chunks)

    return chunks
