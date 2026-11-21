.. _expected:

Distance background and observed/expected ratios
=================================================

Contacts between nearby bins are always more frequent than contacts between
distant bins. A *distance background* estimates the expected signal level as
a function of genomic distance, so that observed/expected (OE) ratios reveal
which regions remain elevated **after** the distance decay is removed.

The estimator works directly on a fixed cooler archive: bins and raw pixels
are never reordered or re-aggregated, and non-uniform or "holey" bin layouts
are handled from the bin coordinates rather than from sparse pixel IDs.

.. contents::
    :local:
    :depth: 2

Estimating the background
-------------------------

.. autofunction:: cooler.estimate_expected

Distance is the absolute difference between two bin **midpoints**, which can
be a half-integer number of base pairs for non-uniform bins. Buckets are
half-open intervals ``[edges[d], edges[d+1])``. Every valid pair with
``i <= j`` is counted once (the diagonal is counted once), including pairs
that have no stored pixel: implicit zeros are part of every denominator.

The returned object exposes per-chromosome, per-bucket statistics::

    import cooler

    clr = cooler.Cooler("map.cool")
    result = cooler.estimate_expected(clr, [0, 1_000, 5_000, 25_000, 100_000])
    print(result.summary())

``summary()`` lists, for each chromosome and distance bucket, the weighted
observed total, the number of valid bin pairs, the raw count total and raw
mean, the final expected value, whether it is usable as a divisor and the
provenance status (``raw``, ``smoothed``, ``despiked`` or ``unavailable``).
Buckets without valid pairs are never given a value; buckets whose observed
total is zero keep a raw mean of zero but cannot be used as divisors.

Only ``symmetric-upper`` coolers are supported. A ``square`` cooler raises
``ValueError``.

Balancing weights
-----------------

Pass ``balance=True`` for the default ``weight`` column or a column name for
any other vector::

    result = cooler.estimate_expected(clr, edges, balance="weight")
    result = cooler.estimate_expected(clr, edges, balance="KR")  # divisive by default
    result = cooler.estimate_expected(
        clr, edges, balance="KR", divisive_weights=False
    )

Finite positive weights mark valid bins. NaN and zero weights exclude a bin;
negative or infinite weights raise. The multiplicative/divisive convention
and the ``KR``/``VC``/``VC_SQRT`` alias defaults are the same ones used by
:meth:`cooler.Cooler.matrix`. The weighted count is the observed quantity for
the background; the raw count statistics are always retained alongside it.

Stabilization
-------------

With ``stabilize=False`` (the default) the expected value is exactly the
bucket mean. With ``stabilize=True`` a robust, single-chromosome policy is
applied in log space:

* a Huber-weighted local trend smooths low-support buckets (``window``,
  ``min_support``),
* isolated positive spikes are replaced by the trend only when their
  neighbours stay within a robust MAD band (``spike_z``), so sustained
  multi-bucket enhancements are retained,
* the low-support tail is optionally forced monotone non-increasing
  (``tail_monotone``),
* unsupported buckets are never filled and a constant positive background is
  preserved exactly.

All parameters are stored in ``result.policy`` and the raw statistics remain
available for audit::

    stable = cooler.estimate_expected(
        clr,
        edges,
        stabilize=True,
        min_support=30,
        window=7,
        huber_delta=1.345,
        spike_z=3.0,
        tail_monotone=True,
    )

Stabilization is a bias/variance trade-off: it lowers the variance of sparse
tail estimates at the price of shrinking genuine single-bucket spikes. It
never borrows samples across chromosomes.

Saving and reloading named layers
----------------------------------

An estimate can be stored as a named analysis layer **inside the cooler
group**, then reused in a later process::

    cooler.save_expected(clr, result, "my_background")
    cooler.list_expected(clr)                 # ['my_background']
    again = cooler.load_expected(clr, "my_background")

Each layer binds:

* the exact source bin layout (content hashes of chrom/start/end),
* the full weight vector and its multiplicative/divisive interpretation,
* the distance edges and the complete fitting policy.

Reloading verifies the bindings. If the bin coordinates or the source weights
change, selecting the layer raises ``ValueError``; the raw pixel data is
treated as immutable, and copying the file to another path does not affect
usability because bindings are content hashes rather than paths.

Overwriting requires an explicit flag, and writes go through a temporary
group so a failure never damages an existing layer::

    cooler.save_expected(clr, result, "my_background")
    # cooler.save_expected(clr, result, "my_background")          # raises
    cooler.save_expected(clr, result, "my_background", overwrite=True)
    cooler.remove_expected(clr, "my_background")

Querying observed/expected ratios
----------------------------------

:meth:`cooler.Cooler.matrix` accepts ``expected=`` with either an in-memory
:class:`~cooler.ExpectedResult` or a stored layer name::

    clr.matrix(balance=False, expected=result)[:]            # dense OE array
    clr.matrix(expected="my_background", sparse=True)[:]     # sparse COO
    clr.matrix(expected="my_background", as_pixels=True)[:]  # pixel rows

The balance configuration and weight interpretation bound to the estimate are
used automatically. Explicitly requesting an incompatible ``balance``, weight
name or ``divisive_weights`` value raises; omitting ``balance`` does not
conflict with an unweighted layer. Non-count fields cannot be combined with
``expected``.

Same-chromosome cells with a finite positive expected value return OE
ratios. Inter-chromosomal cells, distances outside the edges, unsupported
buckets, non-divisive (zero) expectations and excluded bins are NaN in dense
output; valid implicit zeros become an exact ratio of zero. Dense and sparse
outputs agree cell by cell, including NaN positions. Pixel output only
returns originally stored rows within the query window and adds
``expected`` and ``oe`` columns while keeping the raw ``count``; no
symmetric or missing pixels are fabricated.

A whole-map result sliced into windows gives the same values as the
corresponding slice of a full read; backgrounds are never re-estimated per
fetch.

Low-level range queries
-----------------------

The streaming engines accept a
:class:`~cooler.expected.BackgroundModel` (built once with
``result.as_background()``) and expose the virtual numeric fields
``expected`` and ``oe`` alongside the raw stored fields::

    from cooler.core import CSRReader, DirectRangeQuery2D, FillLowerRangeQuery2D

    background = result.as_background()
    with clr.open() as h5:
        reader = CSRReader(h5["pixels"], h5["indexes/bin1_offset"][:])
        query = DirectRangeQuery2D(
            reader, ["count", "expected", "oe"], bbox, chunksize,
            background=background,
        )
        for chunk in query:          # get_chunk / to_frame / to_array all work
            ...

``DirectRangeQuery2D`` never adds transposed rows;
``FillLowerRangeQuery2D`` reflects the symmetric lower triangle exactly as
without a background. Default field requests remain unchanged when no
background is supplied.

Command line
------------

``cooler dump`` streams the same two virtual columns from a stored layer::

    cooler dump map.cool --expected my_background -H
    cooler dump map.cool --expected my_background --fill-lower -r chr1:0-1000000

Selection, upper/lower triangle handling and weight interpretation match the
Python matrix reader exactly; all other dump invocations keep their default
output.

API reference
-------------

.. autoclass:: cooler.ExpectedResult
    :members:
.. autoclass:: cooler.expected.BackgroundModel
    :members:
.. autofunction:: cooler.estimate_expected
.. autofunction:: cooler.save_expected
.. autofunction:: cooler.load_expected
.. autofunction:: cooler.list_expected
.. autofunction:: cooler.remove_expected
