#!/usr/bin/env python

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from astropy.time import Time

from eq_fit import FitResults, fit_eq_model
from eq_plots import FitDiagnostics, FitGallery
from poly_temp_eq_model import PolyTempEqModel, PolyTempEqModel2
from spline_temp_eq_model import SplineTempEqModel

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def load(
    fn: Path,
    ccdid: int = 6,
    qid: int = 1,
    mjd_range: tuple | None = None,  # specifier que c'est n tuple de 2 int ?
    night: str = None,
):
    df = pl.scan_parquet(fn)
    df = df.filter(
        (pl.col("ccdid") == ccdid)
        & (pl.col("qid") == qid)
        & (pl.col("cryotemp").is_not_null())
    )
    if mjd_range is not None:
        df = df.filter(pl.col("mjd").is_between(*mjd_range))
    if night is not None:
        year, month, day = map(int, night.split("-"))
        df = df.filter(
            (pl.col("year") == year)
            & (pl.col("month") == month)
            & (pl.col("day") == day)
        )

    dd = df.group_by("filefracday").agg(
        [
            pl.col("year").first(),
            pl.col("month").first(),
            pl.col("day").first(),
            pl.col("mjd").first(),
            pl.col("field").first(),
            pl.col("ccdid").first(),
            pl.col("filterid").first(),
            pl.col("qid").first(),
            pl.col("gain").first(),
            pl.col("fieldid").first(),
            pl.col("ccdtemp").first(),
            pl.col("cryotemp").first(),
            pl.col("head_temp").first(),
            pl.col("dewpressure").first(),
            pl.col("azimuth").first(),
            pl.col("elvation").first(),
            pl.col("airmass").first(),
            (pl.col("median") - pl.col("pedestal_avg")).first().alias("skylev"),
            pl.col("pedestal_avg").first(),
            pl.col("pedestal1").first(),
            (pl.col("overscan_val") - pl.col("pedestal_avg"))
            .sum()
            .alias("overscan_sum"),
            (pl.col("last_val") - pl.col("pedestal_avg"))
            .median()
            .alias("last_col_skylev"),
        ]
    )

    return dd.collect()


if __name__ == "__main__":
    # import sys
    # sys.exit(0)

    # equilibrium model -
    eq_model = None
    if False:
        basis_grid = np.geomspace(60.0, 10000.0, 7)
        eq_model = SplineTempEqModel(
            basis_grid=basis_grid, temp_deg=3, temp_ref=160.0, temp_scale=1.0
        )
    else:
        eq_model = PolyTempEqModel2()

    fits = FitResults(model=eq_model)
    gallery = FitGallery()

    for ccdid in range(1, 17):
        for qid in [1]:
            logging.info(f"processing: ccdid={ccdid} qid={qid}")
            train_df = load(
                Path(
                    "/home/nrl/lemaitre/ztf-sensor-analysis/pocket/training/data/pocket_training_all_fields.parquet"
                ),
                ccdid=ccdid,
                qid=qid,
                mjd_range=(Time("2019-10-23").mjd, Time("2024-12-31").mjd),
            )

            record, diag = fit_eq_model(
                train_df,
                ccdid=ccdid,
                qid=qid,
                eq_model=eq_model,
            )

            fits.append(record)
            gallery.add(diag)

    fits.save("prod/eq_db")
    gallery.save_all("prod/training_gallery")
    gallery.write_index("prod/training_gallery/index.parquet")
    gallery.write_html("prod/training_gallery/index.html")
