#!/usr/bin/env python

import logging
from fileinput import filename
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import ztfimg
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt
from saltworks.plottools import binplot
from ztfimg.utils import vignets

import pocket_distortion
from eq_db import EqFuncDb

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


if __name__ == "__main__":
    qid = 1
    ccdid = 13
    filename = f"ztf_20191101191042_000600_zg_c{ccdid:02d}_o.fits.fz"
    rawimg = ztfimg.RawCCD.from_filename(filename, as_path=False)
    quad = rawimg.get_quadrant(qid)
    quad_data, overscan = quad.get_data_and_overscan(
        stacked=False, corr_glow=False, corr_overscan=True
    )

    # meta data from the header
    header = rawimg.get_header()
    temp = float(header["HEADTEMP"])
    mjd = float(header["OBSMJD"])

    # fetch the equilibrium function from the EqDb
    db = EqFuncDb.open("prod/eq_db")
    f_eq = db.get_eq_func(ccdid, qid, mjd=mjd, ccd_temp=temp)

    # plot the equilibrium function, just to inspect it
    xx = jnp.linspace(10.0, 2500.0, 1000)
    plt.figure()
    plt.plot(xx, f_eq(xx), "b--")
    plt.title(
        f"Eq. function: ccdid={ccdid}, qid={qid}, mjd={mjd:.2f}, temp: {temp:.2f}"
    )
    pocket_distortion.plot_1d(f_eq)
    pocket_distortion.plot_2d(f_eq)

    # correct the quadrant
    corrected = pocket_distortion.invert(quad_data, f_eq)

    # plot the frames
    fig, axes = plt.subplots(figsize=(16, 6), nrows=1, ncols=3, sharex=1, sharey=1)
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(quad_data)
    axes[0].imshow(quad_data, interpolation="none", vmin=vmin, vmax=vmax)
    vmin, vmax = interval.get_limits(corrected)
    axes[1].imshow(corrected, interpolation="none", vmin=vmin, vmax=vmax)
    vmin, vmax = interval.get_limits(corrected - quad_data)
    axes[2].imshow(corrected - quad_data, interpolation="none", vmin=vmin, vmax=vmax)

    # compute the moments
    svignets = vignets.SourceVignets.from_image(np.asarray(quad_data))
    moments_nocorr = svignets.get_moments(psf_weighted=False, join_sources=True)
    svignets = vignets.SourceVignets.from_image(np.asarray(corrected))
    moments_corr = svignets.get_moments(psf_weighted=False, join_sources=True)
    r = moments_nocorr, moments_corr

    # and plot the moments
    fig, axes = plt.subplots(figsize=(8, 4), nrows=1, ncols=2, sharey=True)
    axes[0].plot(
        -2.5 * np.log10(r[0].flux), r[0].m_x3, "k,", alpha=0.25, label="uncorrected"
    )
    _ = binplot(
        -2.5 * np.log10(r[0].flux),
        r[0].m_x3,
        color="k",
        marker="o",
        nbins=10,
        ax=axes[0],
        label="uncorrected",
    )
    axes[0].set_title("uncorrected")

    axes[1].plot(
        -2.5 * np.log10(r[1].flux), r[1].m_x3, "r,", alpha=0.25, label="corrected"
    )
    _ = binplot(
        -2.5 * np.log10(r[1].flux),
        r[1].m_x3,
        color="r",
        marker="o",
        nbins=10,
        ax=axes[1],
        label="corrected",
    )
    axes[1].set_ylim((-5.0, 5.0))
    axes[1].set_title("corrected")
