from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from astropy.time import Time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ztfsensors.pocket.db import EqFuncDb

from ..fit import FitDiagnostics

READOUT_UPGRADES = [
    Time("2018-06-26").mjd,
    Time("2019-10-22").mjd,
    Time("2020-04-29").mjd,
    Time("2022-10-04").mjd,
]


# ---------------------------------------------------------------------------
# Shared gallery infrastructure
# ---------------------------------------------------------------------------


@dataclass
class GalleryItem:
    """A single entry in a plot gallery."""

    ccdid: int
    qid: int
    mjd_start: float
    mjd_end: float
    path: Path


# Backward-compatibility alias
FitGalleryItem = GalleryItem


def _write_gallery_html(
    items: list[GalleryItem],
    output: Path,
    title: str = "Gallery",
    thumb_width: int = 250,
) -> None:
    """
    Write an HTML gallery from a list of GalleryItems.

    Layout: one (ccdid, qid) pair per row, one MJD period per column.
    Each cell contains a clickable thumbnail; missing combinations show "—".

    Parameters
    ----------
    items :
        Gallery items produced by a ``save_all()`` call.
    output :
        Path of the HTML file to write.
    title :
        Page title shown in the ``<h1>`` header.
    thumb_width :
        Width in pixels of each thumbnail image.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not items:
        raise ValueError("No items — call save_all() first.")

    html_dir = output.parent

    ccd_qids = sorted({(it.ccdid, it.qid) for it in items})

    # One column per unique mjd_start (keeps mjd_end for the label)
    mjd_map: dict[float, tuple[float, float]] = {}
    for it in items:
        mjd_map.setdefault(it.mjd_start, (it.mjd_start, it.mjd_end))
    periods = sorted(mjd_map.values())

    lookup: dict[tuple[int, int, float], Path] = {
        (it.ccdid, it.qid, it.mjd_start): it.path for it in items
    }

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(html_dir))
        except ValueError:
            return str(p)

    col_headers = "\n".join(
        f'          <th title="[{s:.3f}, {e:.3f})">'
        f"{s:.1f}<br>"
        f'<span class="sub">–{e:.1f}</span></th>'
        for s, e in periods
    )

    rows_html_parts = []
    for ccdid, qid in ccd_qids:
        cells = [f"          <th>ccd{ccdid:02d}<br>q{qid}</th>"]
        for mjd_start, _ in periods:
            img = lookup.get((ccdid, qid, mjd_start))
            if img is None:
                cells.append('          <td class="missing">—</td>')
            else:
                src = rel(img)
                cells.append(
                    f"          <td>"
                    f'<a href="{src}" target="_blank">'
                    f'<img src="{src}" width="{thumb_width}" loading="lazy"'
                    f' alt="ccd{ccdid:02d} q{qid} mjd={mjd_start:.1f}">'
                    f"</a></td>"
                )
        rows_html_parts.append("        <tr>\n" + "\n".join(cells) + "\n        </tr>")

    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; font-size: 13px; padding: 1em; background: #fff; }
    h1 { font-size: 1.2em; margin-bottom: 0.8em; color: #333; }
    .wrap { overflow-x: auto; }
    table { border-collapse: collapse; }
    th, td {
      border: 1px solid #ddd;
      padding: 4px 6px;
      text-align: center;
      vertical-align: middle;
    }
    thead th {
      background: #e4e8ef;
      position: sticky;
      top: 0;
      z-index: 2;
      font-size: 11px;
      white-space: nowrap;
    }
    thead th .sub { color: #888; }
    tbody th {
      background: #f4f4f4;
      position: sticky;
      left: 0;
      z-index: 1;
      font-size: 12px;
      white-space: nowrap;
    }
    td.missing { color: #ccc; background: #fafafa; font-size: 1.4em; }
    img { display: block; height: auto; }"""

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{css}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th>ccdid / qid</th>
{col_headers}
        </tr>
      </thead>
      <tbody>
{chr(10).join(rows_html_parts)}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    output.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Equilibrium function plot
# ---------------------------------------------------------------------------


def plot_tabulated_eq_func(
    db: EqFuncDb,
    ccdid: int,
    qid: int,
    mjd: float,
    n_temps: int = 5,
    figsize: tuple[float, float] = (8, 5),
    flux_min: float = 50.0,
    flux_max: float = 5000.0,
    xscale: str = "log",
    cmap_name: str = "plasma",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the equilibrium function at several CCD temperatures.

    Retrieves the fitted model for ``(ccdid, qid, mjd)`` and evaluates it
    at ``n_temps`` temperatures spanning ``[temp_min, temp_max]`` as stored
    in the database.

    Parameters
    ----------
    db :
        Equilibrium function database.
    ccdid :
        CCD identifier.
    qid :
        Quadrant identifier.
    mjd :
        Any MJD inside the desired validity interval.
    n_temps :
        Total number of temperature curves to draw, including ``temp_min``
        and ``temp_max``.  Must be ≥ 2.
    figsize :
        Matplotlib figure size ``(width, height)`` in inches.
    flux_min :
        Minimum sky level for the x-axis.  Must be > 0 when ``xscale="log"``.
    flux_max :
        Maximum sky level for the x-axis.
    xscale :
        X-axis scale — ``"log"`` (default) or ``"linear"``.
    cmap_name :
        Matplotlib colormap used to colour the temperature curves.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    if flux_min >= flux_max:
        raise ValueError(f"flux_min={flux_min} must be < flux_max={flux_max}")
    if xscale == "log" and flux_min <= 0:
        raise ValueError(
            f"flux_min must be > 0 for xscale='log', got flux_min={flux_min}"
        )
    n_temps = max(2, int(n_temps))

    # Single DB lookup — extract everything we need at once
    row = db.select_row(ccdid, qid, mjd)
    temp_min = float(row["temp_min"])
    temp_max = float(row["temp_max"])
    mjd_start = float(row["mjd_start"])
    mjd_end = float(row["mjd_end"])

    if temp_min >= temp_max:
        raise ValueError(
            f"Degenerate temperature range in DB: "
            f"temp_min={temp_min} >= temp_max={temp_max}"
        )

    temps = np.linspace(temp_min, temp_max, n_temps)

    flux = (
        np.geomspace(flux_min, flux_max, 300)
        if xscale == "log"
        else np.linspace(flux_min, flux_max, 300)
    )

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=temp_min, vmax=temp_max)

    fig, ax = plt.subplots(figsize=figsize)

    for temp in temps:
        f_eq = db.get_eq_func(ccdid, qid, mjd, float(temp))
        y = np.asarray(f_eq(flux))
        ax.plot(
            flux,
            y,
            ls="-",
            marker="",
            color=cmap(norm(float(temp))),
            label=f"$T={temp:.1f}$",
        )

    # Colorbar (more readable than a legend when n_temps > 3)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="CCD temp [K]")

    ax.set_xscale(xscale)
    ax.set_xlabel("sky level [ADU]")
    ax.set_ylabel("overscan signal [ADU]")
    ax.set_title(
        f"Eq. function — ccd{ccdid:02d} q{qid} | mjd=[{mjd_start:.1f}, {mjd_end:.1f})"
    )

    return fig, ax


# ---------------------------------------------------------------------------
# Equilibrium-function gallery
# ---------------------------------------------------------------------------


@dataclass
class EqFuncGallery:
    """
    Gallery of equilibrium-function plots produced from an :class:`EqFuncDb`.

    Usage
    -----
    ::

        gallery = EqFuncGallery(db)
        gallery.save_all("output/eq_func_gallery", n_temps=5)
        gallery.write_html("output/eq_func_gallery/index.html")
    """

    db: EqFuncDb
    items: list[GalleryItem] = field(default_factory=list)

    def save_all(
        self,
        output_dir: str | Path,
        close_figures: bool = True,
        **plot_kwargs,
    ) -> None:
        """
        Generate one plot per DB row and save it to *output_dir*.

        Parameters
        ----------
        output_dir :
            Root directory for the output images.  Created if absent.
        close_figures :
            If ``True`` (default), close each figure after saving to keep
            memory usage low.
        **plot_kwargs :
            Forwarded verbatim to :func:`plot_tabulated_eq_func`
            (e.g. ``n_temps``, ``flux_min``, ``flux_max``, ``xscale``).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.items.clear()

        for row in self.db.df.sort(["ccdid", "qid", "mjd_start"]).iter_rows(named=True):
            ccdid = int(row["ccdid"])
            qid = int(row["qid"])
            mjd_start = float(row["mjd_start"])
            mjd_end = float(row["mjd_end"])

            fig, _ = plot_tabulated_eq_func(
                db=self.db,
                ccdid=ccdid,
                qid=qid,
                mjd=mjd_start,  # mjd_start satisfies start <= mjd < end
                **plot_kwargs,
            )

            subdir = output_dir / f"ccdid_{ccdid:02d}" / f"qid_{qid:02d}"
            subdir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"eq_func_ccdid_{ccdid:02d}"
                f"_qid_{qid:02d}"
                f"_mjd_{mjd_start:.3f}_{mjd_end:.3f}.png"
            )
            path = subdir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            if close_figures:
                plt.close(fig)

            self.items.append(
                GalleryItem(
                    ccdid=ccdid,
                    qid=qid,
                    mjd_start=mjd_start,
                    mjd_end=mjd_end,
                    path=path,
                )
            )

    def to_dataframe(self) -> pl.DataFrame:
        """Return gallery metadata as a Polars DataFrame."""
        return pl.DataFrame(
            {
                "ccdid": [it.ccdid for it in self.items],
                "qid": [it.qid for it in self.items],
                "mjd_start": [it.mjd_start for it in self.items],
                "mjd_end": [it.mjd_end for it in self.items],
                "path": [str(it.path) for it in self.items],
            }
        )

    def write_html(
        self,
        output: str | Path,
        thumb_width: int = 250,
        title: str = "Equilibrium Function Gallery",
    ) -> None:
        """Write an HTML gallery index.  Call :meth:`save_all` first."""
        _write_gallery_html(
            self.items,
            output=Path(output),
            title=title,
            thumb_width=thumb_width,
        )


# ---------------------------------------------------------------------------
# Fit-diagnostics plot
# ---------------------------------------------------------------------------


def plot_equilibrium_fit(
    diag: FitDiagnostics,
    figsize=(10, 10),
    scatter_size=2,
    hist_bins=50,
    show_reference_curve=True,
    reference_temperature=None,
    model_on_data=True,
):
    xx = diag.xx
    yy = diag.yy
    temp = diag.temp
    mjd = diag.mjd
    yhat = diag.yhat
    resid = diag.resid
    model = diag.model
    record = diag.record
    bads = diag.bads

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[4, 1, 0.3, 1],
        hspace=0.05,
    )

    ax_main = fig.add_subplot(gs[0])
    ax_res_x = fig.add_subplot(gs[1], sharex=ax_main)
    ax_res_t = fig.add_subplot(gs[3])

    # main panel
    sc = ax_main.scatter(xx[~bads], yy[~bads], c=temp[~bads], s=scatter_size)

    if model_on_data:
        order = np.argsort(xx)
        ax_main.plot(xx[order], yhat[order], ",", color="red", alpha=0.8)

    if show_reference_curve:
        x_min = max(float(xx.min()), 1e-12)
        x_max = float(xx.max())
        xg = np.geomspace(x_min, x_max, 500)

        tref = (
            float(np.mean(temp))
            if reference_temperature is None
            else float(reference_temperature)
        )
        yg = model.evaluate(
            x=xg,
            ccd_temp=np.full_like(xg, tref),
            params=record.params,
        )
        ax_main.plot(xg, yg, "-", lw=1.5, color="black")

    if hasattr(model, "basis_grid"):
        for g in model.basis_grid:
            ax_main.axvline(g, ls=":", lw=0.5)
            ax_res_x.axvline(g, ls=":", lw=0.5)

    ax_main.set_xscale("log")
    ax_main.set_ylabel("overscan_sum")
    ax_main.tick_params(axis="x", labelbottom=False)
    ax_main.set_title(
        f"Equilibrium model"
        f"| ccdid={record.ccdid} qid={record.qid} "
        f"| mjd=[{record.mjd_start:.3f}, {record.mjd_end:.3f})"
    )

    # residuals vs skylev
    ax_res_x.scatter(xx[~bads], resid[~bads], c=temp[~bads], s=scatter_size)
    ax_res_x.axhline(0.0, ls="--", lw=0.8, color="k")
    ax_res_x.set_xscale("log")
    ax_res_x.set_ylabel("res.")
    ax_res_x.tick_params(axis="x", labelbottom=True)

    # residuals vs mjd
    ax_res_t.scatter(mjd[~bads], resid[~bads], c=temp[~bads], s=scatter_size)
    ax_res_t.axhline(0.0, ls="--", lw=0.8, color="k")
    ax_res_t.set_xlabel("mjd")
    ax_res_t.set_ylabel("res.")

    # common colorbar
    cbar = fig.colorbar(sc, ax=[ax_main, ax_res_x, ax_res_t], pad=0.02)
    cbar.set_label("cryotemp")

    # histogram of residuals
    ax_hist = inset_axes(ax_main, width="30%", height="40%", loc="lower right")
    ax_hist.hist(resid[~bads], bins=hist_bins, density=True)
    ax_hist.axvline(0.0, ls="--", lw=0.8, color="k")
    ax_hist.set_title("residuals", fontsize=9)
    ax_hist.tick_params(labelsize=8)
    ax_hist.set_yscale("log")

    return fig, {
        "ax_main": ax_main,
        "ax_res_x": ax_res_x,
        "ax_res_t": ax_res_t,
        "ax_hist": ax_hist,
    }


# ---------------------------------------------------------------------------
# Fit-diagnostics gallery
# ---------------------------------------------------------------------------


@dataclass
class FitGallery:
    diagnostics: list[FitDiagnostics] = field(default_factory=list)
    items: list[GalleryItem] = field(default_factory=list)

    def add(self, diag: FitDiagnostics) -> None:
        self.diagnostics.append(diag)

    def save_all(
        self,
        output_dir: str | Path,
        close_figures: bool = True,
        **plot_kwargs,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.items.clear()

        for diag in self.diagnostics:
            rec = diag.record

            fig, _ = diag.plot(**plot_kwargs)

            subdir = output_dir / f"ccdid_{rec.ccdid:02d}" / f"qid_{rec.qid:02d}"
            subdir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"fit_ccdid_{rec.ccdid:02d}"
                f"_qid_{rec.qid:02d}"
                f"_mjd_{rec.mjd_start:.3f}_{rec.mjd_end:.3f}.png"
            )
            path = subdir / filename

            fig.savefig(path, dpi=150, bbox_inches="tight")
            if close_figures:
                plt.close(fig)

            self.items.append(
                GalleryItem(
                    ccdid=rec.ccdid,
                    qid=rec.qid,
                    mjd_start=rec.mjd_start,
                    mjd_end=rec.mjd_end,
                    path=path,
                )
            )

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ccdid": [it.ccdid for it in self.items],
                "qid": [it.qid for it in self.items],
                "mjd_start": [it.mjd_start for it in self.items],
                "mjd_end": [it.mjd_end for it in self.items],
                "path": [str(it.path) for it in self.items],
            }
        )

    def write_index(self, output: str | Path) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if output.suffix == ".parquet":
            df.write_parquet(output)
        else:
            df.write_csv(output)

    def write_html(
        self,
        output: str | Path,
        thumb_width: int = 250,
        title: str = "Equilibrium Fit Gallery",
    ) -> None:
        """Write an HTML gallery.  Call :meth:`save_all` first."""
        _write_gallery_html(
            self.items,
            output=Path(output),
            title=title,
            thumb_width=thumb_width,
        )
