from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from astropy.time import Time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..fit import FitDiagnostics

READOUT_UPGRADES = [
    Time("2018-06-26").mjd,
    Time("2019-10-22").mjd,
    Time("2020-04-29").mjd,
    Time("2022-10-04").mjd,
]


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


@dataclass
class FitGalleryItem:
    ccdid: int
    qid: int
    mjd_start: float
    mjd_end: float
    path: Path


@dataclass
class FitGallery:
    diagnostics: list[FitDiagnostics] = field(default_factory=list)
    items: list[FitGalleryItem] = field(default_factory=list)

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
                FitGalleryItem(
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
        """
        Write an HTML gallery.

        Layout: one (ccdid, qid) pair per row, one MJD period per column.
        Each cell contains a thumbnail image (clickable for full resolution).
        Missing combinations appear as an empty cell.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if not self.items:
            raise ValueError("No items in gallery. Call save_all() first.")

        html_dir = output.parent

        # Unique rows (ccdid, qid) and columns (mjd periods), both sorted
        ccd_qids = sorted(set((it.ccdid, it.qid) for it in self.items))
        # Group by mjd_start only to avoid duplicate columns
        # Build a mapping of mjd_start -> (mjd_start, mjd_end) for display
        mjd_periods_map = {}
        for it in self.items:
            if it.mjd_start not in mjd_periods_map:
                mjd_periods_map[it.mjd_start] = (it.mjd_start, it.mjd_end)
        periods = sorted(mjd_periods_map.values(), key=lambda p: p[0])

        # Lookup: (ccdid, qid, mjd_start) -> Path
        lookup: dict[tuple[int, int, float], Path] = {
            (it.ccdid, it.qid, it.mjd_start): it.path for it in self.items
        }

        def rel(p: Path) -> str:
            try:
                return str(p.relative_to(html_dir))
            except ValueError:
                return str(p)

        # Column headers — MJD start en gros, end en petit
        col_headers = "\n".join(
            f'          <th title="[{s:.3f}, {e:.3f})">'
            f"{s:.1f}<br>"
            f'<span class="sub">–{e:.1f}</span></th>'
            for s, e in periods
        )

        # Table rows
        rows = []
        for ccdid, qid in ccd_qids:
            cells = [f"          <th>ccd{ccdid:02d}<br>q{qid}</th>"]
            for mjd_start, _mjd_end in periods:
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
            rows.append("        <tr>\n" + "\n".join(cells) + "\n        </tr>")

        rows_html = "\n".join(rows)

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
<html lang="fr">
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
{rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""

        output.write_text(html, encoding="utf-8")
