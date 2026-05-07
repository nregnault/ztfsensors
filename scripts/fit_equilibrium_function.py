#!/usr/bin/env python

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
import yaml
from astropy.time import Time

from ztfsensors.pocket import FitResults, fit_eq_model
from ztfsensors.pocket.db import EqFuncDb
from ztfsensors.pocket.fit import FitDiagnosticsDb
from ztfsensors.pocket.models import PolyTempEqModel, SplineTempEqModel
from ztfsensors.pocket.plots import EqFuncGallery, FitGallery, FitGalleryItem

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def load(
    fn: Path,
    ccdid: int = 6,
    qid: int = 1,
    mjd_range: tuple[float, float] | None = None,
    night: str | None = None,
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
            # in the early days of the ZTF survey (before 2018-11)
            # the overscans are only 29 column-wide.
            (
                (pl.col("overscan_val") - pl.col("pedestal1"))  # was pedestal_avg
                .filter(pl.col("j_overscan") >= 0)
                .sum()
                .alias("overscan_sum")
            ),
            (pl.col("last_val") - pl.col("pedestal1"))  # was pedestal_avg
            .median()
            .alias("last_col_skylev"),
        ]
    )

    return dd.collect()


def parse_mjd_range(value: str) -> tuple[float, float]:
    """Parse MJD range from string format 'START:END' or 'YYYY-MM-DD:YYYY-MM-DD'."""
    start_str, end_str = value.split(":")

    # Try to parse as float first, then as date
    try:
        start = float(start_str)
        end = float(end_str)
    except ValueError:
        start = Time(start_str).mjd
        end = Time(end_str).mjd

    return (start, end)


def parse_mjd_ranges(value: str) -> list[tuple[float, float]]:
    """Parse multiple MJD ranges from comma-separated list."""
    return [parse_mjd_range(r.strip()) for r in value.split(",")]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def setup_matplotlib_backend(backend_mode: str) -> None:
    """Configure matplotlib backend.

    Args:
        backend_mode: 'interactive', 'agg', or 'auto' (default)
    """
    if backend_mode == "agg":
        matplotlib.use("Agg")
        logging.info("Using matplotlib backend: Agg (non-interactive)")
    elif backend_mode == "interactive":
        # Force a GUI backend if available
        try:
            matplotlib.use("TkAgg")
            logging.info("Using matplotlib backend: TkAgg (interactive)")
        except ImportError:
            try:
                matplotlib.use("Qt5Agg")
                logging.info("Using matplotlib backend: Qt5Agg (interactive)")
            except ImportError:
                logging.warning(
                    "No interactive backend available, falling back to default"
                )
    # else: auto, use default backend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit equilibrium function to ZTF sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit single MJD range
  %(prog)s --mjd-range 2019-10-23:2019-12-31 -O output/

  # Fit multiple MJD ranges
  %(prog)s --mjd-ranges "2019-10-23:2019-12-31,2020-01-01:2020-12-31" -O output/

  # Use config file
  %(prog)s --config config.yaml -O output/

  # Non-interactive plotting (for servers)
  %(prog)s --backend agg --mjd-range 2019-10-23:2019-12-31 -O output/
        """,
    )

    # Input data
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Input parquet file with training data (default: pocket_training_all_fields.parquet)",
    )

    # MJD range options (mutually exclusive with mjd-ranges)
    mjd_group = parser.add_mutually_exclusive_group()
    mjd_group.add_argument(
        "--mjd-range",
        type=str,
        help="Single MJD range as 'START:END' (can use MJD floats or dates YYYY-MM-DD)",
    )
    mjd_group.add_argument(
        "--mjd-ranges",
        type=str,
        help="Multiple MJD ranges as comma-separated list: 'START1:END1,START2:END2,...'",
    )

    # CCD/QID selection
    parser.add_argument(
        "--ccdids",
        type=str,
        help="CCD IDs to process (e.g., '1-16', '1,3,5', '6') (default: '1-16')",
    )
    parser.add_argument(
        "--qids",
        type=str,
        help="Quadrant IDs to process (e.g., '1-4', '1,3', '1') (default: '1')",
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["poly", "spline"],
        help="Equilibrium model type (default: poly)",
    )
    parser.add_argument(
        "--spline-knots",
        type=int,
        help="Number of knots for spline model (default: 7)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        "-O",
        type=Path,
        help="Output directory for results (default: prod)",
    )

    # Matplotlib backend options
    parser.add_argument(
        "--backend",
        choices=["auto", "interactive", "agg"],
        help="Matplotlib backend mode: auto (default), interactive (GUI), or agg (non-interactive)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively and wait for user to close them (implies --no-close-figures)",
    )
    parser.add_argument(
        "--no-close-figures",
        action="store_true",
        help="Don't close figures after saving (keeps them in memory)",
    )
    parser.add_argument(
        "--pause-after-each",
        action="store_true",
        help="Pause and wait for 'q' or Enter after each fit plot (useful for reviewing)",
    )

    # Config file
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="YAML configuration file (overrides command-line options)",
    )
    parser.add_argument(
        "--save-config",
        type=Path,
        help="Save current configuration to YAML file and exit",
    )

    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")

    # Helper function: command-line args override config file
    def get_value(arg_value, config_key, default):
        """Get value with priority: CLI arg > config file > default.

        If arg is None, it means it wasn't provided on CLI.
        """
        # If argument was explicitly provided on CLI, use it
        if arg_value is not None:
            return arg_value
        # Otherwise, check config file
        if config_key in config:
            return config[config_key]
        # Finally, use the default
        return default

    # Defaults
    defaults = {
        "input": Path(
            "/home/nrl/lemaitre/ztf-sensor-analysis/pocket/training/data/pocket_training_all_fields.parquet"
        ),
        "output_dir": Path("prod"),
        "model": "poly",
        "spline_knots": 7,
        "backend": "auto",
        "ccdids": "1-16",
        "qids": "1",
        "show_plots": False,
        "no_close_figures": False,
        "pause_after_each": False,
    }

    # Apply priority: CLI > config > defaults
    input_file = get_value(args.input, "input", defaults["input"])
    output_dir = Path(get_value(args.output_dir, "output_dir", defaults["output_dir"]))
    model_type = get_value(args.model, "model", defaults["model"])
    spline_knots = get_value(
        args.spline_knots, "spline_knots", defaults["spline_knots"]
    )
    backend_mode = get_value(args.backend, "backend", defaults["backend"])
    ccdids_str = get_value(args.ccdids, "ccdids", defaults["ccdids"])
    qids_str = get_value(args.qids, "qids", defaults["qids"])
    show_plots = get_value(args.show_plots, "show_plots", defaults["show_plots"])
    no_close_figures = get_value(
        args.no_close_figures, "no_close_figures", defaults["no_close_figures"]
    )
    pause_after_each = get_value(
        args.pause_after_each, "pause_after_each", defaults["pause_after_each"]
    )

    # Parse MJD ranges
    mjd_ranges = []
    if args.config and "mjd_ranges" in config:
        mjd_ranges = [(r["start"], r["end"]) for r in config["mjd_ranges"]]
    elif args.mjd_ranges:
        mjd_ranges = parse_mjd_ranges(args.mjd_ranges)
    elif args.mjd_range:
        mjd_ranges = [parse_mjd_range(args.mjd_range)]
    else:
        # Default range
        mjd_ranges = [(Time("2019-10-23").mjd, Time("2019-12-31").mjd)]

    # Parse CCD and QID ranges
    def parse_range(s: str) -> list[int]:
        """Parse range string like '1-16' or '1,3,5' into list of ints."""
        result = []
        for part in s.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))
        return result

    ccdids = parse_range(ccdids_str)
    qids = parse_range(qids_str)

    # Handle plot display logic
    if show_plots:
        no_close_figures = True  # --show-plots implies --no-close-figures
        if backend_mode == "agg":
            logging.warning(
                "--show-plots used with --backend agg. Switching to interactive backend."
            )
            backend_mode = "interactive"
    elif backend_mode == "auto":
        # If no interactive plotting requested and backend is auto, use Agg
        # to avoid opening windows
        if not pause_after_each:
            backend_mode = "agg"
            logging.info("No interactive plotting requested, using Agg backend")

    close_figures = not no_close_figures

    # Save config if requested
    if args.save_config:
        config_to_save = {
            "input": str(input_file),
            "output_dir": str(output_dir),
            "model": model_type,
            "spline_knots": spline_knots,
            "backend": backend_mode,
            "ccdids": ccdids_str,
            "qids": qids_str,
            "show_plots": show_plots,
            "no_close_figures": no_close_figures,
            "pause_after_each": pause_after_each,
            "mjd_ranges": [
                {"start": float(start), "end": float(end)} for start, end in mjd_ranges
            ],
        }
        save_config(config_to_save, args.save_config)
        logging.info(f"Configuration saved to {args.save_config}")
        sys.exit(0)

    # Setup matplotlib backend BEFORE importing pyplot
    setup_matplotlib_backend(backend_mode)

    # Import pyplot if needed (we'll use lazy import in interactive mode)
    plt = None
    if show_plots or pause_after_each:
        import matplotlib.pyplot as plt

    # Create equilibrium model
    if model_type == "spline":
        basis_grid = np.geomspace(60.0, 10000.0, spline_knots)
        eq_model = SplineTempEqModel(
            basis_grid=basis_grid, temp_deg=3, temp_ref=160.0, temp_scale=1.0
        )
        logging.info(f"Using SplineTempEqModel with {spline_knots} knots")
    else:
        eq_model = PolyTempEqModel()
        logging.info("Using PolyTempEqModel")

    # Process each MJD range
    global_fits = FitResults(model=eq_model)  # eq_db globale, unique
    global_gallery = FitGallery()  # Galerie globale pour l'index HTML final
    global_diags: list = []  # Tous les FitDiagnostics, pour FitDiagnosticsDb

    for mjd_idx, (mjd_start, mjd_end) in enumerate(mjd_ranges):
        logging.info(f"\n{'=' * 60}")
        logging.info(
            f"Processing MJD range {mjd_idx + 1}/{len(mjd_ranges)}: [{mjd_start:.3f}, {mjd_end:.3f})"
        )
        logging.info(f"{'=' * 60}")

        fits = FitResults(
            model=eq_model
        )  # Résultats pour cette plage MJD uniquement (diagnostics)
        gallery = FitGallery()

        for ccdid in ccdids:
            for qid in qids:
                logging.info(f"Processing: ccdid={ccdid} qid={qid}")
                train_df = load(
                    Path(input_file),
                    ccdid=ccdid,
                    qid=qid,
                    mjd_range=(mjd_start, mjd_end),
                )

                if len(train_df) == 0:
                    logging.warning(
                        f"No data for ccdid={ccdid} qid={qid} in MJD range [{mjd_start:.3f}, {mjd_end:.3f})"
                    )
                    continue

                record, diag = fit_eq_model(
                    train_df,
                    ccdid=ccdid,
                    qid=qid,
                    eq_model=eq_model,
                    mjd_start=mjd_start,
                    mjd_end=mjd_end,
                )

                fits.append(record)
                global_fits.append(record)
                global_diags.append(diag)
                gallery.add(diag)

        # Save results for this MJD range
        mjd_suffix = f"mjd_{mjd_start:.1f}_{mjd_end:.1f}"
        gallery_dir = output_dir / "training_gallery" / mjd_suffix

        # Save gallery with interactive display if requested
        if show_plots or pause_after_each:
            # Lazy import if not already done
            if plt is None:
                import matplotlib.pyplot as plt

            # Display plots interactively during save
            output_dir_path = Path(gallery_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            gallery.items.clear()

            for diag in gallery.diagnostics:
                rec = diag.record

                fig, _ = diag.plot()

                # Show interactively
                plt.show(block=False)
                plt.pause(0.1)

                # Save to file
                subdir = (
                    output_dir_path / f"ccdid_{rec.ccdid:02d}" / f"qid_{rec.qid:02d}"
                )
                subdir.mkdir(parents=True, exist_ok=True)
                filename = (
                    f"fit_ccdid_{rec.ccdid:02d}"
                    f"_qid_{rec.qid:02d}"
                    f"_mjd_{rec.mjd_start:.3f}_{rec.mjd_end:.3f}.png"
                )
                path = subdir / filename
                fig.savefig(path, dpi=150, bbox_inches="tight")

                # Handle interactive pauses
                if pause_after_each:
                    print(
                        f"\n[Plot displayed for ccdid={rec.ccdid} qid={rec.qid}] "
                        "Press Enter to continue (or 'q' + Enter to skip remaining)... ",
                        end="",
                        flush=True,
                    )
                    response = input()
                    if response.lower().strip() == "q":
                        logging.info("User requested to skip interactive plotting")
                        pause_after_each = False
                        plt.close(fig)
                    elif not show_plots:
                        plt.close(fig)
                elif not show_plots:
                    # In pause_after_each mode but user already quit
                    plt.close(fig)
                elif not close_figures:
                    pass  # Keep figure open
                # Note: if show_plots and close_figures, we keep them open

                item = FitGalleryItem(
                    ccdid=rec.ccdid,
                    qid=rec.qid,
                    mjd_start=rec.mjd_start,
                    mjd_end=rec.mjd_end,
                    path=path,
                )
                gallery.items.append(item)
                global_gallery.items.append(item)
        else:
            # Normal non-interactive save
            gallery.save_all(gallery_dir, close_figures=close_figures)
            for item in gallery.items:
                global_gallery.items.append(item)

        # Save per-MJD-range index (optionnel, peut être supprimé si non désiré)
        gallery.write_index(gallery_dir / "index.parquet")

        logging.info(f"Results saved to {output_dir}")

        # Wait for user to close all plots at the end of each MJD range
        if show_plots and plt is not None:
            print(
                "\n[All plots displayed] Press Enter to continue to next MJD range (or close plot windows)... ",
                end="",
                flush=True,
            )
            input()
            plt.close("all")

    # Sauvegarder l'eq_db globale unique
    logging.info("\n" + "=" * 60)
    logging.info("Saving global eq_db...")
    logging.info("=" * 60)

    global_fits.save(output_dir / "eq_db")
    logging.info(f"Global eq_db saved to {output_dir / 'eq_db'}")

    # Sauvegarder les diagnostics de fit (données brutes + résidus + bads)
    logging.info("\n" + "=" * 60)
    logging.info("Saving fit diagnostics...")
    logging.info("=" * 60)

    FitDiagnosticsDb.from_diagnostics(global_diags).save(output_dir / "eq_diag.parquet")
    logging.info(f"Fit diagnostics saved to {output_dir / 'eq_diag.parquet'}")

    # Générer la galerie des fonctions d'équilibre tabulées
    logging.info("\n" + "=" * 60)
    logging.info("Generating eq_func gallery...")
    logging.info("=" * 60)

    eq_db = EqFuncDb.open(output_dir / "eq_db")
    eq_func_gallery = EqFuncGallery(eq_db)
    eq_func_gallery_dir = output_dir / "eq_func_gallery"
    eq_func_gallery.save_all(eq_func_gallery_dir, close_figures=True)
    eq_func_gallery.write_html(
        eq_func_gallery_dir / "index.html",
        title="Equilibrium Function Gallery",
    )
    logging.info(f"Eq func gallery saved to {eq_func_gallery_dir / 'index.html'}")

    # Générer la galerie HTML globale avec toutes les plages MJD
    logging.info("\n" + "=" * 60)
    logging.info("Generating global HTML gallery...")
    logging.info("=" * 60)

    global_gallery_dir = output_dir / "training_gallery"
    global_gallery.write_index(global_gallery_dir / "index.parquet")
    global_gallery.write_html(global_gallery_dir / "index.html")

    logging.info(f"Global gallery saved to {global_gallery_dir / 'index.html'}")

    logging.info("\n" + "=" * 60)
    logging.info("All processing complete!")
    logging.info("=" * 60)
