#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
PyEPICS command-line interface

Provides batch-processing commands for the full data pipeline:

1. **download** — Download ENDF files from LLNL
2. **raw**      — Create raw HDF5 files (original grids, breakpoints)
3. **mcdc**     — Create MCDC-format HDF5 files (common grid, PDFs)
4. **all**      — Run raw + mcdc for a set of libraries

Usage
-----
::

    # Download all libraries
    python -m pyepics.cli download

    # Create raw + MCDC for electrons only
    python -m pyepics.cli all --libraries electron

    # Create only MCDC data for all libraries
    python -m pyepics.cli mcdc

    # Process specific Z range
    python -m pyepics.cli all --z-min 1 --z-max 30

Directory structure after a full run::

    data/
        endf/
            eedl/           ← downloaded ENDF (EEDL)
            epdl/           ← downloaded ENDF (EPDL)
            eadl/           ← downloaded ENDF (EADL)
        raw/
            electron/       ← raw HDF5 (electron)
            photon/         ← raw HDF5 (photon)
            atomic/         ← raw HDF5 (atomic)
        mcdc/
            H.h5            ← combined MCDC (electron + photon + atomic)
            He.h5
            ...
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pyepics.utils.constants import PERIODIC_TABLE

logger = logging.getLogger("pyepics.cli")

# ---------------------------------------------------------------------------
# Library configuration
# ---------------------------------------------------------------------------

LIBRARY_CONFIG = {
    "electron": {
        "dataset_type": "EEDL",
        "endf_dir": "data/endf/eedl",
        "endf_prefix": "EEDL",
        "raw_dir": "data/raw/electron",
        "mcdc_dir": "data/mcdc/electron",
        "download_key": "eedl",
    },
    "photon": {
        "dataset_type": "EPDL",
        "endf_dir": "data/endf/epdl",
        "endf_prefix": "EPDL",
        "raw_dir": "data/raw/photon",
        "mcdc_dir": "data/mcdc/photon",
        "download_key": "epdl",
    },
    "atomic": {
        "dataset_type": "EADL",
        "endf_dir": "data/endf/eadl",
        "endf_prefix": "EADL",
        "raw_dir": "data/raw/atomic",
        "mcdc_dir": "data/mcdc/atomic",
        "download_key": "eadl",
    },
}


def _find_endf_file(endf_dir: Path, prefix: str, Z: int) -> Path | None:
    """Locate the ENDF file for a given Z in *endf_dir*."""
    pattern = f"{prefix}.ZA{Z:03d}000*"
    matches = list(endf_dir.glob(pattern))
    if matches:
        return matches[0]
    # Try alternate naming
    pattern2 = f"*ZA{Z:03d}000*"
    matches2 = list(endf_dir.glob(pattern2))
    return matches2[0] if matches2 else None


def _element_symbol(Z: int) -> str:
    """Get element symbol, fallback to Z number."""
    info = PERIODIC_TABLE.get(Z)
    if info:
        return info.get("symbol", f"Z{Z:03d}")
    return f"Z{Z:03d}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_download(args):
    """Download ENDF files from LLNL."""
    from pyepics.io.download import download_library

    base = Path(args.data_dir)
    libraries = args.libraries or list(LIBRARY_CONFIG.keys())

    for lib_name in libraries:
        cfg = LIBRARY_CONFIG[lib_name]
        out = base / cfg["endf_dir"]
        print(f"\nDownloading {lib_name} ({cfg['dataset_type']}) -> {out}")
        try:
            download_library(cfg["download_key"], out)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if not args.continue_on_error:
                return 1
    return 0


def cmd_raw(args):
    """Create raw HDF5 files from ENDF sources."""
    from pyepics.converters.hdf5 import create_raw_hdf5

    base = Path(args.data_dir)
    libraries = args.libraries or list(LIBRARY_CONFIG.keys())
    z_min, z_max = args.z_min, args.z_max

    total_ok = 0
    total_fail = 0

    for lib_name in libraries:
        cfg = LIBRARY_CONFIG[lib_name]
        endf_dir = base / cfg["endf_dir"]
        raw_dir = base / cfg["raw_dir"]
        raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Creating raw HDF5: {lib_name} ({cfg['dataset_type']})")
        print(f"  ENDF source:  {endf_dir}")
        print(f"  Output:       {raw_dir}")
        print(f"  Z range:      {z_min}–{z_max}")
        print(f"{'=' * 60}")

        for Z in range(z_min, z_max + 1):
            sym = _element_symbol(Z)
            endf_file = _find_endf_file(endf_dir, cfg["endf_prefix"], Z)
            if endf_file is None:
                continue

            out_path = raw_dir / f"{sym}.h5"
            print(
                f"  Z={Z:3d} ({sym:>2s}): {endf_file.name} -> {out_path.name}",
                end=" ... ",
                flush=True,
            )

            try:
                create_raw_hdf5(
                    cfg["dataset_type"],
                    endf_file,
                    out_path,
                    overwrite=args.overwrite,
                )
                print("OK")
                total_ok += 1
            except Exception as exc:
                print(f"FAIL: {exc}")
                total_fail += 1
                if not args.continue_on_error:
                    return 1

    print(f"\nRaw HDF5: {total_ok} OK, {total_fail} failed")
    return 0 if total_fail == 0 else 1


def cmd_mcdc(args):
    """Create combined MCDC-format HDF5 files (one per element).

    Each output file contains ``electron_reactions``,
    ``photon_reactions``, and ``atomic_relaxation`` groups — whichever
    ENDF sources are available for that element.
    """
    from pyepics.converters.hdf5 import create_combined_mcdc_hdf5

    base = Path(args.data_dir)
    z_min, z_max = args.z_min, args.z_max

    mcdc_dir = base / "data/mcdc"
    mcdc_dir.mkdir(parents=True, exist_ok=True)

    # ENDF directories
    eedl_dir = base / LIBRARY_CONFIG["electron"]["endf_dir"]
    epdl_dir = base / LIBRARY_CONFIG["photon"]["endf_dir"]
    eadl_dir = base / LIBRARY_CONFIG["atomic"]["endf_dir"]

    print(f"\n{'=' * 60}")
    print("  Creating combined MCDC HDF5 files")
    print(f"  Output:  {mcdc_dir}")
    print(f"  Z range: {z_min}–{z_max}")
    print(f"{'=' * 60}")

    total_ok = 0
    total_fail = 0

    for Z in range(z_min, z_max + 1):
        sym = _element_symbol(Z)
        eedl_file = _find_endf_file(eedl_dir, "EEDL", Z)
        epdl_file = _find_endf_file(epdl_dir, "EPDL", Z)
        eadl_file = _find_endf_file(eadl_dir, "EADL", Z)

        if not any([eedl_file, epdl_file, eadl_file]):
            continue

        libs = []
        if eedl_file:
            libs.append("EEDL")
        if epdl_file:
            libs.append("EPDL")
        if eadl_file:
            libs.append("EADL")

        out_path = mcdc_dir / f"{sym}.h5"
        print(
            f"  Z={Z:3d} ({sym:>2s}): {'+'.join(libs)} -> {out_path.name}",
            end=" ... ",
            flush=True,
        )

        try:
            create_combined_mcdc_hdf5(
                Z,
                out_path,
                eedl_path=eedl_file,
                epdl_path=epdl_file,
                eadl_path=eadl_file,
                overwrite=args.overwrite,
            )
            print("OK")
            total_ok += 1
        except Exception as exc:
            print(f"FAIL: {exc}")
            total_fail += 1
            if not args.continue_on_error:
                return 1

    print(f"\nMCDC HDF5: {total_ok} OK, {total_fail} failed")
    return 0 if total_fail == 0 else 1


def cmd_all(args):
    """Run raw + MCDC for selected libraries."""
    print("=" * 60)
    print("  Step 1/2: Creating raw HDF5 files")
    print("=" * 60)
    rc1 = cmd_raw(args)

    print()
    print("=" * 60)
    print("  Step 2/2: Creating MCDC HDF5 files")
    print("=" * 60)
    rc2 = cmd_mcdc(args)

    return max(rc1, rc2)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_common_parser() -> argparse.ArgumentParser:
    """Build the shared options parser used as a parent for every subcommand.

    Attaching the common flags to each subparser (rather than to the
    top-level parser) lets invocations follow the natural
    ``pyepics <command> --flag value`` order documented in the README.
    A top-level parent would otherwise refuse flags placed after the
    subcommand name.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        "-d",
        default=".",
        help="Base data directory (default: current directory)",
    )
    common.add_argument(
        "--libraries",
        "-l",
        nargs="+",
        choices=["electron", "photon", "atomic"],
        default=None,
        help="Libraries to process (default: all three)",
    )
    common.add_argument(
        "--z-min",
        type=int,
        default=1,
        help="Minimum atomic number (default: 1)",
    )
    common.add_argument(
        "--z-max",
        type=int,
        default=100,
        help="Maximum atomic number (default: 100)",
    )
    common.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    common.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing after errors",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pyepics",
        description="PyEPICS data pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m pyepics.cli download                         # download all
    python -m pyepics.cli download --libraries electron    # EEDL only
    python -m pyepics.cli download --libraries photon      # EPDL only
    python -m pyepics.cli download --libraries atomic      # EADL only
    python -m pyepics.cli raw                              # raw HDF5 for all
    python -m pyepics.cli mcdc --libraries electron        # MCDC electron only
    python -m pyepics.cli all --z-min 1 --z-max 30         # first 30 elements
    python -m pyepics.cli all --overwrite                  # overwrite existing
""",
    )

    common = _build_common_parser()

    sub = parser.add_subparsers(dest="command", help="Pipeline step to run")
    sub.add_parser(
        "download",
        parents=[common],
        help="Download ENDF files from LLNL",
    )
    sub.add_parser(
        "raw",
        parents=[common],
        help="Create raw HDF5 files",
    )
    sub.add_parser(
        "mcdc",
        parents=[common],
        help="Create MCDC-format HDF5 files",
    )
    sub.add_parser(
        "all",
        parents=[common],
        help="Run raw + mcdc (full pipeline)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # ``--verbose`` is attached to every subparser via the shared parent;
    # it is only available on *args* when a subcommand has been selected.
    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s"
        )
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()

    commands = {
        "download": cmd_download,
        "raw": cmd_raw,
        "mcdc": cmd_mcdc,
        "all": cmd_all,
    }

    rc = commands[args.command](args)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
