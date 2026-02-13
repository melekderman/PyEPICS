#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
EPICS dataset downloader

Downloads EEDL, EPDL, and EADL ENDF files from the IAEA Nuclear Data
Services website.

Data Sources
------------
* EEDL: ``https://www-nds.iaea.org/epics/ENDF2023/EEDL.ELEMENTS/``
* EPDL: ``https://www-nds.iaea.org/epics/ENDF2023/EPDL.ELEMENTS/``
* EADL: ``https://www-nds.iaea.org/epics/ENDF2023/EADL.ELEMENTS/``

Examples
--------
>>> from pyepics.io.download import download_library, download_all
>>> download_library("eedl")           # downloads to ./eedl/
>>> download_all(out_dir="data")       # downloads all three to data/{lib}/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

from pyepics.exceptions import DownloadError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Library metadata
# ---------------------------------------------------------------------------

LIBRARY_URLS: dict[str, dict[str, str]] = {
    "eedl": {
        "url": "https://www-nds.iaea.org/epics/ENDF2023/EEDL.ELEMENTS/getza.htm",
        "prefix": "EEDL",
        "description": "Evaluated Electron Data Library",
    },
    "epdl": {
        "url": "https://www-nds.iaea.org/epics/ENDF2023/EPDL.ELEMENTS/getza.htm",
        "prefix": "EPDL",
        "description": "Evaluated Photon Data Library",
    },
    "eadl": {
        "url": "https://www-nds.iaea.org/epics/ENDF2023/EADL.ELEMENTS/getza.htm",
        "prefix": "EADL",
        "description": "Evaluated Atomic Data Library",
    },
}
"""Download URLs and metadata for each EPICS library."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_library(
    library_name: Literal["eedl", "epdl", "eadl"],
    out_dir: Path | str | None = None,
) -> Path:
    """Download a specific EPICS library from the IAEA website

    Fetches the IAEA index page for the requested library, discovers all
    element ENDF files, and downloads each one to *out_dir*.

    Parameters
    ----------
    library_name : ``"eedl"`` | ``"epdl"`` | ``"eadl"``
        Which library to download.
    out_dir : Path | str | None, optional
        Output directory.  Defaults to ``"./{library_name}"``.

    Returns
    -------
    Path
        Path to the directory containing the downloaded files.

    Raises
    ------
    DownloadError
        If network requests fail or the HTML cannot be parsed.
    ValueError
        If *library_name* is not one of the supported values.
    """
    if library_name not in LIBRARY_URLS:
        raise ValueError(
            f"Unknown library: {library_name!r}.  "
            f"Choose from: {sorted(LIBRARY_URLS.keys())}"
        )

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise DownloadError(
            "Download requires 'requests' and 'beautifulsoup4'.  "
            "Install with: pip install requests beautifulsoup4"
        ) from exc

    config = LIBRARY_URLS[library_name]
    base_url = config["url"]
    prefix = config["prefix"]

    if out_dir is None:
        out_dir = Path(library_name)
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %s (%s) from %s",
        config["description"], prefix, base_url,
    )
    print(f"\n{'=' * 60}")
    print(f"  Downloading {config['description']} ({prefix})")
    print(f"{'=' * 60}")

    try:
        resp = requests.get(base_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Failed to fetch index page: {exc}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.find_all("a", href=True)

    if not links:
        raise DownloadError(
            f"No download links found on {base_url}.  "
            "The IAEA page format may have changed."
        )

    n_downloaded = 0
    for a in links:
        href = a["href"]
        url = urljoin(base_url, href)
        fname = f"{prefix}.{Path(href).name}.endf"
        dst = out_dir / fname

        logger.debug("Downloading %s -> %s", url, dst)
        print(f"  {fname}", end=" ... ", flush=True)
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            print("OK")
            n_downloaded += 1
        except requests.RequestException as exc:
            print(f"FAILED: {exc}")
            logger.warning("Failed to download %s: %s", url, exc)

    print(f"\n  Downloaded {n_downloaded} files to {out_dir}/")
    logger.info("Downloaded %d files to %s", n_downloaded, out_dir)
    return out_dir


def download_all(out_dir: Path | str | None = None) -> dict[str, Path]:
    """Download all three EPICS libraries (EEDL, EPDL, EADL)

    Parameters
    ----------
    out_dir : Path | str | None, optional
        Parent output directory.  Each library will be placed in a
        sub-folder (``eedl/``, ``epdl/``, ``eadl/``).  Defaults to
        the current working directory.

    Returns
    -------
    dict[str, Path]
        Mapping from library name to download directory.
    """
    parent = Path(out_dir) if out_dir else Path.cwd()
    results: dict[str, Path] = {}
    for name in LIBRARY_URLS:
        results[name] = download_library(name, parent / name)
    return results
