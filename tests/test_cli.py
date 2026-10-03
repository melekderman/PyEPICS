#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Tests for :mod:`pyepics.cli`

Covers the argument parser and the ``download`` subcommand wiring.  The
download tests monkey-patch :func:`pyepics.io.download.download_library`
so no network requests are issued; only the routing from CLI flags to
the underlying library key (``eedl`` / ``epdl`` / ``eadl``) is
exercised.

These tests guard against regressions in the README-documented
``pyepics <command> --flag value`` invocation order, which had
previously been broken by attaching the shared options to the
top-level parser instead of the subparsers.
"""

from __future__ import annotations

import pytest

from pyepics import cli

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


class TestArgumentParser:
    """Parsing of the documented invocation patterns"""

    @pytest.fixture
    def parser(self):
        return cli.build_parser()

    def test_no_subcommand(self, parser) -> None:
        args = parser.parse_args([])
        assert args.command is None

    @pytest.mark.parametrize("command", ["download", "raw", "mcdc", "all"])
    def test_subcommand_no_flags(self, parser, command: str) -> None:
        args = parser.parse_args([command])
        assert args.command == command
        assert args.libraries is None
        assert args.z_min == 1
        assert args.z_max == 100
        assert args.overwrite is False
        assert args.continue_on_error is False
        assert args.verbose is False
        assert args.data_dir == "."

    @pytest.mark.parametrize(
        "command",
        ["download", "raw", "mcdc", "all"],
    )
    @pytest.mark.parametrize(
        "library",
        ["electron", "photon", "atomic"],
    )
    def test_libraries_flag_after_subcommand(
        self,
        parser,
        command: str,
        library: str,
    ) -> None:
        """The README pattern ``cmd --libraries <name>`` must parse"""
        args = parser.parse_args([command, "--libraries", library])
        assert args.command == command
        assert args.libraries == [library]

    def test_libraries_short_flag(self, parser) -> None:
        args = parser.parse_args(["download", "-l", "photon"])
        assert args.libraries == ["photon"]

    def test_multiple_libraries(self, parser) -> None:
        args = parser.parse_args(
            ["download", "--libraries", "photon", "atomic"],
        )
        assert args.libraries == ["photon", "atomic"]

    def test_z_range_flags(self, parser) -> None:
        args = parser.parse_args(
            ["raw", "--z-min", "5", "--z-max", "10"],
        )
        assert args.z_min == 5
        assert args.z_max == 10

    def test_overwrite_and_continue(self, parser) -> None:
        args = parser.parse_args(
            ["all", "--overwrite", "--continue-on-error"],
        )
        assert args.overwrite is True
        assert args.continue_on_error is True

    def test_data_dir_flag(self, parser) -> None:
        args = parser.parse_args(["download", "--data-dir", "/tmp/data"])
        assert args.data_dir == "/tmp/data"

    def test_verbose_flag(self, parser) -> None:
        args = parser.parse_args(["download", "--verbose"])
        assert args.verbose is True

    def test_unknown_library_rejected(self, parser) -> None:
        with pytest.raises(SystemExit):
            parser.parse_args(["download", "--libraries", "neutron"])

    def test_unknown_subcommand_rejected(self, parser) -> None:
        with pytest.raises(SystemExit):
            parser.parse_args(["bogus"])


# ---------------------------------------------------------------------------
# Download subcommand routing
# ---------------------------------------------------------------------------


class TestDownloadRouting:
    """The ``download`` subcommand must call ``download_library`` with
    the right ``(key, out_dir)`` pair for each library selection."""

    @pytest.fixture
    def calls(self, monkeypatch) -> list[tuple[str, str]]:
        recorded: list[tuple[str, str]] = []

        def fake_download(name: str, out_dir=None) -> None:
            recorded.append((name, str(out_dir)))

        monkeypatch.setattr(
            "pyepics.io.download.download_library",
            fake_download,
        )
        return recorded

    def test_default_downloads_all_three(
        self,
        calls: list[tuple[str, str]],
    ) -> None:
        rc = cli.main(["download", "--data-dir", "/tmp/x"])
        assert rc == 0
        keys = [name for name, _ in calls]
        assert keys == ["eedl", "epdl", "eadl"]

    def test_photon_only_routes_to_epdl(
        self,
        calls: list[tuple[str, str]],
    ) -> None:
        cli.main(["download", "--libraries", "photon", "--data-dir", "/tmp/x"])
        assert len(calls) == 1
        name, out_dir = calls[0]
        assert name == "epdl"
        assert out_dir.endswith("data/endf/epdl")

    def test_atomic_only_routes_to_eadl(
        self,
        calls: list[tuple[str, str]],
    ) -> None:
        cli.main(["download", "--libraries", "atomic", "--data-dir", "/tmp/x"])
        assert len(calls) == 1
        name, out_dir = calls[0]
        assert name == "eadl"
        assert out_dir.endswith("data/endf/eadl")

    def test_electron_only_routes_to_eedl(
        self,
        calls: list[tuple[str, str]],
    ) -> None:
        cli.main(["download", "--libraries", "electron", "--data-dir", "/tmp/x"])
        assert len(calls) == 1
        name, out_dir = calls[0]
        assert name == "eedl"
        assert out_dir.endswith("data/endf/eedl")

    def test_continue_on_error(
        self,
        monkeypatch,
        calls: list[tuple[str, str]],
    ) -> None:
        """When ``--continue-on-error`` is set, a failing library does
        not stop the others from being attempted."""
        attempts: list[str] = []

        def failing_download(name: str, out_dir=None) -> None:
            attempts.append(name)
            if name == "eedl":
                raise RuntimeError("simulated network failure")

        monkeypatch.setattr(
            "pyepics.io.download.download_library",
            failing_download,
        )

        rc = cli.main(
            ["download", "--continue-on-error", "--data-dir", "/tmp/x"],
        )
        # All three should have been attempted despite the eedl failure
        assert attempts == ["eedl", "epdl", "eadl"]
        assert rc == 0

    def test_abort_on_error_by_default(self, monkeypatch) -> None:
        """Without ``--continue-on-error``, the first failure aborts."""
        attempts: list[str] = []

        def failing_download(name: str, out_dir=None) -> None:
            attempts.append(name)
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "pyepics.io.download.download_library",
            failing_download,
        )

        rc = cli.main(["download", "--data-dir", "/tmp/x"])
        assert attempts == ["eedl"]
        assert rc == 1


# ---------------------------------------------------------------------------
# raw / mcdc / all library selection (command behavior, not just parsing)
# ---------------------------------------------------------------------------


class TestLibrarySelectionBehavior:
    """The ``--libraries`` flag must restrict which libraries are actually
    processed by ``raw``, ``mcdc``, and ``all`` — not merely parse.

    Regression guard for a bug where ``cmd_mcdc`` ignored
    ``--libraries`` entirely and always combined EEDL + EPDL + EADL.
    """

    @pytest.fixture
    def data_dir(self, tmp_path):
        """A data directory with placeholder ENDF files for all three
        libraries at Z=1, so file discovery succeeds for every library."""
        for sub, prefix in [
            ("data/endf/eedl", "EEDL"),
            ("data/endf/epdl", "EPDL"),
            ("data/endf/eadl", "EADL"),
        ]:
            d = tmp_path / sub
            d.mkdir(parents=True)
            (d / f"{prefix}.ZA001000.endf").write_text("placeholder")
        return tmp_path

    @pytest.fixture
    def raw_calls(self, monkeypatch):
        """Record the dataset types passed to ``create_raw_hdf5``."""
        recorded: list[str] = []

        def fake_raw(dataset_type, *args, **kwargs):
            recorded.append(dataset_type)

        monkeypatch.setattr(
            "pyepics.converters.hdf5.create_raw_hdf5",
            fake_raw,
        )
        return recorded

    @pytest.fixture
    def mcdc_calls(self, monkeypatch):
        """Record which library paths reach ``create_combined_mcdc_hdf5``."""
        recorded: list[dict] = []

        def fake_combined(
            Z,
            out_path,
            *,
            eedl_path=None,
            epdl_path=None,
            eadl_path=None,
            validate=True,
            overwrite=False,
        ):
            recorded.append(
                {
                    "Z": Z,
                    "eedl": eedl_path is not None,
                    "epdl": epdl_path is not None,
                    "eadl": eadl_path is not None,
                }
            )

        monkeypatch.setattr(
            "pyepics.converters.hdf5.create_combined_mcdc_hdf5",
            fake_combined,
        )
        return recorded

    # -- raw -----------------------------------------------------------

    def test_raw_electron_only(self, data_dir, raw_calls) -> None:
        rc = cli.main(
            ["raw", "--libraries", "electron", "--data-dir", str(data_dir),
             "--z-min", "1", "--z-max", "1"]
        )
        assert rc == 0
        assert raw_calls == ["EEDL"]

    def test_raw_default_all_three(self, data_dir, raw_calls) -> None:
        rc = cli.main(
            ["raw", "--data-dir", str(data_dir), "--z-min", "1", "--z-max", "1"]
        )
        assert rc == 0
        assert raw_calls == ["EEDL", "EPDL", "EADL"]

    # -- mcdc ----------------------------------------------------------

    def test_mcdc_electron_only(self, data_dir, mcdc_calls) -> None:
        """``mcdc --libraries electron`` must combine ONLY electron data."""
        rc = cli.main(
            ["mcdc", "--libraries", "electron", "--data-dir", str(data_dir),
             "--z-min", "1", "--z-max", "1"]
        )
        assert rc == 0
        assert mcdc_calls == [{"Z": 1, "eedl": True, "epdl": False, "eadl": False}]

    def test_mcdc_photon_atomic(self, data_dir, mcdc_calls) -> None:
        rc = cli.main(
            ["mcdc", "--libraries", "photon", "atomic", "--data-dir", str(data_dir),
             "--z-min", "1", "--z-max", "1"]
        )
        assert rc == 0
        assert mcdc_calls == [{"Z": 1, "eedl": False, "epdl": True, "eadl": True}]

    def test_mcdc_default_all_three(self, data_dir, mcdc_calls) -> None:
        """With no ``--libraries`` all available libraries are combined."""
        rc = cli.main(
            ["mcdc", "--data-dir", str(data_dir), "--z-min", "1", "--z-max", "1"]
        )
        assert rc == 0
        assert mcdc_calls == [{"Z": 1, "eedl": True, "epdl": True, "eadl": True}]

    # -- all (selection must survive both stages) ----------------------

    def test_all_electron_only_both_stages(
        self, data_dir, raw_calls, mcdc_calls
    ) -> None:
        """``all --libraries electron`` must restrict BOTH raw and mcdc."""
        rc = cli.main(
            ["all", "--libraries", "electron", "--data-dir", str(data_dir),
             "--z-min", "1", "--z-max", "1"]
        )
        assert rc == 0
        assert raw_calls == ["EEDL"]
        assert mcdc_calls == [{"Z": 1, "eedl": True, "epdl": False, "eadl": False}]
