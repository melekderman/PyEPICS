#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 CARRE
#
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

"""
Format-agnostic physics shared by the converters and the multigroup package

Modules here depend only on :mod:`pyepics.utils`, so both
:mod:`pyepics.converters` and :mod:`pyepics.multigroup` can import them
without importing each other.

elastic
    Screened Rutherford elastic scattering.
"""
