#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Melek Derman
#
# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------

"""
Shared utilities for parsing and validation

This sub-package centralises all low-level ENDF parsing helpers and
post-parse validation routines so that no logic is duplicated across
the three reader modules.
"""

from __future__ import annotations
