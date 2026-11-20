#!/usr/bin/env bash
# Copyright (C) 2026, Jose Luis Blanco-Claraco
# SPDX-License-Identifier: GPL-3.0-or-later
# Usage: scripts/formatter.sh [--check]
#   (default) Reformat all C/C++ sources in-place with clang-format-14.
#   --check   Dry-run: exit non-zero if any file would be reformatted.

set -euo pipefail

if [ "${1:-}" = "--check" ]; then
  MODE=(--dry-run --Werror)
else
  MODE=(-i)
fi

find \
    mola_libvision \
    mola_visual_tracking \
    \( -iname "*.h" -o -iname "*.hpp" -o -iname "*.cpp" -o -iname "*.c" \) \
  -print0 | xargs -0 -r -t clang-format-14 "${MODE[@]}"
