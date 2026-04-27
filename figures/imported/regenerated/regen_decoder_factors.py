#!/usr/bin/env python3
"""Thin shim that delegates to the canonical regen script in interpertable/.

The canonical implementation lives at:
    /Users/morita/Desktop/MELON/interpertable/regen_decoder_factors.py

We import its main() and pass the paper-folder figure directory as the
output location, so figures land next to this file. Edit the canonical
file (in interpertable/) to change behaviour; this shim should rarely
need to change.
"""
import sys
from pathlib import Path

INTERP_DIR = Path("/Users/morita/Desktop/MELON/interpertable")
if str(INTERP_DIR) not in sys.path:
    sys.path.insert(0, str(INTERP_DIR))

import regen_decoder_factors as canonical  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    canonical.main(out_dir=OUT_DIR)
