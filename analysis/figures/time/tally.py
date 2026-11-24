"""The numbers behind chart.html (next to this file), the time half of the pair with ../tokens/chart.html.

    python3 analysis/figures/time/tally.py            prints the time chart's DATA as JSON
    python3 analysis/figures/time/tally.py --write    also copies it into chart.html, between the DATA markers

The rules and the work live in ../tokens/tally.py, which both charts share (as they share ../tokens/shared.js).
"""
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # loading ../tokens/tally.py would otherwise leave a __pycache__ there
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pair_tally', HERE.parent / 'tokens' / 'tally.py')
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)
pair.main('time', HERE / 'chart.html')
