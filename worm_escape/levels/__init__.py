"""
worm_escape.levels — Built-in level data for Worm Escape.

Each sub-module exports a single LEVEL_DATA dict.  This __init__
aggregates them into the LEVELS list, ordered by level number.
To add a new level, create level_NN.py with a LEVEL_DATA dict
and append it to the LEVELS list below.
"""

from worm_escape.levels.level_01 import LEVEL_DATA as _L01
from worm_escape.levels.level_02 import LEVEL_DATA as _LEVEL_02

LEVELS = [
    _L01,
    _LEVEL_02,
]
