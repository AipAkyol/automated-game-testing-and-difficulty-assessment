"""
entities.py — Game entity classes for Worm Escape.
"""


class Worm:
    """
    A single worm entity on the grid.

    Attributes
    ----------
    worm_id   : str   – single uppercase letter, e.g. 'A'.
    segments  : list  – [(row, col), ...] ordered **tail → head**.
                        segments[-1] is always the HEAD.
    direction : str   – 'up' | 'down' | 'left' | 'right'.
    color     : str   – key into ANSI_COLORS.
    """

    def __init__(self, worm_id: str, segments: list, direction: str, color: str):
        self.worm_id   = worm_id
        self.segments  = list(segments)
        self.direction = direction
        self.color     = color

    @property
    def head(self):
        """The (row, col) of the worm's head (frontmost segment)."""
        return self.segments[-1]
