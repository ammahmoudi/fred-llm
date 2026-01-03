"""
Solution family strategies.

These strategies generate equations with non-unique solutions
(infinite families parameterized by arbitrary constants).

The u field contains the general solution form: u = C*φ(x) + u_p
All strategies have solution_type='family'.
"""

from src.data.augmentations.family.resonance import ResonanceAugmentation

__all__ = [
    "ResonanceAugmentation",
]
