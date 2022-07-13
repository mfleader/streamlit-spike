"""
This module defines the PerformanceRange class, which defines expected performance.
"""

import pandas as pd
from dataclasses import dataclass

class PerformanceRange:
    great_lo: float = 0
    great_hi: float = 0
    poor_hi: float = 0
    bad_hi: float = 0
    color: str = 'inverse'

    def __init__(self, sr: pd.Series, great_hi: float = None, color: str = 'inverse'):
        if great_hi:
            self.great_hi = great_hi
        else:
            self.great_hi = sr.quantile(q=.1)
        self.great_lo = sr.min()
        self.bad_hi = sr.max()
        self.poor_hi = self.great_hi + .5 * (self.bad_hi - self.great_hi)


@dataclass
class PerformanceRangeHigherBetter:
    great_lo: float = 0
    great_hi: float = 0
    poor_hi: float = 0
    bad_hi: float = 0

    # def __init__(self, sr: pd.Series, great_lo: float = None):
    #     if great_lo:
    #         self.great_lo = great_lo
    #     else:
    #         self.great_hi = sr.quantile(q=.1)
    #     self.great_lo = sr.min()
    #     self.bad_hi = sr.max()
    #     self.poor_hi = self.great_hi + .5 * (self.bad_hi - self.great_hi)

