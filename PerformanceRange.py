"""
This module defines the PerformanceRange class, which defines expected performance.
"""

import pandas as pd
from typing import Optional

class PerformanceRange:
    great_lo: Optional[float] = None
    great_hi: Optional[float] = None
    poor_hi: Optional[float] = None
    poor_lo: Optional[float] = None
    bad_hi: Optional[float] = None
    bad_lo: Optional[float] = None
    max_value: Optional[float] = None
    sr: Optional[pd.Series] = None
    perf_delta: str = 'inverse'  # 'inverse', 'normal'

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        if self.sr is not None and self.perf_delta == 'normal' and self.sr.max() > self.great_hi:
            self.great_hi = self.sr.max()
        elif self.sr is not None and self.perf_delta == 'inverse' and self.sr.max() > self.bad_hi:
            self.bad_hi = self.sr.max()

    def __str__(self):
        return (
            f"PerformanceRange("
            f"great_hi: {self.great_hi}, "
            f"great_lo: {self.great_lo}, "
            f"poor_hi: {self.poor_hi}, "
            f"poor_lo: {self.poor_lo}, "
            f"bad_hi: {self.bad_hi}, "
            f"bad_lo: {self.bad_lo}"
            ")"
        )

    def get_msg_suffix(self, value):
        if self.perf_delta == 'inverse':
            if value < self.great_hi:
                return "_GOOD"
            elif value < self.poor_hi:
                return "_BELOW_EXPECTATIONS"
            elif value >= self.poor_hi:
                return "_BAD"
        elif self.perf_delta == 'normal':
            if value < self.poor_lo:
                return "_BAD"
            elif value < self.great_lo:
                return "_BELOW_EXPECTATIONS"
            elif value >= self.great_lo:
                return "_GOOD"
        return "_MISSING"


class QuantilePerfRange(PerformanceRange):

    def __init__(self, sr: pd.Series, great_hi: float = None, color: str = 'inverse'):
        if great_hi:
            self.great_hi = great_hi
        else:
            self.great_hi = sr.quantile(q=.1)
        super().__init__(sr)
        self.poor_hi = self.great_hi + .5 * (self.bad_hi - self.great_hi)
