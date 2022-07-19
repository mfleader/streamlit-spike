import pandas as pd
from vyper import v

from PerformanceRange import PerformanceRange

def get_config():
    v.set_config_name('config')
    v.add_config_path('.')
    v.read_in_config()
    return v

def get_datasource():
    return get_config().get("settings.datasource")

def get_thresholds(metric: str, sr: pd.Series = None, platform: str ='default', cni: str = 'default'):
    t = get_config().get(f"thresholds.default.default.{metric}")
    return PerformanceRange(**t, sr=sr)
