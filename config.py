import yaml
import pandas as pd
from PerformanceRange import PerformanceRange

def get_config():
    with open('config.yaml', 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def get_datasource():
    return get_config()["settings"]["datasource"]

def get_thresholds(platform: str, cni: str, metric: str, sr: pd.Series):
    thresholds = get_config()["thresholds"]
    if platform not in thresholds:
        platform = "default"
    section = thresholds[platform]

    if cni not in section:
        cni = "default"

    section = section[cni]

    section = section[metric]

    great_low = section["great_low"]
    great_high = section["great_high"]
    poor_high = section["poor_high"]
    bad_high = section["bad_high"]

    range = PerformanceRange(sr)
    range.great_lo = great_low
    range.great_hi = great_high
    range.poor_hi = poor_high
    if (bad_high != -1):
        range.bad_hi = bad_high
    return range

