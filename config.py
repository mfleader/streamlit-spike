import yaml
import pandas as pd

from vyper import v

from PerformanceRange import PerformanceRange

def get_config():
    # with open('config.yaml', 'r') as f:
        # return yaml.load(f, Loader=yaml.FullLoader)
    v.set_config_name('config')
    v.add_config_path('.')
    v.read_in_config()
    return v

def get_datasource():
    return get_config().get("settings.datasource")

def get_thresholds(metric: str, sr: pd.Series = None, platform: str ='default', cni: str = 'default'):
    # thresholds = get_config().get("thresholds.default.default")

    # if platform not in thresholds:
    #     platform = "default"
    # section = thresholds[platform]

    # if cni not in section:
    #     cni = "default"

    # section = section[cni]

    # section = section[metric]

    # great_low = section["great_low"]
    # great_high = section["great_high"]
    # poor_high = section["poor_high"]
    # bad_high = section["bad_high"]

    # range = PerformanceRange(sr)
    # range.great_lo = great_low
    # range.great_hi = great_high
    # range.poor_hi = poor_high
    # if (bad_high != -1):
    #     range.bad_hi = bad_high
    # return range
    t = get_config().get(f"thresholds.default.default.{metric}")

    if sr is not None:
        max_value = sr.max()
    else:
        max_value = t['great_hi']

    return PerformanceRange(**t, max_value = max_value)


# if __name__ == '__main__':
    # ts = get_config().get("thresholds.default.default")
    # print(len(ts))
    # print(type(ts))
    # for name,t in ts.items():
    #     print(name)
    #     pt = PerformanceRange(**t)
    #     print(pt)
        # print(t)
    # print(get_thresholds('pod_start_latency'))