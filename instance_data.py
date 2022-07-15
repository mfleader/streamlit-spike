
import pandas as pd

def get_instance_data():
    """
    @return a dataframe with the following columns:
            platform: The cloud the instance is on.
            series: A grouping of instances that are in a
                    series. Used for looking for compatible upgrades.
            instance_type: The instance type.
            vcpu: Number of vcpus available in the instance.
            memory: The amount of memory in the instance, in MiB.
    """
    return pd.read_csv('data/instance_data.csv')

def get_larger_instances(df: pd.DataFrame, to_match: str, greater_cpu: bool, greater_mem: bool):
    """
    Finds matching instances that have the required CPU, RAM, and series.
    df: The dataframe
    return: dataframe with matching instances.
    """
    existing_instance = df[df.instance_type == to_match]
    if len(existing_instance) == 0:
        # Empty
        return pd.DataFrame(columns=["platform", "series", "instance_type", "vcpu", "memory"])
    else:
        existing_instance = existing_instance.values[0]
    print(existing_instance)

    series = existing_instance[1]
    instance_type = existing_instance[2]
    vcpu = existing_instance[3]
    mem = existing_instance[4]


    min_mem = mem
    min_vcpu = vcpu
    if greater_mem:
        min_mem += 1024
    if greater_cpu:
        min_vcpu += 1
    result = df[(df.instance_type != instance_type)&(df.series == series)&(df.vcpu >= min_vcpu)&(df.memory_mb >= min_mem)]
    result = result.sort_values(by=["vcpu", "memory"])
    return result


#df = get_instance_data()
#print(df)
#print(get_larger_instances(df, "m5.4xlarge", False, False))
