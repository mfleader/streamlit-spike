import boto3
import pandas as pd

regions= [
    #'ap-east-1',
#    'ap-northeast-1',
#    'ap-northeast-2',
#    'ap-south-1',
#    'ap-southeast-1',
#    'ap-southeast-2',
#    'ca-central-1',
#    'eu-central-1',
#    'eu-north-1',
#    'eu-west-1',
#    'eu-west-2',
#    'eu-west-3',
    #'me-south-1',
#    'sa-east-1',
    'us-east-1',
    'us-east-2',
    'us-west-1',
    'us-west-2'
]


def add_instance_types(instances):
    for instance in instances['InstanceTypes']:
        instance_type = instance["InstanceType"]
        if not instance_type in added_instances:
            added_instances.add(instance_type)
            new_data = [
                 "aws",
                instance_type.split(".", 1)[0],
                instance_type,
                instance["VCpuInfo"]["DefaultVCpus"],
                instance["MemoryInfo"]["SizeInMiB"]
            ]
            df.loc[len(df.index)] = new_data


df = pd.DataFrame(columns=["platform", "series", "instance_type", "vcpu", "memory"])
added_instances = set()

for region_name in regions:
    print(f'region_name: {region_name}')
    ec2= boto3.resource('ec2', region_name=region_name)
    next_token = None
    instances= ec2.meta.client.describe_instance_types()
    add_instance_types(instances)
    next_token = instances["NextToken"]
    while next_token != None:
        instances= ec2.meta.client.describe_instance_types(NextToken=next_token)
        add_instance_types(instances)
        next_token = instances.get("NextToken", None)

df.to_csv(r'instance_data.csv', index = False)
