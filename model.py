from typing import Optional

from sqlmodel import Field, SQLModel
from pydantic import condecimal



class Run_Metrics(SQLModel, table=True):
    __tablename__ = 'run_metrics'
    uuid: str = Field(primary_key=True)
    workload: str
    platform: str
    # ocp_version: condecimal(max_digits=4, decimal_places=2)
    ocp_version: float
    sdn_type: str
    master_nodes_type: str
    worker_nodes_type: str
    infra_nodes_type: str
    workload_nodes_type: str
    master_nodes_count: int
    worker_nodes_count: int
    infra_nodes_count: int
    workload_nodes_count: int
    total_nodes: int
    timestamp: int
    end_date: int
    k8s_version: str
    ocp_nightly_version: str
    podlatencyquantilesmeasurement_containersready_avg_p99: float
    podlatencyquantilesmeasurement_initialized_avg_p99: float
    podlatencyquantilesmeasurement_podscheduled_avg_p99: float
    containermemory_avg: float
    containermemory_max: float
    containercpu_avg: float
    containercpu_max: float
    nodecpu_workers_avg: float
    nodecpu_workers_max: float
    nodecpu_masters_avg: float
    nodecpu_masters_max: float
    nodecpu_infra_avg: float
    nodecpu_infra_max: float
    apirequestrate_avg: float
    apirequestrate_max: float
    readonlyapicallslatency_avg: float
    readonlyapicallslatency_max: float
    criocpu_avg: float
    criocpu_max: float
    criomemory_avg: float
    criomemory_max: float
    kubeletcpu_avg: float
    kubeletcpu_max: float
    kubeletmemory_avg: float
    kubeletmemory_max: float
    mutatingapicallslatency_avg: float
    mutatingapicallslatency_max: float
    containernetworksetuplatency_avg: float
    containernetworksetuplatency_max: float
    p99thetcdroundtriptimeseconds_avg: float
    p99thetcddiskbackendcommitdurationseconds_avg: float
    p99thetcddiskbackendcommitdurationseconds_max: float
    p99thetcddiskwalfsyncdurationseconds_avg: float
    p99thetcddiskwalfsyncdurationseconds_max: float
    nodememoryavailable_workers_avg: float
    nodememorytotal_workers_max: float
    nodememoryutilization_workers_avg: float
    nodememoryutilization_workers_max: float
    nodememoryavailable_infra_avg: float
    nodememoryavailable_masters_avg: float
    nodememorytotal_infra_max: float
    nodememorytotal_masters_max: float
    etcdleaderchangesrate_max: float
    namespacecount_max: float
    schedulingthroughput_avg: float
    schedulingthroughput_max: float
    configmapcount_max: float
    deploymentcount_max: float
    etcdversion_max: float
    secretcount_max: float
    servicecount_max: float
    poddistribution_max: float