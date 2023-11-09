from classes.multigpu import MultiGPU
from networkx import DiGraph
from schedule_algorithm.list_algorithm import list_scheduling_algorithm

class CostModel :
    '''TODO: data visualization for the CN schedule and memory utilization.'''

    def __init__(
            self,
            workload: DiGraph,
            accelerater: MultiGPU,
        ) -> None:
        self.workload = workload
        self.accelerater = accelerater
        self.scheduled_cn_map, self.memory_usage = list_scheduling_algorithm(self.workload, self.accelerater)
        
    def get_latency(self) :
        latency = 0
        for _, scheduled_cns in self.scheduled_cn_map.items() :
            for cn_info in scheduled_cns:
                if cn_info[2] > latency:
                    latency = cn_info[2]

        return latency

    def set_core_allocation(self, core_allocation: list) :
        for node in self.workload.nodes() :
            id = node.node_ID
            core_id = core_allocation[id]
            node.set_core_allocation(core_id)
            self.accelerater.reset()
        self.scheduled_cn_map, self.memory_usage = list_scheduling_algorithm(self.workload, self.accelerater)

    def get_core_allocation(self):
        allocation_list = [None] * self.workload.number_of_nodes()
        for node in self.workload.nodes():
            node_id = node.node_ID
            allocation_list[node_id] = node.core_allocation

        return allocation_list
    
    def print_scheduled_cn(self):
        for gpuid, scheduled_cns in self.scheduled_cn_map.items():
            print(f'GPU_{gpuid}:')
            for cn_info in scheduled_cns:
                print(f'CN id: {cn_info[0].node_ID}\tlayer: {cn_info[0].layer}\tstart time: {cn_info[1]}\tend time:{cn_info[2]}')