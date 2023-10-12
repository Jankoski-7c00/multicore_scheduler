from classes.accelerater import Accelerater
from networkx import DiGraph
from scheduler import scheduler

class CostModel :
    '''TODO: data visualization for the CN schedule and memory utilization.'''

    def __init__(
            self,
            workload: DiGraph,
            accelerater: Accelerater,
            strategy: str = 'Latency'
        ) -> None:
        self.workload = workload
        self.accelerater = accelerater
        self.strategy = strategy
        #self.scheduled_cn, self.memory_usage = scheduler(self.accelerater, self.workload ,self.strategy)
        
    def get_latency(self) :
        latency = 0
        for cn in self.scheduled_cn :
            if cn[2] > latency :
                latency = cn[2]

        return latency
    
    def change_strategy(self, strategy: str) -> None:
        if strategy == 'Latency' :
            self.strategy = strategy
            self.scheduled_cn, self.memory_usage = scheduler(self.accelerater, self.workload ,self.strategy)
        elif strategy == 'Memory' :
            self.strategy = strategy
            self.scheduled_cn, self.memory_usage = scheduler(self.accelerater, self.workload ,self.strategy)
        else:
            raise ValueError('Wrong strategy.\n')

    def set_core_allocation(self, core_allocation: list) :
        for node in self.workload.nodes() :
            id = node.node_ID
            core_id = core_allocation[id]
            node.set_core_allocation(core_id)

        self.scheduled_cn, self.memory_usage = scheduler(self.accelerater, self.workload ,self.strategy)