from classes.computationNode import ComputationNode

class Accelerater:
    '''emulator for a multi-core npu'''

    def __init__(
            self,
            cores: list,
        ) -> None:
        self.cores = cores
        self.off_chip_memory = []
        self.core_number = len(cores)

    def runtime(self, CN: ComputationNode, is_result: bool):
        for core in self.cores :
            if core.core_id == CN.core_allocation :
                if is_result:
                    runtime = core.runtime(CN, self.off_chip_memory)
                    runtime += core.store_result(CN, self.off_chip_memory)
                    return runtime
                
                else :
                    return core.runtime(CN, self.off_chip_memory)
            
#    def off_chip_memory_usage(self):
#        usage = 0
#        for tensor in self.off_chip_memory :
#            usage += tensor.mem
#
#        return usage
    
    def on_chip_memory_usage(self, core_id: int):
        for core in self.cores :
            if core.core_id == core_id :
                return core.memory_usage()

