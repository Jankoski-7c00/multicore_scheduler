from . import ComputationNode
from . import MemoryManager
import math

class Core:
    '''emulator for one npu core'''

    def __init__(
            self,
            core_id: int,
            gemm_size: tuple = (16, 16),
            gemm_latency: int = 20, #clk
            ve_size: int = 256,
            ve_latency: int = 10, #clk
            on_chip_memory: int = 1, #KB
        ) -> None:
        self.core_id = core_id
        self.gemm_size = gemm_size
        self.gemm_latency = gemm_latency
        self.ve_size = ve_size
        self.ve_latency = ve_latency
        #self.on_chip_memory = on_chip_memory
        self.memory = MemoryManager(on_chip_memory)
    
    def runtime(self, CN: ComputationNode, off_chip: list):
        load_time = 0
        compute_time = 0
        
        if self.memory.check(CN.tensor_fm) is False :
            load_time += self.memory.load(CN.tensor_fm, off_chip)

        if self.memory.check(CN.tensor_w) is False :
            load_time += self.memory.load(CN.tensor_w, off_chip)

        #time to compute includes the time of data transmission between on-chip memory and PE
        if CN.op_type == 'GEMM' :
            compute_time += self.memory.cache_latency * (CN.tensor_fm.mem + CN.tensor_w.mem) + self.memory.cache(CN.tensor_out)
            compute_time += self.gemm_latency * math.ceil((CN.tensor_fm.loop_ranges[0][1]-CN.tensor_fm.loop_ranges[0][0]+1)/self.gemm_size[0]) * math.ceil((CN.tensor_fm.loop_ranges[1][1]-CN.tensor_fm.loop_ranges[1][0]+1)/self.gemm_size[1])
        elif CN.op_type == 'BatchNorm' :
            compute_time += self.memory.cache_latency * (CN.tensor_fm.mem + CN.tensor_w.mem) + self.memory.cache(CN.tensor_out)
            compute_time += self.ve_latency * math.ceil(CN.tensor_fm.size / self.ve_size) * 3
        elif CN.op_type == 'Relu' :
            compute_time += self.memory.cache_latency * CN.tensor_fm.mem
            compute_time += self.ve_latency * math.ceil(CN.tensor_fm.size / self.ve_size)
        else :
            assert False, 'No op_type.\n'

        return load_time + compute_time
    
    def store_result(self, CN: ComputationNode, off_chip: list):
        return self.memory.store(CN, off_chip)
    
    def memory_usage(self) :
        return self.memory.memory_used
