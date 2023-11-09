from classes.computationNode import ComputationNode
from classes.memoryManager import MemoryManager

class GPUModel:
    '''emulator for one gpu'''

    def __init__(
            self,
            id,
            memory: MemoryManager,
        ) -> None:
        self.ID = id
        self.memory = memory
        self.latency = {}

    def runtime(self, CN: ComputationNode, host_mem: list) :
        for tensor in CN.tensor_fm:
            if self.memory.check(tensor) == False:
                raise ValueError(f'tensor {tensor} not found on GPU{self.ID}.')
        latency = self.latency[CN.layer]
        for cn in CN.tensor_out:
            latency += self.memory.cache_in(cn, host_mem)

        return latency
    
    def runtime_default(self, CN: ComputationNode, host_mem: list):
        for tensor in CN.tensor_fm:
            if self.memory.check(tensor) == False:
                raise ValueError(f'tensor {tensor} not found on GPU{self.ID}.')
        op_type = CN.op_type
        outsize = 0
        for tensor in CN.tensor_out:
            outsize += tensor.size

        if op_type == 'Conv':
            k_1 = CN.tensor_w[0].loop_ranges[2][1] - CN.tensor_w[0].loop_ranges[2][0]
            k_2 = CN.tensor_w[0].loop_ranges[3][1] - CN.tensor_w[0].loop_ranges[3][0]
            latency = ((k_1*k_2*2-1) * outsize) // 10
        elif op_type == 'Add' or op_type == 'Relu':
            latency = outsize // 10
        elif op_type == 'BatchNormalization':
            latency = outsize*5 // 10
        else:
            raise ValueError(f'dont support op: {op_type}.')

        for cn in CN.tensor_out:
            latency += self.memory.cache_in(cn, host_mem)

        return latency
    
    def set_latency(self, latency: dict):
        self.latency = latency
