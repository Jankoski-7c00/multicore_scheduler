from classes.computationNode import ComputationNode
from classes.memoryManager import MemoryManager
from classes.tensor import Tensor
from classes.gpumodel import GPUModel

class MultiGPU:
    '''emulator for a multi-GPU system'''

    def __init__(
            self,
            gpu_num: int,
            compute_latency: dict,
            gpu_memory: int = 4 * 1024 * 1024 * 1024,
            transfer_latency: int = 1
        ) -> None:
        self.gpu_num = gpu_num
        self.compute_latency = compute_latency
        self.transfer_latency = transfer_latency
        self.host_mem = []
        self.GPU_dict = {}
        self.memory_dict = {}
        for i in range(self.gpu_num):
            self.memory_dict[i] = MemoryManager(gpu_memory)
            self.GPU_dict[i] = GPUModel(i, self.memory_dict[i])
            self.GPU_dict[i].set_latency(self.compute_latency)

    def move(self, tensor: Tensor, A: int, B: int, isCopy = False):
        '''move tensor from one gpu to another.'''
        
        buffer_a = self.memory_dict.get(A)
        buffer_b = self.memory_dict.get(B)

        if buffer_a.check(tensor) is False :
            raise ValueError('tensor not found.\n')
        
        if isCopy:
            buffer_b.load(tensor)
            return tensor.mem * self.transfer_latency
        else :
            buffer_a.remove(tensor)
            buffer_b.load(tensor)
            return tensor.mem * self.transfer_latency
        
    def find(self, tensor: Tensor) :
        '''find which gpu the tensor is in'''

        for id, buffer in self.memory_dict.items() :
            if buffer.check(tensor) :
                return id, buffer
            
        return None, None
    
    def load(self, tensor: Tensor, gpu_id: int) :
        '''load a tensor to the specified buffer and return latency.'''

        destination_gpu = self.memory_dict.get(gpu_id)
        source_gpu_id, source_gpu = self.find(tensor)
        if source_gpu is None :
            load_time = destination_gpu.load(tensor)
        elif source_gpu is destination_gpu :
            load_time = 0
        else :
            load_time = self.move(tensor, source_gpu_id, gpu_id, isCopy = True)

        return load_time
    
    def runtime(self, CN: ComputationNode, gpu_id: int):
        load_time = 0
        compute_time = 0
        for tensor in CN.tensor_fm:
            load_time += self.load(tensor, gpu_id)
        for tensor in CN.tensor_w:
            load_time += self.load(tensor, gpu_id)

        compute_time += self.GPU_dict[gpu_id].runtime_default(CN, self.host_mem)
        return compute_time + load_time
    
    def runtime_default(self, CN: ComputationNode, gpu_id: int):
        if CN.op_type == 'Flatten' or CN.op_type == 'MaxPool' or CN.op_type == 'GlobalAveragePool' or CN.op_type == 'Gemm':
            return 0
        load_time = 0
        compute_time = 0
        for tensor in CN.tensor_fm:
            load_time += self.load(tensor, gpu_id)
        for tensor in CN.tensor_w:
            load_time += self.load(tensor, gpu_id)

        compute_time += self.GPU_dict[gpu_id].runtime_default(CN, self.host_mem)
        return compute_time + load_time
    
    def memory_usage(self) :
        mem = 0
        for memory in self.memory_dict.values():
            mem += memory.memory_usage()
        return mem
    
    def reset(self):
        for memory in self.memory_dict.values():
            memory.reset()