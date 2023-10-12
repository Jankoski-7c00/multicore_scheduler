from classes.computationNode import ComputationNode
from classes.memoryManager import MemoryManager
from classes.tensor import Tensor
import math

class Core:
    '''
    emulator for one npu core
    TODO: support GPU tensor core ?
    '''

    def __init__(
            self,
            core_id: int,
            core_type: str = 'NPU',
            gemm_size: tuple = (16, 16),
            gemm_latency: int = 20, #clk
            ve_size: int = 256,
            ve_latency: int = 10, #clk
            on_chip_memory: int = 1, #Bytes
            ve_bufferA: int = 10240,
            ve_bufferB: int = 10240,
            ve_output: int = 10240,
            gemm_Fmap: int = 10240,
            gemm_weights: int = 10240,
            gemm_output: int = 10240,
            transfer_latency:int = 10 #clk per byte
        ) -> None:
        self.core_id = core_id
        self.core_type = core_type
        self.gemm_size = gemm_size
        self.gemm_latency = gemm_latency
        self.ve_size = ve_size
        self.ve_latency = ve_latency
        #self.on_chip_memory = on_chip_memory
        #self.memory = MemoryManager(on_chip_memory)

        if self.core_type == 'NPU' :
            self.ve_bufferA = MemoryManager(ve_bufferA)
            self.ve_bufferB = MemoryManager(ve_bufferB)
            self.ve_output = MemoryManager(ve_output)
            self.gemm_Fmap = MemoryManager(gemm_Fmap)
            self.gemm_weights = MemoryManager(gemm_weights)
            self.gemm_output = MemoryManager(gemm_output)
            self.buffer_map = {
                've_bufferA': self.ve_bufferA,
                've_bufferB': self.ve_bufferB,
                've_output': self.ve_output,
                'gemm_Fmap': self.gemm_Fmap,
                'gemm_weights': self.gemm_weights,
                'gemm_output': self.gemm_output
            }
            self.transfer_latency = transfer_latency

        elif self.core_type == 'GPU' :
            self.on_chip_memory = MemoryManager(on_chip_memory)

        else:
            assert False, 'Wrong core type.\n'

    def move(self, tensor: Tensor, A: str, B: str, isCopy = False):
        '''move tensor from buffer A to B.'''
        
        buffer_a = self.buffer_map.get(A)
        buffer_b = self.buffer_map.get(B)

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
        '''find which buffer the tensor is in'''

        for name, buffer in self.buffer_map.items() :
            if buffer.check(tensor) :
                return name, buffer
            
        return None, None
    
    def load(self, tensor: Tensor, buffer_name: str) :
        '''load a tensor to the specified buffer and return latency.'''

        destinaiton_buffer = self.buffer_map.get(buffer_name)
        source_buffer_name, source_buffer = self.find(tensor)
        if source_buffer is None :
            load_time = destinaiton_buffer.load(tensor)
        elif source_buffer is destinaiton_buffer :
            pass
        elif source_buffer is self.gemm_output or self.ve_output:
            load_time = self.move(tensor, source_buffer_name, buffer_name)
        else :
            load_time = self.move(tensor, source_buffer_name, buffer_name, isCopy = True)

        return load_time + destinaiton_buffer.cache_out(tensor)

    def runtime(self, CN: ComputationNode, off_chip: list):
        '''Simulate the operation of a CN and return latency'''

        load_time = 0
        compute_time = 0
        store_time = 0

        if CN.op_type == 'GEMM' :
            #load
            load_time += self.load(CN.tensor_fm, 'gemm_Fmap')
            load_time += self.load(CN.tensor_w, 'gemm_weights')

            #compute
            compute_time += self.gemm_latency * math.ceil((CN.tensor_fm.loop_ranges[0][1]-CN.tensor_fm.loop_ranges[0][0]+1)/self.gemm_size[0]) * math.ceil((CN.tensor_fm.loop_ranges[1][1]-CN.tensor_fm.loop_ranges[1][0]+1)/self.gemm_size[1])
            
            #store result
            store_time += self.gemm_output.cache_in(CN.tensor_out, off_chip)

        elif CN.op_type == 'BatchNorm' :
            #load
            load_time += self.load(CN.tensor_fm, 've_bufferA')
            load_time += self.load(CN.tensor_w, 've_bufferB')

            compute_time += self.ve_latency * math.ceil(CN.tensor_fm.size / self.ve_size) * 3

            store_time += self.ve_output.cache_in(CN.tensor_out, off_chip)
        
        elif CN.op_type == 'Relu' or 'Add' or 'Sub':
            #load
            load_time += self.load(CN.tensor_fm, 've_bufferA')
            load_time += self.load(CN.tensor_w, 've_bufferB')

            compute_time += self.ve_latency * math.ceil(CN.tensor_fm.size / self.ve_size)

            store_time += self.ve_output.cache_in(CN.tensor_out, off_chip)
        else :
            raise ValueError('No op_type.\n')

        return load_time + compute_time + store_time

    '''
    def runtime(self, CN: ComputationNode, off_chip: list):
        #Simulate the operation of a CN and return latency
        load_time = 0
        compute_time = 0
        
        if self.memory.check(CN.tensor_fm) is False :
            load_time += self.memory.load(CN.tensor_fm, off_chip)

        if self.memory.check(CN.tensor_w) is False :
            load_time += self.memory.load(CN.tensor_w, off_chip)

        #time to compute includes the time of data transmission between on-chip memory and PE
        if CN.op_type == 'GEMM' :
            #compute
            compute_time += self.gemm_latency * math.ceil((CN.tensor_fm.loop_ranges[0][1]-CN.tensor_fm.loop_ranges[0][0]+1)/self.gemm_size[0]) * math.ceil((CN.tensor_fm.loop_ranges[1][1]-CN.tensor_fm.loop_ranges[1][0]+1)/self.gemm_size[1])
            #store the result into the on-chip memory
            compute_time += self.memory.cache_latency * (CN.tensor_fm.mem + CN.tensor_w.mem) + self.memory.cache(CN.tensor_out)
        elif CN.op_type == 'BatchNorm' : 
            compute_time += self.ve_latency * math.ceil(CN.tensor_fm.size / self.ve_size) * 3
            compute_time += self.memory.cache_latency * (CN.tensor_fm.mem + CN.tensor_w.mem) + self.memory.cache(CN.tensor_out)
        elif CN.op_type == 'Relu' :
            compute_time += self.ve_latency * math.ceil(CN.tensor_fm.size / self.ve_size)
            compute_time += self.memory.cache_latency * CN.tensor_fm.mem
        else :
            assert False, 'No op_type.\n'

        return load_time + compute_time
    '''

    def store_result(self, CN: ComputationNode, off_chip: list):
        _, buffer = self.find(CN.tensor_out)
        return buffer.store(CN.tensor_out, off_chip)
    
    def memory_usage(self) :
        mem_usage = 0
        for buffer in self.buffer_map.values() :
            mem_usage += buffer.memory_usage()

        return mem_usage
