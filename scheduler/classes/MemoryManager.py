from . import Tensor

class MemoryManager:
    '''on-chip memory emulator'''

    def __init__(
            self,
            memory_size: int, #Bytes
            load_latency: int = 2, #clk per byte
            store_latency: int = 2, #clk per byte
            cache_latency: int = 1
        ) -> None:
        self.memory_size = memory_size
        self.memory_used = 0
        self.memory_available = memory_size
        self.load_latency = load_latency
        self.store_latency = store_latency
        self.cache_latency = cache_latency
        self.tensors = []

    def load_with_store(self, tensor, off_chip: list) -> int:
        '''load a tensor from off-chip memory and return latency'''

        if tensor is None :
            return 0
        if tensor in self.tensors :
            return 0
        latency = 0
        
        assert tensor.mem <= self.memory_size, 'Not enough memory.\n'
        while tensor.mem > self.memory_available :
            latency += self.store(self.tensors[0], off_chip)

        self.tensors.append(tensor)
        self.memory_available -= tensor.mem
        self.memory_used += tensor.mem
        latency += tensor.mem * self.load_latency

        return latency

    def store(self, tensor, off_chip: list) -> int:
        '''store a tensor into off-chip memory and return latency'''

        if tensor is None :
            return 0
        latency = 0

        try:
            self.tensors.remove(tensor)
            self.memory_available += tensor.mem
            self.memory_used -= tensor.mem
            if tensor not in off_chip :
                latency += tensor.mem * self.store_latency
                off_chip.append(tensor)
            
        except ValueError:
            print(f"Tensor {tensor} not found in memory.")
        
        return latency

    def check(self, tensor: Tensor) ->bool:
        return tensor in self.tensors
    
    def cache_in(self, tensor, off_chip: list) -> int: 
        '''Store a tensor from PE into the on-chip memory and return latency'''

        if tensor is None :
            return 0
        latency = 0

        assert tensor.mem <= self.memory_size, 'Not enough memory.\n'
        while tensor.mem > self.memory_available :
            latency += self.store(self.tensors[0], off_chip)

        self.tensors.append(tensor)
        self.memory_available -= tensor.mem
        self.memory_used += tensor.mem
        latency += tensor.mem * self.cache_latency

        return latency
    
    def cache_out(self, tensor) -> int:
        '''Load a tensor from on-chip memory to PE and return latency'''

        return tensor.mem * self.cache_latency

    def memory_usage(self) :
        return self.memory_used
    
    def load(self, tensor) -> int:
        '''load a tensor from off-chip memory and return latency'''
        
        if tensor is None:
            return 0
        if tensor in self.tensors :
            return 0
        latency = 0

        assert tensor.mem <= self.memory_size, 'Not enough memory.\n'
        while tensor.mem > self.memory_available :
            removed_tensor = self.tensors.pop(0)
            self.memory_available += removed_tensor.mem
            self.memory_used -= removed_tensor.mem

        self.tensors.append(tensor)
        self.memory_available -= tensor.mem
        self.memory_used += tensor.mem
        latency += tensor.mem * self.load_latency

        return latency
    
    def remove(self, tensor) -> None:
        '''remove a tensor.(only used for data transfer between two buffers)'''
        if tensor not in self.tensors :
            return
        else :
            self.tensors.remove(tensor)
            self.memory_available += tensor.mem
            self.memory_used -= tensor.mem