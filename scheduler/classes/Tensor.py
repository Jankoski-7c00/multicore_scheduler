class Tensor:
    '''class to present a tensor used by a computation node'''

    def __init__(
            self,
            datatype: str,
            size: int,
            loop_dimensions: tuple = None,
            loop_ranges: tuple = None,
            #origin: str = None, #whether the tensor comes from feature map, weights or output
            producer = None,
            consumer = None,
            is_weight = False
        ) -> None:
        self.datatype = datatype
        self.size = size
        self.loop_dimensions = loop_dimensions
        self.loop_ranges = loop_ranges
        #self.origin = origin
        self.producer = producer
        self.consumer = consumer
        self.is_weight = is_weight

        #self.mem is the memory usage of the tensor
        if datatype in ['int8', 'uint8']:
            self.mem = size
        elif datatype == 'float16':
            self.mem = size * 2
        elif datatype in ['float32', 'int']:
            self.mem = size * 4
        elif datatype == 'float64':
            self.mem = size * 8
        else:
            raise ValueError('Wrong datatype.\n')
    
    def __repr__(self) -> str:
        return f"dimension: {self.loop_dimensions}, range: {self.loop_ranges}"
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.datatype == other.datatype and self.size == other.size and self.loop_dimensions == other.loop_dimensions and self.loop_ranges == other.loop_ranges and self.producer == other.producer and self.consumer == other.consumer and self.is_weight == other.is_weight
        return False
    
    def __hash__(self):
        return hash((self.datatype, self.size, self.loop_dimensions, self.loop_ranges, self.producer, self.consumer, self.is_weight))
    
    def get_memory_usage(self) -> int:
        return self.mem