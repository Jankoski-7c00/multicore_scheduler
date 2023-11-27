class Tensor:
    '''class to present a tensor used by a computation node'''

    def __init__(
            self,
            size: int,
            datatype: str = 'float32',
            loop_dimensions: tuple = None, #e.g. ('N', 'C', 'H', 'W')
            loop_ranges: tuple = None, #e.g. ((0, 15), (16, 31), (32, 63), (64, 79))
            #origin: str = None, #whether the tensor comes from feature map, weights or output
            producer_layer = None,
            #consumer_layer = None,
            is_weight = False
        ) -> None:
        self.datatype = datatype
        self.size = size
        self.loop_dimensions = loop_dimensions
        self.loop_ranges = loop_ranges
        #self.origin = origin
        self.producer_layer = producer_layer
        #self.consumer_layer = consumer_layer
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
        return f"producer layer: {self.producer_layer}, dimension: {self.loop_dimensions}, range: {self.loop_ranges}"
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.datatype == other.datatype and self.size == other.size and self.loop_dimensions == other.loop_dimensions and self.loop_ranges == other.loop_ranges and self.producer_layer == other.producer_layer and self.is_weight == other.is_weight
        return False
    
    def __hash__(self):
        return hash((self.datatype, self.size, self.loop_dimensions, self.loop_ranges, self.producer_layer, self.is_weight))
    
    def get_memory_usage(self) -> int:
        return self.mem
    
    def get_shape(self):
        shape = []
        for loop_range in self.loop_ranges:
            i = loop_range[1] - loop_range[0] + 1
            shape.append(i)
        return tuple(shape)