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
            consumer = None
        ) -> None:
        self.datatype = datatype
        self.size = size
        self.loop_dimensions = loop_dimensions
        self.loop_ranges = loop_ranges
        #self.origin = origin
        self.producer = producer
        self.consumer = consumer

        #self.mem is the memory usage of the tensor
        if datatype == 'int8' or datatype == 'uint8':
            self.mem = size
        elif datatype == 'float16':
            self.mem = size * 2
        elif datatype == 'float32' or datatype == 'int':
            self.mem = size * 4
        elif datatype == 'float64':
            self.mem = size * 8
        else :
            assert 'Wrong datatype.\n'
    
    def __repr__(self) -> str:
        return f"dimension: {self.loop_dimensions}, range: {self.loop_ranges}"
    
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.datatype == other.datatype and self.size == other.size and self.loop_dimensions == other.loop_dimensions and self.loop_ranges == other.loop_ranges and self.producer == other.producer and self.consumer == other.consumer
        return False
    
    def __hash__(self):
        return hash((self.datatype, self.size, self.loop_dimensions, self.loop_ranges, self.producer, self.consumer))