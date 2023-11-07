class ComputationNode:
    '''class to present a computation node'''

    def __init__(
            self,
            layer,
            op_type: str,
            core_allocation: int = 0,
            tensor_fm = [],
            tensor_w = [],
            tensor_out = []
        ) -> None:
        self.layer = layer
        self.op_type = op_type
        self.core_allocation = core_allocation
        self.tensor_fm = tensor_fm
        self.tensor_w = tensor_w
        self.tensor_out = tensor_out
        self.__node_ID = None

    @property
    def node_ID(self) -> int:
        return self.__node_ID

    @node_ID.setter
    def node_ID(self, id: int) -> None:
        if self.__node_ID is None:
            self.__node_ID = id
        else:
            raise ValueError("node_ID can only be set once.")

    def __hash__(self):
        return hash((self.layer, self.op_type, self.tensor_fm, self.tensor_w, self.tensor_out, self.node_ID))

    def __eq__(self, other):
        if isinstance(other, ComputationNode):
            return self.node_ID == other.node_ID and self.layer == other.layer and self.op_type == other.op_type and self.tensor_fm == other.tensor_fm and self.tensor_w == other.tensor_w and self.tensor_out == other.tensor_out
        return False

    def __repr__(self):
        return f"ComputationNode(layer={self.layer}, op_type={self.op_type}, ID={self.node_ID})"
    
    def set_core_allocation(self, core_id :int) ->None :
        self.core_allocation = core_id
