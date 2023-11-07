from onnx_parser import OnnxParser
import onnx
from classes.tensor import Tensor
from classes.computationNode import ComputationNode
from networkx import DiGraph

class ComputationGraph :
    '''computation graph generate'''
    def __init__(
            self,
            parser: OnnxParser,
            split_factor: int = 4
        ) -> None:
        self.parser = parser
        self.split_factor = split_factor
        self.CN_num = 0
        self.CN_graph = DiGraph()

    def split_relu(self, layer, split_axis:str):
        if layer['op_type'] is not ''


model = onnx.load('/Users/xiangyy/Projects/multicore_schedule/scheduler/backend/resnet18.onnx')
parser = OnnxParser(model)
parser.show()
