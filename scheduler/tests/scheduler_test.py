import sys
sys.path.append('/Users/xiangyy/Projects/multicore_schedule/scheduler')
sys.path.append('/Users/xiangyy/Projects/multicore_schedule')
from classes.accelerater import Accelerater
from classes.core import Core
from classes.tensor import Tensor
from classes.computationNode import ComputationNode
from costModel import CostModel
from networkx import DiGraph

a = Tensor('int8', 256, loop_ranges = ((0, 15), (0, 15)), consumer_layer = 0)
a_1 = Tensor('int8', 256, loop_ranges = ((16, 31), (0, 15)), consumer_layer = 0)
a_2 = Tensor('int8', 256, loop_ranges = ((0, 15), (16, 31)), consumer_layer = 0)
a_3 = Tensor('int8', 256, loop_ranges = ((16, 31), (16, 31)), consumer_layer = 0)

w = Tensor('int8', 256, loop_ranges = ((0, 15), (0, 15)), consumer_layer = 0, is_weight = True)
w_1 = Tensor('int8', 256, loop_ranges = ((16, 31), (0, 15)), consumer_layer = 0, is_weight = True)
w_2 = Tensor('int8', 256, loop_ranges = ((0, 15), (16, 31)), consumer_layer = 0, is_weight = True)
w_3 = Tensor('int8', 256, loop_ranges = ((16, 31), (16, 31)), consumer_layer = 0, is_weight = True)

b = Tensor('int8', 256, loop_ranges = ((0, 15), (0, 15)), producer_layer = 0, consumer_layer = 1)
b_1 = Tensor('int8', 256, loop_ranges = ((16, 31), (0, 15)), producer_layer = 0, consumer_layer = 1)
b_2 = Tensor('int8', 256, loop_ranges = ((0, 15), (16, 31)), producer_layer = 0, consumer_layer = 1)
b_3 = Tensor('int8', 256, loop_ranges = ((16, 31), (16, 31)), producer_layer = 0, consumer_layer = 1)

c = Tensor('int8', 256, loop_ranges = ((0, 15), (0, 15)), producer_layer = 1, consumer_layer = 2)
c_1 = Tensor('int8', 256, loop_ranges = ((16, 31), (0, 15)), producer_layer = 1, consumer_layer = 2)
c_2 = Tensor('int8', 256, loop_ranges = ((0, 15), (16, 31)), producer_layer = 1, consumer_layer = 2)
c_3 = Tensor('int8', 256, loop_ranges = ((16, 31), (16, 31)), producer_layer = 1, consumer_layer = 2)

d = Tensor('int8', 256, loop_ranges = ((0, 15), (0, 15)), producer_layer = 2)
d_1 = Tensor('int8', 256, loop_ranges = ((16, 31), (0, 15)), producer_layer = 2)
d_2 = Tensor('int8', 256, loop_ranges = ((0, 15), (16, 31)), producer_layer = 2)
d_3 = Tensor('int8', 256, loop_ranges = ((16, 31), (16, 31)), producer_layer = 2)

workload = DiGraph()

#gemm layer
cn_0 = ComputationNode(0, 'GEMM', tensor_fm = a, tensor_w = w, tensor_out = b)
cn_0.node_ID = 0
cn_1 = ComputationNode(0, 'GEMM', tensor_fm = a_1, tensor_w = w_2, tensor_out = b)
cn_1.node_ID = 1
#workload.add_nodes_from([cn_0, cn_1])
workload.add_edge(cn_0, cn_1)

cn_2 = ComputationNode(0, 'GEMM', tensor_fm = a, tensor_w = w_1, tensor_out = b_1)
cn_2.node_ID = 2
cn_3 = ComputationNode(0, 'GEMM', tensor_fm = a_1, tensor_w = w_3, tensor_out = b_1)
cn_3.node_ID = 3
workload.add_edge(cn_2, cn_3)

cn_4 = ComputationNode(0, 'GEMM', tensor_fm = a_2, tensor_w = w, tensor_out = b_2)
cn_4.node_ID = 4
cn_5 = ComputationNode(0, 'GEMM', tensor_fm = a_3, tensor_w = w_2, tensor_out = b_2)
cn_5.node_ID = 5
workload.add_edge(cn_4, cn_5)

cn_6 = ComputationNode(0, 'GEMM', tensor_fm = a_2, tensor_w = w_1, tensor_out = b_3)
cn_6.node_ID = 6
cn_7 = ComputationNode(0, 'GEMM', tensor_fm = a_3, tensor_w = w_3, tensor_out = b_3)
cn_7.node_ID = 7
workload.add_edge(cn_6, cn_7)

#bn layer
cn_8 = ComputationNode(1, 'BatchNorm', tensor_fm = b, tensor_out = c)
cn_8.node_ID = 8
workload.add_edge(cn_0, cn_8)
workload.add_edge(cn_1, cn_8)

cn_9 = ComputationNode(1, 'BatchNorm', tensor_fm = b_1, tensor_out = c_1)
cn_9.node_ID = 9
workload.add_edge(cn_2, cn_9)
workload.add_edge(cn_3, cn_9)

cn_10 = ComputationNode(1, 'BatchNorm', tensor_fm = b_2, tensor_out = c_2)
cn_10.node_ID = 10
workload.add_edge(cn_4, cn_10)
workload.add_edge(cn_5, cn_10)

cn_11 = ComputationNode(1, 'BatchNorm', tensor_fm = b_3, tensor_out = c_3)
cn_11.node_ID = 11
workload.add_edge(cn_6, cn_11)
workload.add_edge(cn_7, cn_11)

#relu layer
cn_12 = ComputationNode(2, 'Relu', tensor_fm = c, tensor_out = d)
cn_12.node_ID = 12
workload.add_edge(cn_8, cn_12)

cn_13 = ComputationNode(2, 'Relu', tensor_fm = c_1, tensor_out = d_1)
cn_13.node_ID = 13
workload.add_edge(cn_9, cn_13)

cn_14 = ComputationNode(2, 'Relu', tensor_fm = c_2, tensor_out = d_2)
cn_14.node_ID = 14
workload.add_edge(cn_10, cn_14)

cn_15 = ComputationNode(2, 'Relu', tensor_fm = c_3, tensor_out = d_3)
cn_15.node_ID = 15
workload.add_edge(cn_11, cn_15)

core_0 = Core(0)
core_1 = Core(1)
core_2 = Core(2)
core_3 = Core(3)

core_allocation = [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3]

acc = Accelerater([core_0, core_1, core_2, core_3])
CM = CostModel(workload, acc)
CM.set_core_allocation(core_allocation)
for item in CM.scheduled_cn :
    print(item)