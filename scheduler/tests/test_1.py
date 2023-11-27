import sys
sys.path.append('/Users/xiangyy/Projects/multicore_schedule/scheduler')
from frontend.onnx_parser import OnnxParser
from frontend.computationgraph import ComputationGraph
from classes.multigpu import MultiGPU
import onnx
from schedule_algorithm.costModel import CostModel
from schedule_algorithm.genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
from backend.gemm import generate_gemm

def generate_workflow(scheduled_cn_map, file_name = 'workflow'):
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

    fig, ax = plt.subplots()

    gpuids = list(scheduled_cn_map.keys())

    ax.set_yticks([i + 0.5 for i in gpuids])
    ax.set_yticklabels([f'GPU {i}' for i in gpuids])

    for idx, (gpuid, work_times) in enumerate(scheduled_cn_map.items()):
        # 为每个工作时间创建一个(y, width)的元组
        bars = [(start, end - start) for _, start, end in work_times]
        # 绘制颜色块
        ax.broken_barh(bars, (idx, 1), facecolors=colors[idx % len(colors)])

    # 设置图表标题和坐标轴标签
    ax.set_title('GPU Work Schedule')
    ax.set_xlabel('Time')
    ax.set_ylabel('GPU ID')

    # 显示图表到文件
    file_path = f'./results/{file_name}.png'
    plt.savefig(file_path)
    plt.close()

def generate_memoryuse(memory_usage, file_name = 'memory_usage'):
    end_times, memories = zip(*memory_usage)

    plt.figure(figsize=(10, 5))
    plt.plot(end_times, memories)

    plt.title('Memory Usage')
    plt.xlabel('Time')
    plt.ylabel('Memory')
    plt.legend(['Memory Usage'])

    plt.grid(True)
    file_path = f'./results/{file_name}.png'
    plt.savefig(file_path)
    plt.close()

#model = onnx.load('/Users/xiangyy/Downloads/resnet50-v1-12.onnx')
model = onnx.load('/Users/xiangyy/Downloads/resnet18-v1-7.onnx')
parser = OnnxParser(model)
cg = ComputationGraph(parser)
DIG = cg.CN_graph
system = MultiGPU(4, {})
individual_length = DIG.number_of_nodes()
cm = CostModel(DIG, system)
gemm_CN = cg.split_layers_dict['resnetv15_dense0_fwd'][2]
print(gemm_CN)
print(cg.CN_num)

func = generate_gemm(gemm_CN)