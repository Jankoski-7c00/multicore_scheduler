from classes.multigpu import MultiGPU
from networkx import DiGraph

def list_scheduling_algorithm(DIG: DiGraph, system: MultiGPU):
    workload = DIG.copy()
    candidate_cn_table = []
    memory_usage = []

    scheduled_cn_map = {}
    for i in range(system.gpu_num):
        scheduled_cn_map[i] = []
    
    gpu_timeline = {}
    for i in range(system.gpu_num):
        gpu_timeline[i] = 0

    def find_start_time(cn):
        start_time = 0
        for pre_cn in DIG.predecessors(cn):
            for _, scheduled_cns in scheduled_cn_map.items():
                for i in scheduled_cns:
                    if i[0] == pre_cn:
                        if i[2] > start_time:
                            start_time = i[2]

        return start_time
    
    while workload.number_of_nodes() != 0 :
        #Step 1: upgrade candidate CN table
        for CN in workload.nodes() :
            if workload.in_degree(CN) == 0 :
                if CN not in candidate_cn_table:
                    candidate_cn_table.append(CN)
        
        #Step 2: choose best candidate CN
        best_CN = candidate_cn_table[0]

        #Step 3: schedule best candidate CN
        start_time = find_start_time(best_CN)
        gpu_id = best_CN.core_allocation
        start_time = max(gpu_timeline[gpu_id], start_time)
        runtime = system.runtime_default(best_CN, gpu_id)
        memory = system.memory_usage()
        end_time = start_time + runtime
        scheduled_cn_map[gpu_id].append([best_CN, start_time, end_time])
        memory_usage.append([end_time, memory])
        
        gpu_timeline[gpu_id] = end_time

        candidate_cn_table.remove(best_CN)
        workload.remove_node(best_CN)

    memory_usage.sort(key=lambda x: x[0])

    return scheduled_cn_map, memory_usage