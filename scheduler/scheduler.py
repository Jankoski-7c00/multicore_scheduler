from classes import Accelerater
from networkx import DiGraph

def scheduler(accelerater: Accelerater, workload: DiGraph, strategy: str):
    '''nothing'''

    #Step 0: initialize three tables
    core_idle_timetable = []
    candidate_cn_table = []
    scheduled_cn_table = []
    
    on_chip_memory_usage = []

    for i in range(len(accelerater.cores)) :
        core_idle_timetable.append([f'core{i}', None, 0, 0])#core, CN, start time, end time
        on_chip_memory_usage.append([f'core{i}', []])#core, memory usage at some certain clk
    

    while workload.number_of_nodes() != 0 :
        #Step 1: upgrade candidate CN table
        for CN in workload.nodes() :
            if workload.in_degree(CN) == 0 :
                candidate_cn_table.append(CN)

        #Step 2: choose best candidate CN
        best_CN = candidate_cn_table[0]
        if strategy == 'Memory' :
            for cn in candidate_cn_table :
                if cn.layer > best_CN.layer :
                    best_CN = cn

        if strategy == 'Latency' :
            for cn in candidate_cn_table :
                cn_start = core_idle_timetable[cn.core_allocation][3]
                if cn_start < core_idle_timetable[best_CN.core_allocation][3] :
                    best_CN = cn

        #Step 3: upgrade core idle timetable
        core_id = best_CN.core_allocation
        start_time = core_idle_timetable[core_id][3]
        is_result = workload.out_degree(best_CN) == 0
        end_time = start_time + accelerater.runtime(best_CN, is_result)
        
        core_idle_timetable[core_id][1] = best_CN
        core_idle_timetable[core_id][2] = start_time
        core_idle_timetable[core_id][3] = end_time
        
        memory_usage = accelerater.on_chip_memory_usage(core_id)
        on_chip_memory_usage[core_id][1].append([end_time, memory_usage])

        candidate_cn_table.remove(best_CN)

        #Step 4: upgrade scheduled CN table
        scheduled_cn_table.append([best_CN, start_time, end_time])
        workload.remove_node(best_CN)

    return scheduled_cn_table, on_chip_memory_usage