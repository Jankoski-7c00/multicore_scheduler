from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(input: T.Buffer((3136, 64), "float32"), weights_0: T.Buffer((64, 64), "float32"), weights_1: T.Buffer((576, 64), "float32"), weights_2: T.Buffer((64, 256), "float32"), weights_shortcut: T.Buffer((64, 256), "float32"), mean_0: T.Buffer((64,), "float32"), mean_1: T.Buffer((64,), "float32"), mean_2: T.Buffer((64,), "float32"), mean_shortcut: T.Buffer((64,), "float32"), var_0: T.Buffer((64,), "float32"), var_1: T.Buffer((64,), "float32"), var_2: T.Buffer((64,), "float32"), var_shortcut: T.Buffer((64,), "float32"), gamma_0: T.Buffer((64,), "float32"), gamma_1: T.Buffer((64,), "float32"), gamma_2: T.Buffer((64,), "float32"), gamma_shortcut: T.Buffer((64,), "float32"), beta_0: T.Buffer((64,), "float32"), beta_1: T.Buffer((64,), "float32"), beta_2: T.Buffer((64,), "float32"), beta_shortcut: T.Buffer((64,), "float32"), relu: T.Buffer((3136, 256), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        matmul = T.allocate([802816], "float32", "global")
        compute = T.allocate([1806336], "float32", "global")
        normalized_data = T.allocate([802816], "float32", "global")
        matmul_1 = T.Buffer((28224,), data=matmul)
        input_1 = T.Buffer((200704,), data=input.data)
        for i_outer in range(196):
            for i_inner_init, j_inner_init in T.grid(16, 9):
                matmul_1[i_outer * 144 + i_inner_init * 9 + j_inner_init] = T.float32(0)
            for k_outer, i_inner, j_inner, k_inner in T.grid(4, 16, 9, 16):
                cse_var_1: T.int32 = i_outer * 144 + i_inner * 9 + j_inner
                weights_0_1 = T.Buffer((4096,), data=weights_0.data)
                matmul_1[cse_var_1] = matmul_1[cse_var_1] + input_1[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_0_1[k_outer * 1024 + k_inner * 64 + j_inner]
        matmul_2 = T.Buffer((28224,), data=matmul)
        for a, b in T.grid(3136, 9):
            cse_var_2: T.int32 = a * 9 + b
            matmul_2[cse_var_2] = T.max(matmul_1[cse_var_2], T.float32(0))
        compute_1 = T.Buffer((1806336,), data=compute)
        for a, b in T.grid(3136, 576):
            compute_1[a * 576 + b] = matmul_2[a * 9 + b % 9]
        matmul_3 = T.Buffer((200704,), data=matmul)
        for i_outer, j_outer in T.grid(196, 4):
            for i_inner_init, j_inner_init in T.grid(16, 16):
                matmul_3[i_outer * 1024 + i_inner_init * 64 + j_outer * 16 + j_inner_init] = T.float32(0)
            for k_outer, i_inner, j_inner, k_inner in T.grid(36, 16, 16, 16):
                cse_var_4: T.int32 = j_outer * 16
                cse_var_3: T.int32 = i_outer * 1024 + i_inner * 64 + cse_var_4 + j_inner
                weights_1_1 = T.Buffer((36864,), data=weights_1.data)
                matmul_3[cse_var_3] = matmul_3[cse_var_3] + compute_1[i_outer * 9216 + i_inner * 576 + k_outer * 16 + k_inner] * weights_1_1[k_outer * 1024 + k_inner * 64 + cse_var_4 + j_inner]
        matmul_4 = T.Buffer((200704,), data=matmul)
        for a, b in T.grid(3136, 64):
            cse_var_5: T.int32 = a * 64 + b
            matmul_4[cse_var_5] = T.max(matmul_3[cse_var_5], T.float32(0))
        compute_2 = T.Buffer((802816,), data=compute)
        for i_outer, j_outer in T.grid(196, 16):
            for i_inner_init, j_inner_init in T.grid(16, 16):
                compute_2[i_outer * 4096 + i_inner_init * 256 + j_outer * 16 + j_inner_init] = T.float32(0)
            for k_outer, i_inner, j_inner, k_inner in T.grid(4, 16, 16, 16):
                cse_var_7: T.int32 = j_outer * 16
                cse_var_6: T.int32 = i_outer * 4096 + i_inner * 256 + cse_var_7 + j_inner
                weights_2_1 = T.Buffer((16384,), data=weights_2.data)
                compute_2[cse_var_6] = compute_2[cse_var_6] + matmul_4[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_2_1[k_outer * 4096 + k_inner * 256 + cse_var_7 + j_inner]
        matmul_5 = T.Buffer((802816,), data=matmul)
        for i, j in T.grid(3136, 256):
            cse_var_8: T.int32 = i * 256 + j
            mean_2_1 = T.Buffer((64,), data=mean_2.data)
            var_2_1 = T.Buffer((64,), data=var_2.data)
            matmul_5[cse_var_8] = (compute_2[cse_var_8] - mean_2_1[j]) / T.sqrt(var_2_1[j] + T.float32(1.0000000000000001e-05))
        matmul_6 = T.Buffer((802816,), data=matmul)
        for i, j in T.grid(3136, 256):
            cse_var_9: T.int32 = i * 256 + j
            gamma_2_1 = T.Buffer((64,), data=gamma_2.data)
            beta_2_1 = T.Buffer((64,), data=beta_2.data)
            matmul_6[cse_var_9] = matmul_5[cse_var_9] * gamma_2_1[j] + beta_2_1[j]
        compute_3 = T.Buffer((802816,), data=compute)
        for i_outer, j_outer in T.grid(196, 16):
            for i_inner_init, j_inner_init in T.grid(16, 16):
                compute_3[i_outer * 4096 + i_inner_init * 256 + j_outer * 16 + j_inner_init] = T.float32(0)
            for k_outer, i_inner, j_inner, k_inner in T.grid(4, 16, 16, 16):
                cse_var_11: T.int32 = j_outer * 16
                cse_var_10: T.int32 = i_outer * 4096 + i_inner * 256 + cse_var_11 + j_inner
                weights_shortcut_1 = T.Buffer((16384,), data=weights_shortcut.data)
                compute_3[cse_var_10] = compute_3[cse_var_10] + input_1[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_shortcut_1[k_outer * 4096 + k_inner * 256 + cse_var_11 + j_inner]
        normalized_data_1 = T.Buffer((802816,), data=normalized_data)
        for i, j in T.grid(3136, 256):
            cse_var_12: T.int32 = i * 256 + j
            mean_shortcut_1 = T.Buffer((64,), data=mean_shortcut.data)
            var_shortcut_1 = T.Buffer((64,), data=var_shortcut.data)
            normalized_data_1[cse_var_12] = (compute_3[cse_var_12] - mean_shortcut_1[j]) / T.sqrt(var_shortcut_1[j] + T.float32(1.0000000000000001e-05))
        normalized_data_2 = T.Buffer((802816,), data=normalized_data)
        for i, j in T.grid(3136, 256):
            cse_var_13: T.int32 = i * 256 + j
            gamma_shortcut_1 = T.Buffer((64,), data=gamma_shortcut.data)
            beta_shortcut_1 = T.Buffer((64,), data=beta_shortcut.data)
            normalized_data_2[cse_var_13] = normalized_data_1[cse_var_13] * gamma_shortcut_1[j] + beta_shortcut_1[j]
        matmul_7 = T.Buffer((802816,), data=matmul)
        for i, j in T.grid(3136, 256):
            cse_var_14: T.int32 = i * 256 + j
            matmul_7[cse_var_14] = matmul_6[cse_var_14] + normalized_data_2[cse_var_14]
        for a, b in T.grid(3136, 256):
            cse_var_15: T.int32 = a * 256 + b
            relu_1 = T.Buffer((802816,), data=relu.data)
            relu_1[cse_var_15] = T.max(matmul_7[cse_var_15], T.float32(0))