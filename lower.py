from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(input: T.Buffer((3136, 64), "float32"), weights_0: T.Buffer((64, 64), "float32"), weights_1: T.Buffer((576, 64), "float32"), weights_2: T.Buffer((64, 256), "float32"), weights_shortcut: T.Buffer((64, 256), "float32"), mean_0: T.Buffer((64,), "float32"), mean_1: T.Buffer((64,), "float32"), mean_2: T.Buffer((64,), "float32"), mean_shortcut: T.Buffer((64,), "float32"), var_0: T.Buffer((64,), "float32"), var_1: T.Buffer((64,), "float32"), var_2: T.Buffer((64,), "float32"), var_shortcut: T.Buffer((64,), "float32"), gamma_0: T.Buffer((64,), "float32"), gamma_1: T.Buffer((64,), "float32"), gamma_2: T.Buffer((64,), "float32"), gamma_shortcut: T.Buffer((64,), "float32"), beta_0: T.Buffer((64,), "float32"), beta_1: T.Buffer((64,), "float32"), beta_2: T.Buffer((64,), "float32"), beta_shortcut: T.Buffer((64,), "float32"), relu: T.Buffer((3136, 256), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        matmul = T.allocate([802816], "float32", "global")
        compute = T.allocate([1806336], "float32", "global")
        batch_normalization = T.allocate([802816], "float32", "global")
        matmul_1 = T.Buffer((28224,), data=matmul)
        input_1 = T.Buffer((200704,), data=input.data)
        for i_outer, i_inner, j_inner in T.grid(196, 16, 9):
            matmul_1[i_outer * 144 + i_inner * 9 + j_inner] = T.float32(0)
            for k_outer, k_inner in T.grid(4, 16):
                cse_var_1: T.int32 = i_outer * 144 + i_inner * 9 + j_inner
                weights_0_1 = T.Buffer((4096,), data=weights_0.data)
                matmul_1[cse_var_1] = matmul_1[cse_var_1] + input_1[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_0_1[k_outer * 1024 + k_inner * 64 + j_inner]
        matmul_2 = T.Buffer((28224,), data=matmul)
        for i_outer, i_inner, j_inner in T.grid(196, 16, 9):
            cse_var_2: T.int32 = i_outer * 144 + i_inner * 9 + j_inner
            mean_0_1 = T.Buffer((64,), data=mean_0.data)
            var_0_1 = T.Buffer((64,), data=var_0.data)
            gamma_0_1 = T.Buffer((64,), data=gamma_0.data)
            beta_0_1 = T.Buffer((64,), data=beta_0.data)
            matmul_2[cse_var_2] = (matmul_1[cse_var_2] - mean_0_1[j_inner]) / T.sqrt(var_0_1[j_inner] + T.float32(1.0000000000000001e-05)) * gamma_0_1[j_inner] + beta_0_1[j_inner]
        matmul_3 = T.Buffer((28224,), data=matmul)
        for i_outer, i_inner, j_inner in T.grid(196, 16, 9):
            cse_var_3: T.int32 = i_outer * 144 + i_inner * 9 + j_inner
            matmul_3[cse_var_3] = T.max(matmul_2[cse_var_3], T.float32(0))
        compute_1 = T.Buffer((1806336,), data=compute)
        for i, j in T.grid(3136, 576):
            compute_1[i * 576 + j] = matmul_3[i * 9 + j % 9]
        matmul_4 = T.Buffer((200704,), data=matmul)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 4, 16):
            matmul_4[i_outer * 1024 + i_inner * 64 + j_outer * 16 + j_inner] = T.float32(0)
            for k_outer, k_inner in T.grid(36, 16):
                cse_var_5: T.int32 = j_outer * 16
                cse_var_4: T.int32 = i_outer * 1024 + i_inner * 64 + cse_var_5 + j_inner
                weights_1_1 = T.Buffer((36864,), data=weights_1.data)
                matmul_4[cse_var_4] = matmul_4[cse_var_4] + compute_1[i_outer * 9216 + i_inner * 576 + k_outer * 16 + k_inner] * weights_1_1[k_outer * 1024 + k_inner * 64 + cse_var_5 + j_inner]
        matmul_5 = T.Buffer((200704,), data=matmul)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 4, 16):
            cse_var_8: T.int32 = j_outer * 16
            cse_var_7: T.int32 = cse_var_8 + j_inner
            cse_var_6: T.int32 = i_outer * 1024 + i_inner * 64 + cse_var_8 + j_inner
            mean_1_1 = T.Buffer((64,), data=mean_1.data)
            var_1_1 = T.Buffer((64,), data=var_1.data)
            gamma_1_1 = T.Buffer((64,), data=gamma_1.data)
            beta_1_1 = T.Buffer((64,), data=beta_1.data)
            matmul_5[cse_var_6] = (matmul_4[cse_var_6] - mean_1_1[cse_var_7]) / T.sqrt(var_1_1[cse_var_7] + T.float32(1.0000000000000001e-05)) * gamma_1_1[cse_var_7] + beta_1_1[cse_var_7]
        matmul_6 = T.Buffer((200704,), data=matmul)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 4, 16):
            cse_var_9: T.int32 = i_outer * 1024 + i_inner * 64 + j_outer * 16 + j_inner
            matmul_6[cse_var_9] = T.max(matmul_5[cse_var_9], T.float32(0))
        compute_2 = T.Buffer((802816,), data=compute)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 16, 16):
            compute_2[i_outer * 4096 + i_inner * 256 + j_outer * 16 + j_inner] = T.float32(0)
            for k_outer, k_inner in T.grid(4, 16):
                cse_var_11: T.int32 = j_outer * 16
                cse_var_10: T.int32 = i_outer * 4096 + i_inner * 256 + cse_var_11 + j_inner
                weights_2_1 = T.Buffer((16384,), data=weights_2.data)
                compute_2[cse_var_10] = compute_2[cse_var_10] + matmul_6[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_2_1[k_outer * 4096 + k_inner * 256 + cse_var_11 + j_inner]
        matmul_7 = T.Buffer((802816,), data=matmul)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 16, 16):
            cse_var_14: T.int32 = j_outer * 16
            cse_var_13: T.int32 = cse_var_14 + j_inner
            cse_var_12: T.int32 = i_outer * 4096 + i_inner * 256 + cse_var_14 + j_inner
            mean_2_1 = T.Buffer((64,), data=mean_2.data)
            var_2_1 = T.Buffer((64,), data=var_2.data)
            gamma_2_1 = T.Buffer((64,), data=gamma_2.data)
            beta_2_1 = T.Buffer((64,), data=beta_2.data)
            matmul_7[cse_var_12] = (compute_2[cse_var_12] - mean_2_1[cse_var_13]) / T.sqrt(var_2_1[cse_var_13] + T.float32(1.0000000000000001e-05)) * gamma_2_1[cse_var_13] + beta_2_1[cse_var_13]
        compute_3 = T.Buffer((802816,), data=compute)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 16, 16):
            compute_3[i_outer * 4096 + i_inner * 256 + j_outer * 16 + j_inner] = T.float32(0)
            for k_outer, k_inner in T.grid(4, 16):
                cse_var_16: T.int32 = j_outer * 16
                cse_var_15: T.int32 = i_outer * 4096 + i_inner * 256 + cse_var_16 + j_inner
                weights_shortcut_1 = T.Buffer((16384,), data=weights_shortcut.data)
                compute_3[cse_var_15] = compute_3[cse_var_15] + input_1[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_shortcut_1[k_outer * 4096 + k_inner * 256 + cse_var_16 + j_inner]
        batch_normalization_1 = T.Buffer((802816,), data=batch_normalization)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 16, 16):
            cse_var_19: T.int32 = j_outer * 16
            cse_var_18: T.int32 = cse_var_19 + j_inner
            cse_var_17: T.int32 = i_outer * 4096 + i_inner * 256 + cse_var_19 + j_inner
            mean_shortcut_1 = T.Buffer((64,), data=mean_shortcut.data)
            var_shortcut_1 = T.Buffer((64,), data=var_shortcut.data)
            gamma_shortcut_1 = T.Buffer((64,), data=gamma_shortcut.data)
            beta_shortcut_1 = T.Buffer((64,), data=beta_shortcut.data)
            batch_normalization_1[cse_var_17] = (compute_3[cse_var_17] - mean_shortcut_1[cse_var_18]) / T.sqrt(var_shortcut_1[cse_var_18] + T.float32(1.0000000000000001e-05)) * gamma_shortcut_1[cse_var_18] + beta_shortcut_1[cse_var_18]
        matmul_8 = T.Buffer((802816,), data=matmul)
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 16, 16):
            cse_var_20: T.int32 = i_outer * 4096 + i_inner * 256 + j_outer * 16 + j_inner
            matmul_8[cse_var_20] = matmul_7[cse_var_20] + batch_normalization_1[cse_var_20]
        for i_outer, i_inner, j_outer, j_inner in T.grid(196, 16, 16, 16):
            cse_var_21: T.int32 = i_outer * 4096 + i_inner * 256 + j_outer * 16 + j_inner
            relu_1 = T.Buffer((802816,), data=relu.data)
            relu_1[cse_var_21] = T.max(matmul_8[cse_var_21], T.float32(0))