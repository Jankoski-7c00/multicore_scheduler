from tvm.script import ir as I
from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(input: T.Buffer((3136, 64), "float32"), weights_0: T.Buffer((64, 64), "float32"), weights_1: T.Buffer((576, 64), "float32"), weights_2: T.Buffer((64, 256), "float32"), weights_shortcut: T.Buffer((64, 256), "float32"), mean_0: T.Buffer((64,), "float32"), mean_1: T.Buffer((64,), "float32"), mean_2: T.Buffer((64,), "float32"), mean_shortcut: T.Buffer((64,), "float32"), var_0: T.Buffer((64,), "float32"), var_1: T.Buffer((64,), "float32"), var_2: T.Buffer((64,), "float32"), var_shortcut: T.Buffer((64,), "float32"), gamma_0: T.Buffer((64,), "float32"), gamma_1: T.Buffer((64,), "float32"), gamma_2: T.Buffer((64,), "float32"), gamma_shortcut: T.Buffer((64,), "float32"), beta_0: T.Buffer((64,), "float32"), beta_1: T.Buffer((64,), "float32"), beta_2: T.Buffer((64,), "float32"), beta_shortcut: T.Buffer((64,), "float32"), relu: T.Buffer((3136, 256), "float32")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        relu_1 = T.allocate([200704], "float32", "global")
        matmul = T.allocate([16], "float32", "global")
        compute = T.allocate([1806336], "float32", "global")
        input_1 = T.Buffer((200704,), data=input.data)
        relu_2 = T.Buffer((28224,), data=relu_1)
        for i_outer, i_inner in T.grid(196, 16):
            matmul_1 = T.Buffer((9,), data=matmul, align=32)
            for j_inner in range(9):
                matmul_1[j_inner] = T.float32(0)
                for k_outer, k_inner in T.grid(4, 16):
                    weights_0_1 = T.Buffer((4096,), data=weights_0.data)
                    matmul_1[j_inner] = matmul_1[j_inner] + input_1[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_0_1[k_outer * 1024 + k_inner * 64 + j_inner]
            matmul_2 = T.Buffer((9,), data=matmul, align=32)
            for j_inner in range(9):
                mean_0_1 = T.Buffer((64,), data=mean_0.data)
                var_0_1 = T.Buffer((64,), data=var_0.data)
                gamma_0_1 = T.Buffer((64,), data=gamma_0.data)
                beta_0_1 = T.Buffer((64,), data=beta_0.data)
                matmul_2[j_inner] = (matmul_1[j_inner] - mean_0_1[j_inner]) / T.sqrt(var_0_1[j_inner] + T.float32(1.0000000000000001e-05)) * gamma_0_1[j_inner] + beta_0_1[j_inner]
            for j_inner in range(9):
                relu_2[i_outer * 144 + i_inner * 9 + j_inner] = T.max(matmul_2[j_inner], T.float32(0))
        compute_1 = T.Buffer((1806336,), data=compute)
        for i, j in T.grid(3136, 576):
            compute_1[i * 576 + j] = relu_2[i * 9 + j % 9]
        relu_3 = T.Buffer((200704,), data=relu_1)
        for i_outer, i_inner, j_outer in T.grid(196, 16, 4):
            matmul_1 = T.Buffer((16,), data=matmul)
            for j_inner in range(16):
                matmul_1[j_inner] = T.float32(0)
                for k_outer, k_inner in T.grid(36, 16):
                    weights_1_1 = T.Buffer((36864,), data=weights_1.data)
                    matmul_1[j_inner] = matmul_1[j_inner] + compute_1[i_outer * 9216 + i_inner * 576 + k_outer * 16 + k_inner] * weights_1_1[k_outer * 1024 + k_inner * 64 + j_outer * 16 + j_inner]
            matmul_2 = T.Buffer((16,), data=matmul)
            for j_inner in range(16):
                cse_var_1: T.int32 = j_outer * 16 + j_inner
                mean_1_1 = T.Buffer((64,), data=mean_1.data)
                var_1_1 = T.Buffer((64,), data=var_1.data)
                gamma_1_1 = T.Buffer((64,), data=gamma_1.data)
                beta_1_1 = T.Buffer((64,), data=beta_1.data)
                matmul_2[j_inner] = (matmul_1[j_inner] - mean_1_1[cse_var_1]) / T.sqrt(var_1_1[cse_var_1] + T.float32(1.0000000000000001e-05)) * gamma_1_1[cse_var_1] + beta_1_1[cse_var_1]
            for j_inner in range(16):
                relu_3[i_outer * 1024 + i_inner * 64 + j_outer * 16 + j_inner] = T.max(matmul_2[j_inner], T.float32(0))
        compute_2 = T.Buffer((802816,), data=compute)
        for i_outer, i_inner, j_outer in T.grid(196, 16, 16):
            matmul_1 = T.Buffer((16,), data=matmul)
            for j_inner in range(16):
                matmul_1[j_inner] = T.float32(0)
                for k_outer, k_inner in T.grid(4, 16):
                    weights_2_1 = T.Buffer((16384,), data=weights_2.data)
                    matmul_1[j_inner] = matmul_1[j_inner] + relu_3[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_2_1[k_outer * 4096 + k_inner * 256 + j_outer * 16 + j_inner]
            for j_inner in range(16):
                cse_var_3: T.int32 = j_outer * 16
                cse_var_2: T.int32 = cse_var_3 + j_inner
                mean_2_1 = T.Buffer((64,), data=mean_2.data)
                var_2_1 = T.Buffer((64,), data=var_2.data)
                gamma_2_1 = T.Buffer((64,), data=gamma_2.data)
                beta_2_1 = T.Buffer((64,), data=beta_2.data)
                compute_2[i_outer * 4096 + i_inner * 256 + cse_var_3 + j_inner] = (matmul_1[j_inner] - mean_2_1[cse_var_2]) / T.sqrt(var_2_1[cse_var_2] + T.float32(1.0000000000000001e-05)) * gamma_2_1[cse_var_2] + beta_2_1[cse_var_2]
        for i_outer, i_inner, j_outer in T.grid(196, 16, 16):
            matmul_1 = T.Buffer((16,), data=matmul)
            for j_inner in range(16):
                matmul_1[j_inner] = T.float32(0)
                for k_outer, k_inner in T.grid(4, 16):
                    weights_shortcut_1 = T.Buffer((16384,), data=weights_shortcut.data)
                    matmul_1[j_inner] = matmul_1[j_inner] + input_1[i_outer * 1024 + i_inner * 64 + k_outer * 16 + k_inner] * weights_shortcut_1[k_outer * 4096 + k_inner * 256 + j_outer * 16 + j_inner]
            matmul_2 = T.Buffer((16,), data=matmul)
            for j_inner in range(16):
                cse_var_4: T.int32 = j_outer * 16 + j_inner
                mean_shortcut_1 = T.Buffer((64,), data=mean_shortcut.data)
                var_shortcut_1 = T.Buffer((64,), data=var_shortcut.data)
                gamma_shortcut_1 = T.Buffer((64,), data=gamma_shortcut.data)
                beta_shortcut_1 = T.Buffer((64,), data=beta_shortcut.data)
                matmul_2[j_inner] = (matmul_1[j_inner] - mean_shortcut_1[cse_var_4]) / T.sqrt(var_shortcut_1[cse_var_4] + T.float32(1.0000000000000001e-05)) * gamma_shortcut_1[cse_var_4] + beta_shortcut_1[cse_var_4]
            matmul_3 = T.Buffer((16,), data=matmul)
            for j_inner in range(16):
                matmul_3[j_inner] = compute_2[i_outer * 4096 + i_inner * 256 + j_outer * 16 + j_inner] + matmul_2[j_inner]
            for j_inner in range(16):
                relu_4 = T.Buffer((802816,), data=relu.data)
                relu_4[i_outer * 4096 + i_inner * 256 + j_outer * 16 + j_inner] = T.max(matmul_3[j_inner], T.float32(0))