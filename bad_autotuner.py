# To run on B200 cluster:
#   kubectl cp test-ns.py bonete61/cjob-uploader:/data/t-noahamsel/scripts/test-ns.py
#   cat <<'YAMLEOF' | kubectl create -f -
#   apiVersion: batch.volcano.sh/v1alpha1
#   kind: Job
#   metadata:
#     generateName: t-noahamsel-test-ns-
#     namespace: bonete61
#   spec:
#     queue: bonete61
#     minAvailable: 1
#     plugins: {ssh: [], svc: [], env: []}
#     tasks:
#       - name: master
#         replicas: 1
#         policies: [{event: TaskCompleted, action: CompleteJob}]
#         template:
#           spec:
#             schedulerName: volcano
#             restartPolicy: Never
#             nodeSelector: {nvidia.com/gpu.product: NVIDIA-B200}
#             volumes:
#               - name: dshm
#                 emptyDir: {medium: Memory, sizeLimit: 100Gi}
#               - name: data
#                 persistentVolumeClaim: {claimName: pvc-vast-bonete61}
#             tolerations:
#               - {key: rdma, operator: Exists, effect: NoSchedule}
#               - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
#             containers:
#               - name: master
#                 image: nvcr.io/nvidia/pytorch:25.08-py3
#                 imagePullPolicy: IfNotPresent
#                 volumeMounts:
#                   - {name: dshm, mountPath: /dev/shm}
#                   - {name: data, mountPath: /data}
#                 env: [{name: PYTHONUNBUFFERED, value: "1"}, {name: QUACK_PRINT_AUTOTUNING, value: "1"}]
#                 command: ["/bin/bash", "-c"]
#                 args: ["pip install gram-newton-schulz && python3 /data/t-noahamsel/scripts/test-ns.py"]
#                 resources:
#                   requests: &res {nvidia.com/gpu: "1", cpu: "4", memory: 64Gi}
#                   limits: *res
#   YAMLEOF

# FAILURE: autotuner output:
# `quack autotuning for function gemm_tuned finished after 5.22s; best config selected: config: GemmConfig(tile_m=256, tile_n=224, pingpong=False, is_dynamic_persistent=True, cluster_m=2, cluster_n=1, swap_ab=True, max_swizzle_size=8, device_capacity=10);`

from math import sqrt
import torch
from gram_newton_schulz import GramNewtonSchulz, YOU_COEFFICIENTS, POLAR_EXPRESS_COEFFICIENTS
from quack.gemm_interface import gemm_tuned
from quack.gemm_config import GemmConfig
from quack.autotuner import AutotuneConfig
import sys
import quack
import gram_newton_schulz

print(f"Python {sys.version}")
print(torch.__version__)
print(quack.__version__)
print(gram_newton_schulz.__version__)

# Pin to the config from the failing training run (skip autotuner benchmarking)
failing_config = GemmConfig(
    tile_m=256, tile_n=224, pingpong=False, is_dynamic_persistent=True,
    cluster_m=2, cluster_n=1, swap_ab=True, max_swizzle_size=8, device_capacity=10,
)
gemm_tuned.configs = [AutotuneConfig(config=failing_config)]

# sandbox_path = "/data/t-noahamsel/divergences/diverged_input1.pt"
# sandbox_path = "/data/t-noahamsel/phitrain_results/phinextopt-dion-speed-20260409-195348/divergences/diverged_input1.pt"
# sandbox_path = "/data/t-noahamsel/divergences/phinextopt-dion-speed-20260409-200436/diverged_input1.pt"
# sandbox_path = "/data/cjob/worktrees/phinextopt-dion-speed-20260409-200436/training_output/divergences/diverged_input1.pt"
# inp = torch.load(sandbox_path, weights_only=True).to('cuda')
# print(f"Shape: {inp.shape}, Dtype: {inp.dtype}")

gns = GramNewtonSchulz(ns_use_kernels=True, use_gram_newton_schulz=False, ns_coefficients=YOU_COEFFICIENTS)

# out_std = gns._standard_newton_schulz(inp)
# print(f"Output Norm: {torch.linalg.matrix_norm(out_std)}")
# print(f"Expected output norm: {sqrt(min(*out_std.shape[-2:]))}")

# print(inp.dtype)

# inp2 = torch.randn(21, 2560, 2560, device="cuda")
# out_std = gns(inp2)
# print(f"Output Norm: {torch.linalg.matrix_norm(out_std)}")
# print(f"Expected output norm: {sqrt(min(*out_std.shape[-2:]))}")

inp3 = torch.randn(264, 264, device="cuda")
out_std = gns(inp3)
print(f"Output Norm: {torch.linalg.matrix_norm(out_std)}")
print(f"Expected output norm: {sqrt(min(*out_std.shape[-2:]))}")