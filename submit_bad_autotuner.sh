#!/bin/bash
set -euo pipefail

GNS_PACKAGE="gram-newton-schulz"
# GNS_PACKAGE="gram-newton-schulz @ git+https://github.com/NoahAmsel/gram-newton-schulz@debug-nan"

echo "Using package version (from web): $GNS_PACKAGE"

kubectl cp /home/t-noahamsel/aifsdk.worktrees/submitter/phitrain/scripts/train/test-ns.py \
  bonete61/cjob-uploader:/data/t-noahamsel/scripts/test-ns.py

JOB_NAME=$(cat <<YAMLEOF | kubectl create -f - -o jsonpath='{.metadata.name}'
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  generateName: t-noahamsel-test-ns-
  namespace: bonete61
spec:
  queue: bonete61
  minAvailable: 1
  plugins: {ssh: [], svc: [], env: []}
  tasks:
    - name: master
      replicas: 1
      policies: [{event: TaskCompleted, action: CompleteJob}]
      template:
        spec:
          schedulerName: volcano
          restartPolicy: Never
          nodeSelector: {nvidia.com/gpu.product: NVIDIA-B200}
          volumes:
            - name: dshm
              emptyDir: {medium: Memory, sizeLimit: 100Gi}
            - name: data
              persistentVolumeClaim: {claimName: pvc-vast-bonete61}
          tolerations:
            - {key: rdma, operator: Exists, effect: NoSchedule}
            - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
          containers:
            - name: master
              image: nvcr.io/nvidia/pytorch:25.08-py3
              imagePullPolicy: IfNotPresent
              volumeMounts:
                - {name: dshm, mountPath: /dev/shm}
                - {name: data, mountPath: /data}
              env: [{name: PYTHONUNBUFFERED, value: "1"}, {name: QUACK_PRINT_AUTOTUNING, value: "1"}]
              command: ["/bin/bash", "-c"]
              args: ["pip install '${GNS_PACKAGE}' && python3 /data/t-noahamsel/scripts/test-ns.py"]
              resources:
                requests: &res {nvidia.com/gpu: "1", cpu: "4", memory: 64Gi}
                limits: *res
YAMLEOF
)

echo "Created job: $JOB_NAME"
echo "Monitor with:"
echo "  kubectl logs -f -n bonete61 -l volcano.sh/job-name=$JOB_NAME"
