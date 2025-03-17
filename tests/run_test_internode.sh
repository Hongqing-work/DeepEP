#!/bin/bash

WORK_ROOT=/root/paddlejob/workspace/env_run/liuyiqun
export PYTHONPATH=${WORK_ROOT}/env/virtualenvs_cuda12.8/torch_py310_yiqun
export PATH=${PYTHONPATH}/bin:${PATH}

python -c "import torch; print(torch.__version__)"

export PYTHONPATH=${WORK_ROOT}/PaPerf:$PYTHONPATH

#export NVSHMEM_DIR=$ROOT_DIR/third-party/nvshmem
#export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"

export MASTER_ADDR=10.54.95.204
export MASTER_PORT=8367
export WORLD_SIZE=8

START_NODE=0
END_NODE=$((${START_NODE} + ${WORLD_SIZE}))
export RANK=$(($PADDLE_TRAINER_ID - ${START_NODE}))

if [ ${PADDLE_TRAINER_ID} -lt ${START_NODE} ]; then
  echo "$PADDLE_TRAINER_ID exit"
  exit
elif [ ${PADDLE_TRAINER_ID} -ge ${END_NODE} ]; then
  echo "$PADDLE_TRAINER_ID exit"
  exit
fi

export NCCL_DEBUG=WARN
#export NVSHMEM_DEBUG=DEBUG
#export NVSHMEM_DEBUG=TRACE

# 保证集群稳定性的配置，跟性能无关
export NCCL_IB_QPS_PER_CONNECTION=8 
#export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
# 开启AR功能
#export NCCL_IB_ADAPTIVE_ROUTING=1

export NCCL_IB_GID_INDEX=3
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_TRAFFIC_CLASS=162

#export NVSHMEM_IB_ENABLE_IBGDA=true
#export NVSHMEM_DISABLE_P2P=1
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME==xgbe0
#export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET

#export NVSHMEM_DEBUG=INFO

export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin:$PATH
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_simple_kernel_${WORLD_SIZE}.torch"
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx --nic-metrics=true --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_internode_${WORLD_SIZE}nodes_rank${RANK}.torch"

rm -rf core.*

${nsys_args} python test_internode.py
#${nsys_args} python test_simple.py
