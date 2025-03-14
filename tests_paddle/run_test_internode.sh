#!/bin/bash

#bash kill_process.sh

WORK_ROOT=/root/paddlejob/workspace/env_run/liuyiqun
export PYTHONPATH=${WORK_ROOT}/env/virtualenvs_cuda12.8/paddle_py310_yiqun
export PATH=${PYTHONPATH}/bin:${PATH}

python -c "import paddle; print(paddle.version.commit)"
python -c "import torch; print(torch.__version__)"

#export NVSHMEM_DIR=${WORK_ROOT}/Paddle/build_paddle/third_party_cuda12.3_gcc12.2.0_py3.10/install/nvshmem
#export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/liuyiqun/env/virtualenvs_cuda12.3/paddle_py310_yiqun/lib/python3.10/site-packages/paddle/libs:$LD_LIBRARY_PATH

export PYTHONPATH=${WORK_ROOT}/PaPerf:$PYTHONPATH
#export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/liuyiqun/DeepEP/tests_paddle:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/liuyiqun/Paddle/paddle/fluid/distributed/collective/deep_ep/kernels/build:$LD_LIBRARY_PATH


# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

export MASTER_ADDR=10.54.95.204
export MASTER_PORT=58978
export WORLD_SIZE=2
#nnodes=$PADDLE_TRAINERS_NUM
export NNODES=${WORLD_SIZE}
export RANK=$PADDLE_TRAINER_ID

if [ ${PADDLE_TRAINER_ID} -ge ${WORLD_SIZE} ]; then
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
#export NVSHMEM_DISABLE_P2P=0
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME==xgbe0
#export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
#export NVSHMEM_IB_ENABLE_RELAXED_ORDERING=0

#export NVSHMEM_CUMEM_GRANULARITY=2M

#export NVSHMEM_DEBUG=INFO
#export GLOG_vmodule=deep_ep=4

#export FLAGS_use_nvml=False

export PATH=/opt/nvidia/nsight-systems/2025.1.1/bin:$PATH
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_internode_${WORLD_SIZE}nodes_rank${RANK}.paddle"
#nsys_args="nsys profile --stats true -w true -t cuda,nvtx --capture-range=cudaProfilerApi -x true --force-overwrite true -o test_simple_${WORLD_SIZE}.paddle"

rm -rf core.*
rm -rf log

${nsys_args} python -m paddle.distributed.launch --master=${MASTER_ADDR}:${MASTER_PORT} --nnodes=${NNODES} --rank ${RANK} test_internode.py
#${nsys_args} python -m paddle.distributed.launch --master=${MASTER_ADDR}:${MASTER_PORT} --nnodes=${NNODES} --rank ${RANK} test_simple.py
