#!/bin/bash

LOG_DIR=/home/log/dsv4flash_4_3p1d
mkdir -p "$LOG_DIR"
exec >> "${LOG_DIR}/vllm-dsv4pro-d.log" 2>&1

unset ftp_proxy
unset https_proxy
unset http_proxy
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

TIME=$(date +"%Y%m%d_%H%M%S")
BAK_ROOT="/home/shell/dsv41flash_4_3p1d/logbak"
BAK_DIR="${BAK_ROOT}/log_${TIME}"
mkdir -p "$BAK_DIR"
cp -r ~/ascend/log "$BAK_DIR"
rm -rf ~/ascend/log
export HCCL_IF_BASE_PORT=50000

nic_name="bond4"
local_ip=10.73.68.14

jemalloc_path=$(find /usr/lib /usr/lib64 -maxdepth 2 -type f -name "libjemalloc.so.2" 2>/dev/null | head -n 1)

if [[ -n "$jemalloc_path" ]]; then
    export LD_PRELOAD="${jemalloc_path}:${LD_PRELOAD}"
    echo "jemalloc found at: $jemalloc_path"
    echo "LD_PRELOAD is set successfully."
else
    echo "Warning: libjemalloc.so.2 not found under /usr"
    echo "Please make sure jemalloc is installed."
fi

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

# # AIV
export HCCL_OP_EXPANSION_MODE="AIV"

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=1200


export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1800
export ASCEND_RT_VISIBLE_DEVICES=$1

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
export PYTHONHASHSEED=0
export MOONCAKE_CONFIG_PATH="/home/shell/dsv41flash_4_3p1d/dsv4KVpool/mooncake_d.json"
export HCCL_INTRA_ROCE_ENABLE=1 
export ASCEND_CONNECT_TIMEOUT=120000
export ASCEND_TRANSFER_TIMEOUT=120000
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=36000


vllm serve /data/models/DeepSeek-V4.1-Flash-1017-w4a8 \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --prefill-context-parallel-size $8 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name deepseek_v41_flash \
    --max-model-len 1048576 \
    --max-num-batched-tokens 400 \
    --max-num-seqs 32 \
    --async-scheduling \
    --block-size 32 \
    --no-enable-prefix-caching \
    --no-swa-bounded-replay \
    --tokenizer-mode deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --safetensors-load-strategy lazy \
    --trust-remote-code \
    --default-chat-template-kwargs '{"thinking": true,"reasoning_effort": "low"}' \
    --gpu-memory-utilization 0.95 \
    --kv-cache-memory-bytes 4600000000 \
    --quantization ascend \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
	'{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_port": "30400",
    "engine_id": "4",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_consumer",
                "kv_port": "30400",
                "kv_connector_extra_config": {
                            "prefill": {"dp_size": 2, "tp_size": 2},
                            "decode": {"dp_size": 8, "tp_size": 2}
                    }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "mooncake",
                    "use_layerwise":false
                }
            }  
        ]
    }
    }' \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{
        "ascend_compilation_config":{
              "enable_npugraph_ex":true,
              "enable_static_kernel":false
        },
       "enable_cpu_binding":true,
       "enable_flashcomm1":false,
       "multistream_overlap_shared_expert":true,
       "recompute_scheduler_enable":true
    }'
