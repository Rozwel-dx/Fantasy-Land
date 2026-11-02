  #!/usr/bin/env bash
  # bench_sweep.sh
  set -euo pipefail

  # 内网访问必须绕过公司代理，否则 httpx/aiohttp 会被 HIS 代理截断成 504
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  export NO_PROXY=127.0.0.1,localhost,141.61.133.136,141.61.133.137
  export no_proxy=$NO_PROXY

  # ========= 可配置变量 =========
  MODEL_PATH="/mnt/share/weight/DeepSeek-V4.1-Flash-1017"
  SERVED_NAME="dsv4_1"
  BASE_URL="http://172.27.18.115:8113"
  ENDPOINT="/v1/chat/completions"

  # 你要扫的 request-rate 列表（可含小数或 inf）
  REQUEST_RATES=(10 12 14 16)
  NUM_PROMPTS=500

  # 输出目录
  OUT_DIR="./bench_results_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT_DIR"
  echo "Results will be saved to: $OUT_DIR"
  echo "Request rates: ${REQUEST_RATES[*]}"
  echo

  # 汇总 CSV
  SUMMARY_CSV="$OUT_DIR/summary.csv"
  echo "request_rate,num_prompts,cache_salt,log_file,result_json" > "$SUMMARY_CSV"

  # ========= 随机 cache_salt 生成（0-9999） =========
  USED_SALTS=" "

  gen_salt() {
    local salt tries=0
    while :; do
      salt=$(printf "run-%04d" $((RANDOM % 10000)))
      case "$USED_SALTS" in
        *" $salt "*) ;;
        *) USED_SALTS="$USED_SALTS$salt "; echo "$salt"; return 0 ;;
      esac
      tries=$((tries + 1))
      if [ "$tries" -ge 20 ]; then
        USED_SALTS="$USED_SALTS$salt "
        echo "$salt"
        return 0
      fi
    done
  }

  # ========= 主循环 =========
  for RATE in "${REQUEST_RATES[@]}"; do
    TAG=$(echo "$RATE" | tr '.' 'p')
    LOG_FILE="$OUT_DIR/log_rate_${TAG}.txt"

    CACHE_SALT=$(gen_salt)

    echo "=================================================="
    echo ">>> Testing request-rate = $RATE  (cache_salt = $CACHE_SALT)"
    echo "=================================================="

    CMD=(vllm bench serve
      --backend openai-chat
      --model "$MODEL_PATH"
      --served-model-name "$SERVED_NAME"
      --tokenizer "$MODEL_PATH"
      --percentile-metrics ttft,tpot,itl,e2el
      --metric-percentiles 50,95,99
      --dataset-name prefix_repetition
      --request-rate "$RATE"
      --num-prompts "$NUM_PROMPTS"
      --endpoint "$ENDPOINT"
      --base-url "$BASE_URL"
      --ignore-eos
      --temperature 1
      --seed 10801
      --prefix-repetition-prefix-len 87040
      --prefix-repetition-suffix-len 15360
      --prefix-repetition-output-len 600
      --prefix-repetition-num-prefixes 20
      --extra-body "{\"cache_salt\":\"$CACHE_SALT\"}"
      --save-result
      --result-dir "$OUT_DIR"
    )

    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
      RESULT_JSON=$(ls -t "$OUT_DIR"/*.json 2>/dev/null | head -n1 || true)
      echo "$RATE,$NUM_PROMPTS,$CACHE_SALT,$LOG_FILE,${RESULT_JSON:-none}" >> "$SUMMARY_CSV"
      echo ">>> rate=$RATE (cache_salt=$CACHE_SALT) done."
    else
      echo "!!! ERROR: rate=$RATE (cache_salt=$CACHE_SALT) failed, continue..." >&2
      echo "$RATE,$NUM_PROMPTS,$CACHE_SALT,$LOG_FILE,FAILED" >> "$SUMMARY_CSV"
    fi

    echo
  done

  echo "=================================================="
  echo "All finished. Summary: $SUMMARY_CSV"
  echo "=================================================="

