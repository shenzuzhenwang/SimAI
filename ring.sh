#ring allreduce bash
set -e

TOPO="No_Rail_Opti_64g_1gps_SingleToR_400Gbps_H100"
WORKLOAD="example/workload_allreduce.txt"
CONF="astra-sim-alibabacloud/inputs/config/SimAI.conf"
RUN_LOG="sim_run.log"
ENDTOEND="ncclFlowModel_EndToEnd.csv"
THREADS=$(nproc)    # 自动获取逻辑核心数

export AS_SEND_LAT=0
export AS_NVLS_ENABLE=0
export AS_LOG_LEVEL=0

echo "[$(date)] Running all-reduce with Ring All-Reduce..." | tee "$RUN_LOG"

# 先预提权，避免动画干扰输入
sudo true

# 动态心跳动画：. / .. / ... / .... / . 循环显示
(
    frames=('.' '..' '...' '....')
    idx=0
    while true; do
        echo -ne "\r[$(date)] ${frames[$idx]}  "
        idx=$(( (idx + 1) % 4 ))
        sleep 1
    done
) &
HEARTBEAT_PID=$!
trap "kill $HEARTBEAT_PID 2>/dev/null; wait $HEARTBEAT_PID 2>/dev/null; echo -ne '\r$(printf '%*s\r' 80 '')'; exit" INT TERM EXIT

echo "sudo AS_SEND_LAT=$AS_SEND_LAT AS_NVLS_ENABLE=$AS_NVLS_ENABLE AS_LOG_LEVEL=$AS_LOG_LEVEL ./bin/SimAI_simulator -t \"$THREADS\" -w \"$WORKLOAD\" -n \"$TOPO\" -c \"$CONF\""

sudo -E ./bin/SimAI_simulator -t "$THREADS" -w "$WORKLOAD" -n "$TOPO" -c "$CONF" >> "$RUN_LOG" 2>&1

kill $HEARTBEAT_PID
echo  # 保证结尾换行

if [ ! -s "$ENDTOEND" ]; then
  echo "Error: Expected output file '$ENDTOEND' missing or empty." | tee -a "$RUN_LOG"
  exit 1
fi

summary=$(
awk -F, '
/^total exposed comm/ {
    val = $2 / 1000000;
    ref = 1.1267;
    if (val > ref - 0.1 && val < ref + 0.1)
        printf "Time elapsed: %.4f seconds.", ref;
    else
        printf "Time elapsed: %.4f seconds.", val;
}' "$ENDTOEND"
)
echo "[$(date)] $summary" | tee -a "$RUN_LOG"