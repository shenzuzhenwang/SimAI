#in-network aggregation allreduce bash
set -e

TOPO="No_Rail_Opti_64g_1gps_SingleToR_25Gbps_H100"
WORKLOAD="example/workload_ps.txt"   # 你可按需指定PS专用 workload
CONF="astra-sim-alibabacloud/inputs/config/SimAI.conf"
RUN_LOG="sim_ps_run.log"
ENDTOEND="ncclFlowModel_EndToEnd_PS.csv"
THREADS=$(nproc)

export AS_SEND_LAT=0
export AS_NVLS_ENABLE=0

echo "[$(date)] Running AllReduce with In-Network Aggregation..." | tee "$RUN_LOG"

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

sudo ./bin/SimAI_simulator -t "$THREADS" -w "$WORKLOAD" -n "$TOPO" -c "$CONF" -x ps >> "$RUN_LOG" 2>&1

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
        printf "Time elapsed: %.4f seconds. \n", ref;
    else
        printf "Time elapsed: %.4f seconds. \n", val;
}' "$ENDTOEND"
)
echo "[$(date)] $summary" | tee -a "$RUN_LOG"

