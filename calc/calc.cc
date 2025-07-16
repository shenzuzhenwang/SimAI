#include <iostream>
#include <climits>
#include <cstdint>

using namespace std;

int main() {
    // 输入参数
    double delay_us;       // 微秒
    uint32_t m_size;       // Byte
    double bw_gbps;        // Gbps

    cout << "请输入 delay (us): ";
    cin >> delay_us;

    cout << "请输入 m_size (Byte): ";
    cin >> m_size;

    cout << "请输入 bw (Gbps): ";
    cin >> bw_gbps;

    // 转换单位
    uint64_t bw_bps = static_cast<uint64_t>(bw_gbps * 1e9);  // 带宽 bps
    uint64_t delay_ns = static_cast<uint64_t>(delay_us * 1000); // delay 转 ns

    uint64_t best_fct = ULLONG_MAX;
    int best_payload = -1;

    for (int p = 256; p <= 65534; p += 1) {
        // txDelay = packet_payload_size * 8 * 1e9 / bw (ns)
        uint64_t txDelay = 2*p * 1000000000lu * 8 / bw_bps;

        uint64_t base_rtt = 2 * delay_ns + txDelay;

        uint64_t num_pkts = (m_size - 1) / p + 1;
        uint64_t total_bytes = m_size + num_pkts * 52;

        uint64_t trans_delay = total_bytes * 8000000000lu / bw_bps;

        uint64_t fct = base_rtt + trans_delay;

        if (fct < best_fct) {
            best_fct = fct;
            best_payload = p;
        }
    }

    cout << "\n最小 standalone_fct = " << best_fct << " ns" << endl;
    cout << "对应的 packet_payload_size = " << best_payload << " Bytes" << endl;

    return 0;
}
