#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cassert>
#include <unordered_map>
#include<map>
#include<iostream>
using namespace std;

class SimSetting{
public:
	std::map<uint16_t, std::unordered_map<uint8_t, uint64_t> > port_speed; // port_speed[i][j] is node i's j-th port's speed
	uint32_t win; // window bound

	void Serialize(FILE* file){
		// write port_speed
		uint32_t len = 0;
		for (auto i: port_speed)
			for (auto j : i.second)
				len++;
		fwrite(&len, sizeof(len), 1, file);
		for (auto i: port_speed)
			for (auto j : i.second){
				fwrite(&i.first, sizeof(i.first), 1, file);
				fwrite(&j.first, sizeof(j.first), 1, file);
				fwrite(&j.second, sizeof(j.second), 1, file);
			}
		// write win
		fwrite(&win, sizeof(win), 1, file);
	}
	void Deserialize(FILE *file){
		int ret;
		// read port_speed
		uint32_t len;
		ret = fread(&len, sizeof(len), 1, file);
		for (uint32_t i = 0; i < len; i++){
			uint16_t node;
			uint8_t intf;
			uint64_t bps;
			ret &= fread(&node, sizeof(node), 1, file);
			ret &= fread(&intf, sizeof(intf), 1, file);
			ret &= fread(&bps, sizeof(bps), 1, file);
			port_speed[node][intf] = bps;
		}
		// read win
		ret &= fread(&win, sizeof(win), 1, file);

		// make sure read successfully
		assert(ret != 0);
	}
    void show() {
        // 打印 port_speed 信息
        std::cout << "Port Speed Settings:\n";
        for (const auto& node_entry : port_speed) {
            uint16_t node_id = node_entry.first;
            const auto& interfaces = node_entry.second;
            
            std::cout << "  Node " << node_id << ":\n";
            for (const auto& intf_entry : interfaces) {
                uint8_t intf_id = intf_entry.first;
                uint64_t speed = intf_entry.second;
                
                std::cout << "    Port " << static_cast<int>(intf_id) 
                        << ": " << speed << " bps\n";
            }
        }
        
        // 打印 win 信息
        std::cout << "Window Bound (win): " << win << "\n";
    }
};

int main(){
    FILE *trace_output = fopen("/etc/astra-sim/simulation/llama_hpn7_mix.tr", "r");
    SimSetting simset;
    simset.Deserialize(trace_output);
    simset.show();

}