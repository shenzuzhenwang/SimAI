#!/bin/bash

sudo AS_SEND_LAT=0 AS_NVLS_ENABLE=0 ./bin/SimAI_simulator -t 4 -w ./example/28gallreduce.txt -n ./No_Rail_Opti_64g_1gps_SingleToR_400Gbps_H100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf
