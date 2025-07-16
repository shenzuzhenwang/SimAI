#!/bin/bash

cd ../
AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 4 -w ./example/16Mallreduce.txt -n ./No_Rail_Opti_64g_1gps_SingleToR_25Gibps_H100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf
