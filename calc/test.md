```cpp
              if(nccl_info->algorithm == NCCL_ALGO_RING) {
                std::shared_ptr<MockNccl::FlowModels> RingFlowModels = std::static_pointer_cast<MockNccl::FlowModels>(ptr_FlowModels);
                std::map<int,std::map<int,std::vector<int>>> channels;
                {
                  Sys::sysCriticalSection cs;
                  channels = mock_nccl_comms[comm_ps]->get_rings();
                  cs.ExitSection();
                }
                NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d generate FlowModels",id);
                if(RingFlowModels != nullptr){
                  NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d NcclMock generate  %d channel and flow model count:  %d",id,channels.size(),RingFlowModels->size());
                  for (auto flow : *RingFlowModels) {
                    int prev;
                    int parent_flow_id;
                    int child_flow_id;
                    if (flow.second.prev.size() == 0) {
                      prev = -1;
                    } else {
                      prev = flow.second.prev[0];
                    }
                    if (flow.second.child_flow_id.size() == 0) {
                      child_flow_id = -1;
                    } else {
                      child_flow_id = flow.second.child_flow_id[0];
                    }
                    if (flow.second.parent_flow_id.size() == 0) {
                      parent_flow_id = -1;
                    } else {
                      parent_flow_id = flow.second.parent_flow_id[0];
                    }
                    NcclLog->writeLog(NcclLogLevel::DEBUG," %d,  %d,  %d to  %d current_flow_id %d prev rank:  %d parent_flow_id:  %d child_flow_id:  %d chunk_id:  %d flow_size: %lu chunk_count:  %d ",flow.first.first,flow.first.second,flow.second.src,flow.second.dest,flow.second.flow_id,prev,parent_flow_id,child_flow_id,flow.second.chunk_id,flow.second.flow_size,flow.second.chunk_count);
                  }
                }
                CollectivePhase vn(
                    this,
                    queue_id,
                    new NcclTreeFlowModel(
                        collective_type,
                        id,
                        layer_num,
                        (RingTopology*)topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        RingFlowModels,
                        channels.size()));
                return vn;
              }  else if(nccl_info->algorithm == NCCL_ALGO_NVLS) {
                collective_type = ComType::All_Reduce_NVLS;
                std::shared_ptr<MockNccl::FlowModels> RingFlowModels = std::static_pointer_cast<MockNccl::FlowModels>(ptr_FlowModels);
                MockNccl::TreeChannels treechannels;
                {
                  Sys::sysCriticalSection cs;
                  treechannels = mock_nccl_comms[comm_ps]->get_treechannels();
                  cs.ExitSection();
                }
                NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d generate FlowModels",id);
                if(RingFlowModels != nullptr){
                  NcclLog->writeLog(NcclLogLevel::DEBUG,"rank %d NcclMock generate  %d channel and flow model count:  %d",id,treechannels.size(),RingFlowModels->size());
                  for (auto flow : *RingFlowModels) {
                    int prev;
                    int parent_flow_id;
                    int child_flow_id;
                    if (flow.second.prev.size() == 0) {
                      prev = -1;
                    } else {
                      prev = flow.second.prev[0];
                    }
                    if (flow.second.child_flow_id.size() == 0) {
                      child_flow_id = -1;
                    } else {
                      child_flow_id = flow.second.child_flow_id[0];
                    }
                    if (flow.second.parent_flow_id.size() == 0) {
                      parent_flow_id = -1;
                    } else {
                      parent_flow_id = flow.second.parent_flow_id[0];
                    }
                    NcclLog->writeLog(NcclLogLevel::DEBUG," %d,  %d,  %d to  %d current_flow_id %d prev rank:  %d parent_flow_id:  %d child_flow_id:  %d chunk_id:  %d flow_size: %lu chunk_count:  %d ",flow.first.first,flow.first.second,flow.second.src,flow.second.dest,flow.second.flow_id,prev,parent_flow_id,child_flow_id,flow.second.chunk_id,flow.second.flow_size,flow.second.chunk_count);
                  }
                }
                CollectivePhase vn(
                    this,
                    queue_id,
                    new NcclTreeFlowModel(
                        collective_type,
                        id,
                        layer_num,
                        (RingTopology*)topology,
                        data_size,
                        direction,
                        injection_policy,
                        boost_mode,
                        RingFlowModels,
                        treechannels.size()));
                return vn;
              }
```