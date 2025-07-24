/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "DataSet.hh"
#include "IntData.hh"
#include "MockNcclLog.h"
#include "Sys.hh"
namespace AstraSim
{
int DataSet::id_auto_increment = 0;
DataSet::DataSet(int total_streams)
{
    this->my_id = id_auto_increment++;
    this->total_streams = total_streams;
    this->finished_streams = 0;
    this->finished = false;
    this->finish_tick = 0;
    this->active = true;
    this->creation_tick = Sys::boostedTick();
    this->notifier = nullptr;
}
void DataSet::set_notifier(Callable *layer, EventType event) { notifier = new std::pair<Callable *, EventType>(layer, event); }
// 当一个流完成，统计完成流数，必要时触发回调
void DataSet::notify_stream_finished(StreamStat *data)
{
    MockNcclLog *NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "stream_finished dataset: %d stream: %d total: %d", my_id, finished_streams + 1, total_streams);
    finished_streams++; // 累加已完成流数
    if (data != nullptr)
    {
        update_stream_stats(data); // 更新流统计信息
    }
    // 如果所有流都完成
    if (finished_streams == total_streams)
    {
        finished = true;                  // 标记数据集已完成
        finish_tick = Sys::boostedTick(); // 记录完成时刻
        if (notifier != nullptr)
        {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "total_stream_finished dataset %d type %d call %p", my_id, notifier->second, notifier->first);
            take_stream_stats_average();     // 统计平均流信息
            Callable *c = notifier->first;   // 获取回调对象
            EventType ev = notifier->second; // 获取事件类型
            delete notifier;                 // 释放回调资源
            c->call(ev, new IntData(my_id)); // 触发回调Layer::call，通知上层数据集已完成
        }
        else
        {
            NcclLog->writeLog(NcclLogLevel::ERROR, "notify_stream_finished notifier = nullptr ");
        }
    }
}
void DataSet::call(EventType event, CallData *data) { notify_stream_finished(((StreamStat *)data)); }
bool DataSet::is_finished() { return finished; }
} // namespace AstraSim
