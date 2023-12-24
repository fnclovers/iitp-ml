#include "ftl/ml-prediction/interface/workload_monitor.hh"

#include "ftl/ml-prediction/interface/predictor.hh"
#include "ftl/ml-prediction/interface/predictor_impl.hh"
#include "ftl/predictor_interface.hh"

namespace SimpleSSD::ML {

WorkloadMonitor::WorkloadMonitor(ObjectData &o)
    : Object(o),
      pPredictor(nullptr),
      lastRequestTime(0),
      lastRead(UINT64_MAX) {}

WorkloadMonitor::~WorkloadMonitor() {
  warn_if(!window.empty(), "WorkloadMonitor | window is not empty %lu",
          window.size());
}

void WorkloadMonitor::init(CoReadPredictor *pp) {
  pPredictor = pp;
}

void WorkloadMonitor::enqueueNewRequest(bool isRead, uint64_t tag, LPN slpn,
                                        uint32_t nlp) {
  uint64_t now = getTick();

  // Note that enqueueNewRequest can be called on filling phase
  lastRequestTime = now;
  lastRead = tag;

  if (!isRead) {
    return;
  }

  auto ret = window.emplace(tag, WindowEntry(slpn, nlp, now));
  if (!ret.second) {
    abort();
  }

  lastRead = tag;
}

void WorkloadMonitor::dequeueNewRequest(uint64_t tag) {
  auto iter = window.find(tag);
  if (iter == window.end()) {
    return;
  }

  pPredictor->pModel->train(std::move(iter->second));
  window.erase(iter);
}

void WorkloadMonitor::getStatList(std::vector<Stat> &list,
                                  std::string prefix) noexcept {
  UNUSED(list);
  UNUSED(prefix);
}

void WorkloadMonitor::getStatValues(std::vector<double> &values) noexcept {
  UNUSED(values);
}

void WorkloadMonitor::resetStatValues() noexcept {}

}  // namespace SimpleSSD::ML
