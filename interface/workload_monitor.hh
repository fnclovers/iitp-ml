#pragma once

#ifndef __SIM_CO_READ_WORKLOAD_MONITOR_HH__
#define __SIM_CO_READ_WORKLOAD_MONITOR_HH__

#include <deque>

#include "ftl/predictor_interface.hh"
#include "sim/object.hh"
namespace SimpleSSD::ML {

class CoReadPredictor;

// monitor average read size
// monitor co-read window
class WorkloadMonitor : public Object {
 private:
  CoReadPredictor *pPredictor;

  std::unordered_map<uint64_t, WindowEntry> window;
  uint64_t lastRequestTime;
  uint64_t lastRead;

 public:
  WorkloadMonitor(ObjectData &o);
  ~WorkloadMonitor();

  void init(CoReadPredictor *);

  void enqueueNewRequest(bool isRead, uint64_t tag, LPN slpn, uint32_t nlp);
  void dequeueNewRequest(uint64_t tag);

  void getStatList(std::vector<Stat> &, std::string) noexcept;
  void getStatValues(std::vector<double> &) noexcept;
  void resetStatValues() noexcept;

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

}  // namespace SimpleSSD::ML

#endif