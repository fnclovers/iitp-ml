#pragma once

#ifndef __SIM_CO_READ_PREDICTOR_HH__
#define __SIM_CO_READ_PREDICTOR_HH__

#include "sim/log.hh"

namespace SimpleSSD {

class ObjectData;
namespace ML {

class AbstractMLModel;
class WorkloadMonitor;
class AbstractHistoryTable;
class AbstractTrainig;

class MLObject {
 protected:
  ObjectData *pObject;

  uint64_t getTick() noexcept;

  template <class... T>
  inline void debugprint(const char *format, T... args) noexcept;

  template <class... T>
  inline void warn_log(const char *format, T... args) noexcept;

  template <class... T>
  inline void panic_log(const char *format, T... args) noexcept;

 public:
  MLObject(ObjectData *po) : pObject(po) {}
  virtual ~MLObject() {}

  MLObject(const MLObject &) = delete;
  MLObject(MLObject &&) noexcept = delete;
  MLObject &operator=(const MLObject &) = delete;
  MLObject &operator=(MLObject &&) = delete;
};

class CoReadPredictor : public MLObject {
 public:
  friend AbstractMLModel;
  friend WorkloadMonitor;

  WorkloadMonitor *pMonitor;
  AbstractTrainig *pTable;
  AbstractMLModel *pModel;

  std::unordered_map<uint64_t, CoReadPrediction> predictionBuf;

  uint64_t predictionCnt;

  struct {
    uint64_t numInference;
    uint64_t hit;
  } stat;

 public:
  CoReadPredictor(ObjectData *);
  ~CoReadPredictor();

  void init(uint64_t totalLogicalPages);

  // called by HIL, start prediction using LPN
  // init : are we initializing the mapping? --> if true, prediction will be
  // done immediately
  // returns tag
  uint64_t startPrediction(LPN, bool init = false, bool isBG = false);
  // called by FTL, get prediction if prediction is done
  // return true if prediction done, false otherwise
  bool tryGetPrediction(uint64_t tag, CoReadPrediction &prediction);

  // called by FTL, informing a request handling is started/ended
  void enqueueNewRequest(bool isRead, uint64_t tag, LPN slpn, uint32_t nlp);
  void dequeueNewRequest(uint64_t tag);

  void getStatList(std::vector<Stat> &, std::string) noexcept;
  void getStatValues(std::vector<double> &) noexcept;
  void resetStatValues() noexcept;
};

}  // namespace ML
}  // namespace SimpleSSD

#endif