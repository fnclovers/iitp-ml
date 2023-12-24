#include "ftl/ml-prediction/interface/mlp_model.hh"

#include "ftl/ml-prediction/interface/predictor.hh"
#include "ftl/ml-prediction/interface/predictor_impl.hh"
#include "ftl/ml-prediction/interface/training_impl.hh"
#include "ftl/ml-prediction/interface/workload_monitor.hh"

namespace SimpleSSD::ML {

MLPModel::MLPModel(ObjectData &o) : AbstractMLModel(o) {}

MLPModel::~MLPModel() {}

void MLPModel::init(CoReadPredictor *pp, uint64_t tlp) {
  AbstractMLModel::init(pp, tlp);
  totalLogicalPages = tlp;
}

void MLPModel::inferenceDone(uint64_t, uint64_t tag) {
  auto iterInf = inferenceQueue.find(tag);
  panic_if(iterInf == inferenceQueue.end(), "inference tag not found %" PRIu64,
           tag);
  const LPN lpn = iterInf->second.lpn;

  CoReadPrediction pred(lpn);
  pPredictor->pTable->get(lpn, pred);

  if (iterInf->second.delayedByGC == UINT64_MAX) {
    BGDepth--;
  }
  else {
    UserDepth--;
  }
  storePrediction(tag, std::move(pred));

#if !DEBUG_AT_INIT
  if (getTick() > 0)
#endif
    debugprint(Log::DebugID::ML,
               "inference done | LPN %" PRIx64 "h | Tag %" PRIu64 " | hit", lpn,
               tag);

  inferenceQueue.erase(iterInf);
}

void MLPModel::trainingDone(uint64_t, uint64_t tag) {
  auto iter = trainingQueue.find(tag);

  if (getTick() > 0)
    debugprint(Log::DebugID::ML, "trainingDone | tag %" PRIu64, tag);
  // TODO : do some training

  storeHistory(iter->second);

  trainingQueue.erase(iter);
}

}  // namespace SimpleSSD::ML