#include "ftl/ml-prediction/interface/predictor.hh"

#include "ftl/ml-prediction/interface/ml_model.hh"
#include "ftl/ml-prediction/interface/predictor_impl.hh"
#include "sim/object.hh"

namespace SimpleSSD::ML {

uint64_t MLObject::getTick() noexcept {
  return pObject->cpu->getTick();
}

template <class... T>
void MLObject::debugprint(const char *format, T... args) noexcept {
  pObject->log->debugprint(Log::DebugID::ML, format, args...);
}

template <class... T>
void MLObject::warn_log(const char *format, T... args) noexcept {
  pObject->log->print(Log::LogID::Warn, format, args...);
}

template <class... T>
void MLObject::panic_log(const char *format, T... args) noexcept {
  pObject->log->print(Log::LogID::Panic, format, args...);
}

CoReadPredictor::CoReadPredictor(ObjectData *po)
    : MLObject(po), predictionCnt(0) {
  const uint8_t modelMode = po->config->readUint(
      SimpleSSD::Section::Simulation, SimpleSSD::Config::Key::MLModel);

  switch (modelMode) {
    case 0:
      pModel = new FileModel(*po);
      break;

    case 1:
      pModel = new FileMLModel(*po);
      break;

    default:
      pModel = nullptr;
      panic_log("invalid model");
      break;
  }

  const uint8_t historyTableMode = po->config->readUint(
      SimpleSSD::Section::Simulation, SimpleSSD::Config::Key::MLTable);
  const uint64_t numEntry = po->config->readUint(
      SimpleSSD::Section::Simulation, SimpleSSD::Config::Key::MLNumTableEntry);

  switch (historyTableMode) {
    case 0:
      pTable = new NoopHistoryTable(*po);
      break;
    case 1:
      pTable = new SpaceSavingHistoryTable(*po, numEntry);
      break;
    case 2:
      pTable = new LRUHistoryTable(*po, numEntry);
      break;
    case 3:
      pTable = new AccurateWritePredict(*po, numEntry);
      break;
    default:
      pTable = nullptr;
      panic_log("invalid history model");
      break;
  }

  pMonitor = new WorkloadMonitor(*po);

  predMode = (Config::MLPredictionType)po->config->readUint(
      Section::Simulation, Config::Key::MLPredictionMode);

  resetStatValues();
}

CoReadPredictor::~CoReadPredictor() {
  delete pModel;
  delete pTable;
  delete pMonitor;
}

void CoReadPredictor::init(uint64_t totalLogicalPages) {
  pModel->init(this, totalLogicalPages);
  pTable->init(this, totalLogicalPages);
  pMonitor->init(this);
}

uint64_t CoReadPredictor::startPrediction(LPN lpn, bool init, bool isBG) {
  const uint64_t tag = predictionCnt;
  pModel->inference(lpn, init, tag, isBG);

  predictionCnt++;
  return tag;
}

bool CoReadPredictor::tryGetPrediction(uint64_t tag,
                                       CoReadPrediction &prediction) {
  panic_if(tag == UINT64_MAX, "invalid ML Tag");

  auto iter = predictionBuf.find(tag);

  if (iter != predictionBuf.end()) {
    if (getTick() > 0) {
      debugprint("get prediction success | LPN %" PRIx64 "h | Tag %" PRIu64,
                 iter->second.targetLPN, tag);
    }
    prediction = std::move(iter->second);
    predictionBuf.erase(iter);

    // update statistics
    if (prediction.hit) {
      stat.hit++;
    }
    stat.numInference++;

    return true;
  }
  else {
    debugprint("get prediction fail tag %" PRIu64, tag);
    return false;
  }
}

void CoReadPredictor::directPrediction(LPN lpn, uint32_t &intraLeft,
                                       uint32_t &intraRight) {
  pModel->directPrediction(lpn, intraLeft, intraRight);
}

bool CoReadPredictor::direct_get(LPN lpn, std::vector<LPN> &history, int idx) {
  return pTable->direct_get(lpn, history, idx);
}

bool CoReadPredictor::curr_read_predict(LPN lpn, std::vector<LPN> &history,
                                        int idx) {
  return pTable->curr_read_predict(lpn, history, idx);
}

void CoReadPredictor::update(LPN lpn, uint32_t nlp, bool isRead) {
  pTable->update(lpn, nlp, isRead);
}

void CoReadPredictor::enqueueNewRequest(bool isRead, uint64_t tag, LPN slpn,
                                        uint32_t nlp) {
  pMonitor->enqueueNewRequest(isRead, tag, slpn, nlp);
}
void CoReadPredictor::dequeueNewRequest(uint64_t tag) {
  pMonitor->dequeueNewRequest(tag);
}

void CoReadPredictor::getStatList(std::vector<Stat> &list,
                                  std::string prefix) noexcept {
  list.emplace_back(prefix + "hit_rate", "inference hit rate");
  pModel->getStatList(list, prefix);
  pTable->getStatList(list, prefix);
  pMonitor->getStatList(list, prefix);
}

void CoReadPredictor::getStatValues(std::vector<double> &values) noexcept {
  values.push_back((double)stat.hit / stat.numInference);
  pModel->getStatValues(values);
  pTable->getStatValues(values);
  pMonitor->getStatValues(values);
}

void CoReadPredictor::resetStatValues() noexcept {
  memset(&stat, 0, sizeof(stat));

  pModel->resetStatValues();
  pTable->resetStatValues();
  pMonitor->resetStatValues();
}

}  // namespace SimpleSSD::ML
