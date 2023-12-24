#include "ftl/ml-prediction/interface/mlp_model.hh"

namespace SimpleSSD::ML {

MLPModel::MLPModel(ObjectData &o) : AbstractMLModel(o) {}

MLPModel::~MLPModel() {}

void MLPModel::init(CoReadPredictor *pp, uint64_t tlp) {
  AbstractMLModel::init(pp, tlp);
  totalLogicalPages = tlp;

  std::string modelDir = readConfigString(SimpleSSD::Section::Simulation,
                                          SimpleSSD::Config::Key::MLFilePath);
  std::ifstream predictTableFile;
  predictTableFile.open(modelDir);

  if (!predictTableFile.is_open()) {
    panic("Cannot open predict table file %s", modelDir);
  }

  // consider SSD capacity
  uint64_t numEntry;
  predictTableFile.read((char *)&numEntry, sizeof(uint64_t));

  panic_if(sizeof(LPN) != sizeof(uint64_t), "LPN not 64bit");
  for (uint i = 0; i < numEntry; i++) {
    auto entry = readEntry(predictTableFile);
    const LPN lpn = entry.targetLPN;

    auto iter = predictTable.find(lpn);
    if (LIKELY(iter == predictTable.end())) {
      // add new entry
      predictTable.emplace(lpn, std::move(entry));
    }
  }
}

CoReadPrediction MLPModel::readEntry(std::ifstream &predictionFile) {
  panic_if(predictionFile.eof(), "readEntry() called too many times");
  CoReadPrediction out;

  // see README.md for binary file format
  std::array<uint64_t, 2> buf;
  predictionFile.read((char *)buf.data(), 2 * sizeof(uint64_t));

  out.targetLPN = LPN{buf[0] % totalLogicalPages};
  out.interval = buf[1];

  out.hit = true;
  return out;
}

void MLPModel::inferenceDone(uint64_t, uint64_t tag) {
  auto iterInf = inferenceQueue.find(tag);
  panic_if(iterInf == inferenceQueue.end(), "inference tag not found %" PRIu64,
           tag);
  const LPN lpn = iterInf->second.lpn;

  CoReadPrediction pred(lpn);
  auto iter = predictTable.find(lpn);
  if (iter != predictTable.end()) {
    const auto &entry = iter->second;
    pred.interval = entry.interval;
    pred.hit = true;
  }

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