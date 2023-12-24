#pragma once
#include "ftl/predictor_interface.hh"

namespace SimpleSSD::ML {

class MLPModel : public AbstractMLModel {
 protected:
  std::unordered_map<uint64_t, CoReadPrediction> predictTable;

  uint64_t totalLogicalPages;
  CoReadPrediction readEntry(std::ifstream &predictionFile);

 public:
  MLPModel(ObjectData &);
  ~MLPModel();

  void init(CoReadPredictor *, uint64_t totalLogicalPages);
  void inferenceDone(uint64_t, uint64_t) override;
  void trainingDone(uint64_t, uint64_t) override;
};

}