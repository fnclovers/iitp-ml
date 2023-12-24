#include "ftl/ml-prediction/interface/training_impl.hh"

namespace SimpleSSD::ML {

MLPTrainig::MLPTrainig(ObjectData &o) : AbstractTrainig(o) {}

MLPTrainig::~MLPTrainig() {}

void MLPTrainig::add(WindowEntry &window) {
  UNUSED(window);
}

bool MLPTrainig::get(LPN lpn, CoReadPrediction &prediction) {
  UNUSED(lpn);
  UNUSED(prediction);
  return true;
}

}  // namespace SimpleSSD::ML
