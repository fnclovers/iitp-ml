#pragma once

#include <list>
#include <map>
#include <set>

#include "ftl/predictor_interface.hh"
#include "sim/object.hh"
#include "sim/types.hh"

namespace SimpleSSD::ML {

class AbstractTrainig : public Object {
 protected:
  uint64_t totalLogicalPages;
  CoReadPredictor *pp;

 public:
  AbstractTrainig(ObjectData &o);
  virtual ~AbstractTrainig() = default;

  virtual void init(CoReadPredictor *pp, uint64_t totalLogicalPages) {
    this->pp = pp;
    this->totalLogicalPages = totalLogicalPages;
  }
  virtual void add(WindowEntry &) = 0;
  virtual bool get(LPN, CoReadPrediction) = 0;

  void getStatList(std::vector<Stat> &, std::string) noexcept override {}
  void getStatValues(std::vector<double> &) noexcept override {}
  void resetStatValues() noexcept override {}

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

class MLPTrainig : public AbstractTrainig {
 private:
 public:
  MLPTrainig(ObjectData &o);
  ~MLPTrainig();

  void add(WindowEntry &) override;
  bool get(LPN, CoReadPrediction) override;

  void getStatList(std::vector<Stat> &, std::string) noexcept override {}
  void getStatValues(std::vector<double> &) noexcept override {}
  void resetStatValues() noexcept override {}

  void createCheckpoint(std::ostream &) const noexcept override {}
  void restoreCheckpoint(std::istream &) noexcept override {}
};

}  // namespace SimpleSSD::ML
