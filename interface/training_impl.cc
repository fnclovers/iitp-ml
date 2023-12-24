#include "ftl/ml-prediction/interface/training_impl.hh"

#undef panic_if
#undef panic
#undef warn_if
#undef warn
#undef info

#include <torch/torch.h>

#include <iostream>

namespace SimpleSSD::ML {

// Define the MLP model
struct MLPModel : ::torch::nn::Module {
  ::torch::nn::Linear layer1{nullptr}, layer2{nullptr};

  MLPModel() {
    // Initialize layers
    layer1 = register_module("layer1", ::torch::nn::Linear(1, 64));
    layer2 = register_module("layer2", ::torch::nn::Linear(64, 1));
  }

  // Forward function
  ::torch::Tensor forward(::torch::Tensor x) {
    x = ::torch::relu(layer1->forward(x));
    x = layer2->forward(x);
    return x;
  }
};

// Training class

MLPTrainig::MLPTrainig(ObjectData &o)
    : AbstractTrainig(o),
      optimizer(new ::torch::optim::Adam(model->parameters(),
                                         ::torch::optim::AdamOptions(0.001))) {
  model = new MLPModel();
}

MLPTrainig::~MLPTrainig() {
  delete model;
  delete optimizer;
}

// Online training method
void MLPTrainig::add(WindowEntry &entry) {
  model->train();
  optimizer->zero_grad();
  auto input = ::torch::tensor({static_cast<float>(entry.slpn)});
  auto target = ::torch::tensor({static_cast<float>(entry.issued)});
  auto output = model->forward(input);
  auto loss = ::torch::mse_loss(output, target);
  loss.backward();
  optimizer->step();
}

// Inference method
bool MLPTrainig::get(LPN lpn, CoReadPrediction &prediction) {
  model->eval();
  ::torch::NoGradGuard no_grad;
  auto input = ::torch::tensor({static_cast<float>(lpn)});
  auto output = model->forward(input);
  prediction.interval = static_cast<uint64_t>(output.item<float>());
  return true;
}

}  // namespace SimpleSSD::ML
