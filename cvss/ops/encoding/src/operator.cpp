#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregate_forward", &Aggregate_Forward_CUDA, "Aggregate forward (CUDA)");
  m.def("aggregate_backward", &Aggregate_Backward_CUDA, "Aggregate backward (CUDA)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CUDA, "ScaledL2 forward (CUDA)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CUDA, "ScaledL2 backward (CUDA)");
  m.def("batchnorm_forward", &BatchNorm_Forward_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_inp_forward", &BatchNorm_Forward_Inp_CUDA, "BatchNorm forward (CUDA)");
  m.def("batchnorm_backward", &BatchNorm_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("batchnorm_inp_backward", &BatchNorm_Inp_Backward_CUDA, "BatchNorm backward (CUDA)");
  m.def("expectation_forward", &Expectation_Forward_CUDA, "Expectation forward (CUDA)");
  m.def("expectation_backward", &Expectation_Backward_CUDA, "Expectation backward (CUDA)");
  m.def("expectation_inp_backward", &Expectation_Inp_Backward_CUDA,
        "Inplace Expectation backward (CUDA)");
  m.def("leaky_relu_forward", &LeakyRelu_Forward_CUDA, "Learky ReLU forward (CUDA)");
  m.def("leaky_relu_backward", &LeakyRelu_Backward_CUDA, "Learky ReLU backward (CUDA)");
  m.def("conv_rectify", &CONV_RECTIFY_CUDA, "Convolution Rectifier (CUDA)");
}
