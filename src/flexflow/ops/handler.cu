#include "flexflow/config.h"
#include "flexflow/cuda_helper.h"

using namespace flexflow;

void FFHandler::init() {
  checkCUDA(cudaSetDevice(0));
  checkCUDNN(cudnnCreate(&this->dnn));
  checkCUDA(cublasCreate(&this->blas));
  this->workSpaceSize = WORK_SPACE_SIZE;
  checkCUDA(cudaMalloc(&this->workSpace, this->workSpaceSize));
}
