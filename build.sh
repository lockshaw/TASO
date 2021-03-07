#! /usr/bin/env bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/build"
# export PROTOBUF="$HOME/libs/protobuf-3.15.3/"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PROTOBUF/src/.libs"
#CUDA="/usr/local/cuda-11.1/"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PROTOBUF/src/.libs/"
#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA/lib64/"
# export PATH="$PROTOBUF/src/.libs/:$PATH"
#export PATH="$CUDA/bin/:$PATH"
# export CUDACXX="$CUDA/bin/nvcc"
#cmake -DProtobuf_INCLUDE_DIR="$PROTOBUF/src/" -DCUDA_CUDNN_LIBRARY="$HOME/libs/cudnn/lib64/libcudnn.so" -DProtobuf_LIBRARY="$PROTOBUF/src/.libs/libprotobuf.so" -DBUILD_CPP_EXAMPLES=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug .. -DCUDNN_INCLUDE_DIR="$HOME/libs/cudnn/include/"
#cmake -DCUDA_CUDNN_LIBRARY="$HOME/libs/cudnn/lib64/libcudnn.so" -DBUILD_CPP_EXAMPLES=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug .. -DCUDNN_INCLUDE_DIR="$HOME/libs/cudnn/include/"
cmake -DBUILD_CPP_EXAMPLES=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_FIND_DEBUG_MODE=1 ..
make -j6 VERBOSE=1

cd "$DIR/python/"
rm -rf build/
python3 setup.py install --user

RUN="${RUN:-none}"
case "$RUN" in
  test)
    gdb -ex 'catch throw; run' --args "$DIR/build/test/run_tests" --break --abort
    ;;
  example)
    cd "$DIR/build/cpp_examples"
    gdb -ex run --args ./dnn --dnn alexnet --budget "${BUDGET:-0}" --export test.onnx
    ;;
esac

