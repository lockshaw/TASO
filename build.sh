#! /usr/bin/env bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/build"
PROTOBUF="$HOME/FlexFlow/protobuf/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PROTOBUF/src/.libs"
cmake -DProtobuf_INCLUDE_DIR="$PROTOBUF/src/" -DProtobuf_PROTOC_EXECUTABLE="$PROTOBUF/src/.libs/protoc" -DCUDA_CUDNN_LIBRARY='/share/software/user/open/cudnn/7.6.5/lib64/libcudnn.so' -DProtobuf_LIBRARY="$PROTOBUF/src/.libs/libprotobuf.so" -DBUILD_CPP_EXAMPLES=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
#cmake -DProtobuf_INCLUDE_DIR=/share/software/user/open/protobuf/3.4.0/include/ -DCUDA_CUDNN_LIBRARY='/share/software/user/open/cudnn/7.6.5/lib64/libcudnn.so' -DProtobuf_LIBRARY='/share/software/user/open/protobuf/3.4.0/lib/libprotobuf.so' -DBUILD_CPP_EXAMPLES=ON ..
make -j6 VERBOSE=1
cd "$DIR/python/"
rm -rf build/
python3 setup.py install --user
