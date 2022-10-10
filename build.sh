#!/bin/bash

find() {
    /usr/local/spack/bin/spack find --format '{prefix}' $@ | head -n1
}

SYCL=$(find hipsycl)
CMAKE=$(find cmake)/bin/cmake

CWD=$PWD

rm -fr build
mkdir build
cd build
export CXXFLAGS="-I/usr/local/include"
$CMAKE -DCMAKE_PREFIX_PATH=$SYCL \
       -DCMAKE_CXX_COMPILER=$SYCL/bin/syclcc \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
       -DCMAKE_INSTALL_PREFIX=$CWD/inst \
       ..

make -j4
