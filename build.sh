#!/bin/bash

rm -fr build
mkdir build

cd build
PRE=/ccs/proj/stf006/rogersdd/crusher

#export CXXFLAGS="-I${MPICH_DIR}/include"
#export LDFLAGS="-L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa"

# need release build type becaue -g makes the compile super slow
cmake -DCMAKE_PREFIX_PATH=$PRE \
      -DCMAKE_CXX_COMPILER=$PRE/bin/syclcc \
      -DHIPSYCL_TARGETS=hip:gfx90a \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_DOCS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PRE \
      ..

make -j4

