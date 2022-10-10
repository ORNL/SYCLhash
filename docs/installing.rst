Installing syclhash
###################

syclhash is build using `CMake <https://cmake.org>`_.
It should be installable using the commands below::

    mkdir build && cd build
    cmake ..
    make && make install

Compiling requires a working SYCL language compiler.
This package has been written and tested using
`hipSYCL <https://github.com/illuhad/hipSYCL>`_.

See ``build.sh`` for an example of how to structure your
build using hipSYCL compiled with `spack <https://github.com/spack/spack>`_.

