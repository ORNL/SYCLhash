Running on Crusher / Frontier
-----------------------------

The following environment is needed to compile / run on crusher and frontier::

    module load cmake
    module load craype-accel-amd-gfx90a

    export LD_PRELOAD=/opt/cray/pe/lib64/libsci_cray_mp.so.5
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-5.1.0/llvm/lib:/ccs/proj/stf006/rogersdd/crusher/lib

- /ccs/proj/stf006/rogersdd/crusher/lib provides the boost dependency.
- LD_PRELOAD is needed to avoid "[hipSYCL Warning] /opt/cray/pe/lib64/libsci_cray_mp.so.5: cannot allocate memory in static TLS block"

