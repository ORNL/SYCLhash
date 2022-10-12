#include <syclhash/hash.hpp>

struct Cell {
    int data[10];
};


using namespace syclhash;

int main() {
    sycl::queue q;
    // 1024 cells
    syclhash::Hash<Cell> hash(10, q);

    // Submit a kernel allocating lots of cells:
    q.submit([&](sycl::handler &cgh) {
        //sycl::accessor X{ans, cgh, sycl::write_only, sycl::no_init};
        DeviceHash<Cell,sycl::access::mode::discard_write> dh(hash, cgh);

        cgh.parallel_for(sycl::nd_range<1>(32,32), [=](sycl::nd_item<1> it) {
            uint32_t gid = it.get_group_linear_id();
            uint32_t tid = it.get_local_linear_id();

            //uint32_t loc = dh.get(it.get_group(), gid);

            if(tid == 0) {
            //    X[gid] = loc;
            }
        });
    });

    return 0;
}
