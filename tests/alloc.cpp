#include <syclhash/alloc.hpp>

struct Cell {
    int data[10];
};


using namespace syclhash;

int main() {
    sycl::queue q;
    // 1024 cells
    syclhash::Alloc alloc(10, q);

    // Submit a kernel allocating lots of cells:
    q.submit([&](sycl::handler &cgh) {
        //sycl::accessor X{ans, cgh, sycl::write_only, sycl::no_init};
        DeviceAlloc<sycl::access::mode::discard_write> da(alloc, cgh);

        cgh.parallel_for(sycl::nd_range<1>(32,4), [da=da](sycl::nd_item<1> it) {
            uint32_t gid = it.get_group_linear_id();
            uint32_t tid = it.get_local_linear_id();

            for(int i=0; i<10; i++) {
                Ptr idx = da.alloc(it.get_group(), i);
                if(it.get_local_id(0) == 0)
                    printf("%lu,%lu ~> %lu\n", it.get_group(0), i, idx);
            }
        });
    });

    return 0;
}
