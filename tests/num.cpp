#include <syclhash/hash.hpp>

using namespace syclhash;

template <typename T>
int show(sycl::nd_item<1> it, uint32_t id, const T &x) {
    if(it.get_local_id(0) != 0) return 0;
    return 1;
}

unsigned rng_incr(unsigned seed) {
  constexpr uint64_t a = 1103515245;
  constexpr uint64_t c = 12345;
  return (unsigned)( ((uint64_t)seed*a + c) & 0x7FFFFFFF );
}

int test1(sycl::queue &q, const int width) {
    const int groups = 4;
    const int inserts_per_group = 10;
    typedef int T;
    // 64 cells
    syclhash::Hash<T> hash(6, q); // must be large enough for all inserts!

    q.submit([&](sycl::handler &cgh) {
        DeviceHash dh(hash, cgh, sycl::read_write);

        cgh.parallel_for(sycl::nd_range<1>(groups*width, width),
                         [=](sycl::nd_item<1> it) {
            unsigned rng = it.get_group(0);
            sycl::group<1> g = it.get_group();

            for(int i=0; i<inserts_per_group; i++) {
                rng = rng_incr(rng);
                dh[rng].insert(g, (int)rng);
            }
        });
    });
    
    // Submit a second kernel showing all key:value pairs
    int N = 0;
    if(1) {
        sycl::buffer<int,1> ret(&N,1);
        q.submit([&](sycl::handler &cgh) {
            DeviceHash dh(hash, cgh, sycl::read_only);
            sycl::nd_range<1> rng(groups*width, width);
            dh.parallel_for(cgh, rng, ret, //show<int>);
                [](sycl::nd_item<1> it, uint32_t id, const int &x) {
                  if(it.get_local_id(0) != 0) return 0;
                  return 1;
            });
        });
    }
    printf("%d occupied cells.\n", N);

    return N != groups*inserts_per_group;
}

int main() {
    int err = 0;
    sycl::queue q;
    //err += test1(q, 1);
    err += test1(q, 2);
    err += test1(q, 4);
    err += test1(q, 8);
    err += test1(q, 16);
    err += test1(q, 32);
    return err;
}
