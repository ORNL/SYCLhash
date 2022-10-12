#include <string>

#include <syclhash/hash.hpp>

using namespace syclhash;

template <typename T>
void show(const sycl::stream &out, int id, T &x) {
    std::stringstream ss;
    ss << id << ":";
    for(auto item : x) {
        ss << " " << item;
    }
    ss << "\n";
    std::string ans(ss.str());
    out << ans.c_str();
}

int main() {
    typedef int T;
    sycl::queue q;
    // 1024 cells
    syclhash::Hash<T> hash(10, q);

    // Submit a kernel filling and emptying hashes by group
    q.submit([&](sycl::handler &cgh) {
        //sycl::accessor X{ans, cgh, sycl::write_only, sycl::no_init};
        //DeviceHash<T,sycl::access::mode::discard_write> dh(hash, cgh);
        DeviceHash<T,sycl::access::mode::read_write> dh1(hash, cgh);
        sycl::stream out(1024, 256, cgh);

        cgh.parallel_for(sycl::nd_range<1>(16,4), [=](sycl::nd_item<1> it) {
            DeviceHash<T,sycl::access::mode::read_write> dh( dh1 );
            int gid = it.get_group(0);
            sycl::group<1> g = it.get_group();

            {
                auto bucket = dh[gid];
                bucket.insert(g, 1+10*gid);
                bucket.insert(g, 2+10*gid);
                bucket.insert(g, 3+10*gid);
                if(it.get_local_id(0) == 0) {
                    show(out, gid, bucket);
                }
            }

            // get the next index over
            if(1) {
                auto bucket = dh[ (gid+1)%4 ];
                //bucket.erase(++bucket.begin());
                if(it.get_local_id(0) == 0) {
                    show(out, gid, bucket);
                }

                //bucket.erase(bucket.begin());
                //if(it.get_local_id(0) == 0) {
                //    show(out, gid, bucket);
                //}
            }
        });
    });

    return 0;
}
