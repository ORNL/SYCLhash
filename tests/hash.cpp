#include <syclhash/hash.hpp>

using namespace syclhash;

// write a number to the buffer
int write_num(int num, int width, char *buf, int loc) {
    loc += width;
    for(int i=1; i<=width; ++i) {
        if(num == 0) {
            buf[loc-i] = ' ';
        } else {
            buf[loc-i] = '0' + (char)(num%10);
            num /= 10;
        }
    }
    if(num == 0 && width > 0) {
        buf[loc-1] = '0';
    }
    return loc;
}

template <typename T>
void show(sycl::group<1> g, const sycl::stream &out, int id, T &x) {
    char buf[128];
    int loc = 0;

    loc = write_num(id, 12, buf, loc);
    buf[loc++] = ':';
    for(auto item = x.begin(g); item != x.end(g); ++item) {
        if(loc+1+12+2 >= 128) break;
        buf[loc++] = ' ';
        loc = write_num(*item, 12, buf, loc);
    }
    buf[loc++] = '\n';
    buf[loc++] = '\0';
    if(g.get_local_linear_id() == 0)
        out << buf;
}

template <typename T>
void show_fn(sycl::nd_item<1> it, Ptr key,
             const T &val) {//, const sycl::stream out) {
    sycl::group<1> g = it.get_group();
    if(g.get_local_id(0) != 0) return;
    char buf[32];
    int loc = 0;
    loc = write_num(key, 12, buf, loc);
    buf[loc++] = ' ';
    buf[loc++] = ':';
    buf[loc++] = ' ';
    loc = write_num(val, 12, buf, loc);
    buf[loc++] = '\n';
    buf[loc++] = '\0';
    //out << buf;
    printf("%d : %d\n", key, val);
}

int test1(sycl::queue &q) {
    typedef int T;
    // 64 cells
    syclhash::Hash<T> hash(6, q);

    // Submit a kernel filling and emptying hashes by group
    //
    // Note: this raises a hipSYCL warning for accessing the
    // uninitialized cell array with read_write mode.  However,
    // it's not desirable to initialize the cell array, and in this
    // kernel we do both reading and writing on some cells.
    //
    q.submit([&](sycl::handler &cgh) {
        DeviceHash dh(hash, cgh, sycl::read_write);
        sycl::stream out(1024, 256, cgh);

        cgh.parallel_for(sycl::nd_range<1>(16,4), [=](sycl::nd_item<1> it) {
            int gid = it.get_group(0);
            sycl::group<1> g = it.get_group();

            {
                auto bucket = dh[gid];
                ++bucket.begin(g);
                bucket.insert(g, 1+10*gid);
                bucket.insert(g, 2+10*gid);
                bucket.insert(g, 3+10*gid);
                show(g, out, gid, bucket);
            }

            // get the next index over
            if(1) {
                auto bucket = dh[ (gid+1)%4 ];
                //(++bucket.begin(g)).erase();
                bucket.begin(g).erase();
                show(g, out, gid, bucket);
            }

            /*
            if(1) {
                auto bucket = dh[ gid ];
                bucket.insert(g, 4+10*gid);
                bucket.insert(g, 5+10*gid);
                show(g, out, gid, bucket);
            }*/
        });
    });
    q.wait();
/** Memory access error for unknown reason on device:
 *
    // Submit a second kernel showing all key:value pairs
    q.submit([&](sycl::handler &cgh) {
        //DeviceHash dh(hash, cgh, sycl::read_write);
        DeviceHash dh(hash, cgh, sycl::read_only);
        //sycl::stream out(1024, 256, cgh);
        dh.parallel_for(cgh, sycl::nd_range<1>(1024, 32), show_fn<int>);//, out);
    });*/

    return 0;
}

int test2(sycl::queue &q) {
    typedef int T;
    // 64 cells
    syclhash::Hash<T> hash(6, q);

    // Submit a kernel filling and emptying hashes by group
    q.submit([&](sycl::handler &cgh) {
        DeviceHash dh(hash, cgh, sycl::read_write);
        sycl::stream out(1024, 256, cgh);

        cgh.parallel_for(sycl::nd_range<1>(16,4), [=](sycl::nd_item<1> it) {
            int gid = it.get_group(0);
            sycl::group<1> g = it.get_group();

            {
                auto bucket = dh[gid];
                ++bucket.begin(g);
                bucket.insert_unique(g, 1+10*gid);
                bucket.insert_unique(g, 2+10*gid);
                bucket.insert_unique(g, 3+10*gid);
                show(g, out, gid, bucket);
            }

            // get the next index over
            if(1) {
                auto bucket = dh[ (gid+1)%4 ];
                //(++bucket.begin(g)).erase();
                bucket.begin(g).erase();
                show(g, out, gid, bucket);
            }

            /*
            if(1) {
                auto bucket = dh[ gid ];
                bucket.insert(g, 4+10*gid);
                bucket.insert(g, 5+10*gid);
                show(g, out, gid, bucket);
            }*/
        });
    });
    
    // Submit a second kernel showing all key:value pairs
    q.submit([&](sycl::handler &cgh) {
        DeviceHash dh(hash, cgh, sycl::read_write);
        //sycl::stream out(1024, 256, cgh);
        dh.parallel_for(cgh, sycl::nd_range<1>(1024, 32), show_fn<int>);//, out);
    });

    return 0;
}

int test3(sycl::queue &q) {
    typedef int T;
    // 64 cells
    syclhash::Hash<T> hash(6, q);

    // try some host accesses
    HostHash dh(hash, sycl::read_write);

    Ptr was = null_ptr;
    dh.set_key(10, was, 10, 100);
    T x = dh.get_cell(10);
    printf("Elem: %d\n", x);
    return x != 100;
}

int main() {
    int err = 0;
    sycl::queue q;
    err += test1(q);
    err += test2(q);
    err += test3(q);
    return err;
}
