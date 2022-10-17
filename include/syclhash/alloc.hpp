#pragma once

#include <stdexcept>

#include <syclhash/base.hpp>

namespace syclhash {

template <sycl::access::mode Mode>
class DeviceAlloc;

/** Allocator. Uses a free-list to manage
 *  chunks of identically sized memory.
 *
 *  (requires 2**(size_expt-3) bytes)
 */
class Alloc {
    template <sycl::access::mode Mode>
    friend class DeviceAlloc;

    sycl::buffer<uint32_t, 1> free_list; // bitmask for free cells
    int size_expt;

  public:
    /** Allocate and initialize space for max 2**size_expt items.
     *
     * @arg size_expt - exponent of the allocator's size
     * @arg queue     - SYCL queue to use when initializing free_list
     */
    Alloc(int size_expt, sycl::queue &queue)
        : free_list(1 << (size_expt < 5 ? 0 : size_expt-5))
        , size_expt(size_expt) {

        if(size_expt >= 32) {
            throw std::invalid_argument("2**size_expt is too large.");
        }

        reset(queue);
    }

    void reset(sycl::queue &queue) {
        queue.submit([&](sycl::handler &cgh){
            sycl::accessor F{free_list, cgh, sycl::write_only, sycl::no_init};
            cgh.fill(F, (uint32_t)0);
        });

        // mark out-of range slots as occupied
        if(size_expt < 5) {
            Ptr count = 1<<size_expt; // number of addressable cells
            queue.submit([&](sycl::handler &cgh){
                sycl::accessor F{free_list, cgh, sycl::write_only, sycl::no_init};
                cgh.single_task([=]() {
                    F[0] = ~((1<<count)-1);
                });
            });
        }
    }
};

/** Device-side functionality for Alloc.
 *
 * @tparam Mode The accessor mode for reading/writing from/to the free-list.
 */
template <sycl::access::mode Mode>
class DeviceAlloc {
    sycl::accessor<uint32_t, 1, Mode> free_list; // bitmask for free cells
    int size_expt;

  public:

    /** Construct from the host Alloc class.
     *
     * @arg h host alloc class
     * @arg cgh SYCL handler
     */
    DeviceAlloc(Alloc &h, sycl::handler &cgh)
        : free_list(h.free_list, cgh)
        , size_expt(h.size_expt)
        { }


    /// Count number of used slots at this super-index.
    int count(Ptr si) const {
        // https://graphics.stanford.edu/~seander/bithacks.html
        uint32_t v = free_list[si];
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        return ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    /** Leading thread from Group g calls try_alloc.
     *
     * Every group member will return the same result:
     * - true on successful allocation
     * - false on failure
     *
     */
    template <typename Group>
    bool try_alloc(Group g, Ptr index) const {
        bool ok = false;

        if(g.get_local_linear_id() == 0) {
            ok = try_alloc(index);
        }
        return sycl::select_from_group(g, ok, 0);
    }

    /** Try allocating at the given index.
     *
     * This is implemented with an atomic-or using mask = 1<<(index%32)
     * on free_list[index/32].
     *
     * @arg index 
     * @return true on successful allocation, false otherwise
     */
    bool try_alloc(Ptr index) const {
        const Ptr i = index/32;
        const Ptr j = index%32;

        const Ptr mask = 1 << j;
        auto v = sycl::atomic_ref<
                        uint32_t, sycl::memory_order::acquire,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                                free_list[i]);
        Ptr ans = v.fetch_or(mask);
        return (ans & mask) == 0;
    }

    /// Returns true on successful free, false if already free.
    bool free(Ptr index) const {
        const Ptr i = index/32;
        const Ptr j = index%32;

        Ptr mask = 1 << j;
        auto v = sycl::atomic_ref<
                        uint32_t, sycl::memory_order::release,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
                                free_list[i]);
        Ptr ans = v.fetch_and(~mask);
        return (ans & mask) != 0;
    }

    /** The whole group will search through a range
     * of free-list values, starting at index = base
     *
     * It will attempt to allocate one of the indices
     * that are unoccupied.  Internally, every thread
     * looks at one block of 32 slots.  If one is full,
     * it attempts to allocate it.
     *
     * Every member of the group returns the same value.
     *
     * @arg g collective group doing the search
     * @arg base starting index to search (will be returned if available)
     * @return index of allocated result (or null_ptr on failure)
     */
    template <typename Group>
    Ptr search_index(Group g, Ptr base) const {
        const int tid = g.get_local_linear_id();
        const Ptr my_i = (base/32 + tid) % (1 << (size_expt < 5 ? 0 : size_expt-5));
        // Decrease probability of contention
        // by having each tid increment the offset.
        const Ptr offset = (tid+(base%32)) % 32;

        Ptr idx = null_ptr;
        Ptr occ = free_list[my_i];
        // rotate right so we start searching from offset
        Ptr sorted = rotl32(~occ, 32-offset);
        int j = ctz(sorted); // number of trailing 1-s
        if(j < 32) { // j is the first unoccupied slot in free_list[my_i],
                     // counting circularly from offset
            idx = my_i*32 + (j+offset)%32;
            if(! try_alloc(idx)) {
                idx = null_ptr;
            }
        }
        // Here idx is set to an allocated index, or null_ptr otherwise

        // Note: we could alternatively reduce on idx
        // directly.  However, this approach always
        // returns the `closest' answer to base in the search order.
        int winner = sycl::reduce_over_group(g,
                    (uint32_t)(
                        idx == null_ptr ? g.get_local_linear_range() : tid),
                    sycl::minimum<uint32_t>());
        if(winner == g.get_local_linear_range()) {
            return null_ptr;
        }
        if(idx != null_ptr && tid != winner) {
            // return the speculative allocation
            this->free(idx);
        }
        return sycl::select_from_group(g, idx, winner);
    }

    /** Wrap the index into the range of valid keys, [0, 2**size_expt).
     */
    Ptr mod(Ptr index) const {
        return index & ((1<<(size_expt+1)) - 1);
    }

    /** Increment index in a pseudo-random way
     *
     * Combining `seed` with `accum`,
     * stores the result in `seed`, then
     * returns an index (computed from the output seed)
     * within the addressable range.
     *
     * Implementation note: Consumers of this function
     * require seed to make progress even for accum == 0.
     *
     */
    Ptr next_hash(Ptr *seed, Ptr accum) const {
        return mod(murmur3(seed, accum));
    }

    /** Find and allocate the next free cell by searching randomly,
     * starting from key.
     */
    template <typename Group>
    Ptr alloc(Group g, Ptr key = 0) const {
        const Ptr gid = g.get_group_linear_id();
        Ptr index = mod(key);
        Ptr loc = null_ptr;
        for(int i=0; i < (1 << size_expt); i++) {
            loc = search_index(g, index);
            if(loc != null_ptr) {
                break;
            }
            index = next_hash(&key, gid);
            //printf("Failed attempt - searching at %x\n", index);
        }
        return loc;
    }
};

// FIXME: distribute initial key-space more evenly throughout index space
// to prevent collisions?
}
