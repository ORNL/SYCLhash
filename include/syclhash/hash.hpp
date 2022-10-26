#pragma once

#include <stdexcept>

#include <syclhash/base.hpp>
//#include <syclhash/alloc.hpp>

namespace syclhash {

template <typename, int, sycl::access::mode, sycl::access::target>
class DeviceHash;

/** Hash table. Store key:value pairs, where key is Ptr type.
 *
 * @tparam T type of values held in the table.
 * @tparam search_width 2**width = number of consecutive keys to scan linearly
 */
template <typename T, int search_width=4>
class Hash {
    template <typename, int, sycl::access::mode, sycl::access::target>
    friend class DeviceHash;

    static_assert(search_width < 6, "Width too large!");

    //Alloc                     alloc;
    sycl::buffer<T,1>         cell;
    sycl::buffer<Ptr,1>       keys; // key for each cell
    int size_expt;            ///< Base-2 log of the max hash-table size.

    /// Reset just keys and next pointers (sufficient for new allocator).
    void reset_k(sycl::queue &queue) {
        queue.submit([&](sycl::handler &cgh){
            sycl::accessor K{keys, cgh, sycl::write_only, sycl::no_init};
            cgh.fill(K, null_ptr);
        });
    }

  public:

    static const int width = search_width;

    /** Allocate and initialize space for max 2**size_expt items.
     *
     * @arg size_expt exponent of the allocator's size
     * @arg queue SYCL queue to use when initializing free_list
     */
    Hash(int size_expt, sycl::queue &queue)
        : //alloc(size_expt, queue),
          cell(1 << size_expt)
        , keys(1 << size_expt)
        , size_expt(size_expt) {

        if(size_expt >= 32) {
            throw std::invalid_argument("2**size_expt is too large.");
        }

        reset_k(queue);
    }

    /** Reset this structure to empty.
     */
    void reset(sycl::queue &queue) {
        //alloc.reset(queue);
        reset_k(queue);
    }
};

enum class Step {
    Stop = 7,
    Continue,
    Complete,
};

/** A Bucket points to the set of values in the hash
 * table with the given key.
 *
 * The real work is done with iterators, created
 * through the bucket's begin(g) and end(g) calls.
 *
 * Example:
 *
 *     auto bucket = dh[key];
 *     for(const T &val = bucket.begin(g); val != bucket.end(g); ++val) {
 *         printf("%lu %lu\n", key, val);
 *     } printf("\n");
 *
 * @tparam T type of values held in the Bucket
 * @tparam search_width number of bins to search linearly between random jumps
 * @tparam Mode accessor mode for interacting with the bucket
 * @tparam accessTarget where memory accessors will live
 *
 */
template<class T,
         int search_width,
         sycl::access::mode Mode,
         sycl::access::target accessTarget
                 = sycl::access::target::global_buffer>
class Bucket {
    const DeviceHash<T,search_width,Mode,accessTarget> &dh;

  public:
    const Ptr key;
    using value_type = T;
    using DeviceHashT = DeviceHash<T,search_width,Mode,accessTarget>;
    static constexpr int width = search_width;
    static constexpr sycl::access::target Target = accessTarget;

    /** Construct the Bucket for `dh`
     * that points at `key`.
     */
    Bucket(const DeviceHashT &dh, Ptr key) : dh(dh), key(key) {}

    /** A cursor pointing at a specific cell in the Bucket.
     */
    template <typename Group>
    class iterator {
        friend class Bucket<T,width,Mode,accessTarget>;

        const DeviceHashT *dh;
        static constexpr int width = search_width;
        Ptr key;
        Ptr index;
        Group grp;
    
        /** Only friends can construct iterators.
         */
        iterator(Group g, const DeviceHashT *dh, Ptr key, Ptr index)
                : dh(dh), key(key), index(index), grp(g) {
            seek(false);
        }

        /** Seek forward to the next valid index --
         * where dh->keys[index] == key
         * or index == null_ptr
         *
         * if fwd == true, then seek will start by advancing
         * to next(index)
         */
        void seek(bool fwd) {
            if(index == null_ptr) return;
            const Ptr i0 = index;
            if( ! dh->run_op(grp, key, index, true, [=](Ptr i1, Ptr k1){
#               ifdef DEBUG_SYCLHASH
                printf("seeking for %u from %u (found %u at %u)\n", key, i0, k1, i1);
#               endif
                if(k1 == null_ptr)
                    return Step::Stop;
                if(fwd && i1 == i0) return Step::Continue;
                if(key == k1)
                    return Step::Complete;
                return Step::Continue;
            }) ) {
                index = null_ptr;
            }
        }

      public:
        using DeviceHashT = DeviceHash<T,search_width,Mode,accessTarget>;

        using iterator_category = std::forward_iterator_tag;
        using difference_type = size_t;

        using value_type = std::conditional_t<Mode == sycl::access::mode::read,
                       const T, T >;
        using reference = std::conditional_t<Mode == sycl::access::mode::read,
                       const T&, T& >;
        using pointer = std::conditional_t<Mode == sycl::access::mode::read,
                       const T*, T* >;

        iterator(const iterator &x)
            : dh(x.dh), key(x.key), index(x.index), grp(x.grp) {}
        iterator &operator=(const iterator &x) {
            dh = x.dh;
            key = x.key;
            index = x.index;
            return *this;
        }

        // Is this cursor over an empty cell?
        // (due to potential parallel acccess,
        //  this function would not be stable)
        //bool is_empty() {
        //    return dh->keys[index] == null_ptr;
        //}

        /// Is this cursor a null-pointer?
        bool is_null() const {
            return index == null_ptr;
        }

        /// Access the value this iterator refers to.
        reference operator *() const {
            return dh->get_cell(index);
        }

        /// Erase the key:value pair this iterator refers to.
        bool erase() {
            bool ret;
            //return apply_leader<bool>(grp, dh->erase_key, index, key);
            apply_leader(ret, grp, dh->erase_key(index, key));
            return ret;
        }

        /// pre-increment
        iterator &operator++() {
            if(index != null_ptr) {
                seek(true);
            }
            return *this;
        }
        /// post-increment
        iterator operator++(int) {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }
        bool operator== (const iterator& b) const {
            return index == b.index;
        };
        bool operator!= (const iterator& b) const {
            return index != b.index;
        };
    };

    /** Create an iterator pointing to the first cell
     *  contained in this Bucket.
     *
     * @arg grp collective group that will access the iterator
     */
    template <typename Group>
    iterator<Group> begin(Group grp) const {
        return iterator<Group>(grp, &dh, key, dh.mod(key));
    }

    /** Create an iterator pointing to the end
     *  of this Bucket.
     *
     * @arg grp collective group that will access the iterator
     */
    template <typename Group>
    iterator<Group> end(Group grp) const {
        return iterator<Group>(grp, &dh, key, null_ptr);
    }

    /** Put a new value into this bucket.
     *
     *  This is implemented via a call to DeviceHash::set_key,
     *  which acquires the key (replacing it with `reserved`),
     *  sets the value, then releases the key (replacing it with `key`).
     *
     *  @return iterator where value was successfully placed,
     *          or end(grp) on failure.
     *
     *  .. note::
     *
     *      The bucket is unordered, so there are no guarantees
     *      as to what relative location the value will be inserted.
     *
     *  .. note::
     *
     *      If value read/write and key deletion are happening concurrently
     *      by different threads, there is a chance that previous
     *      threads accessing the cell may still be using it
     *      (if they are unaware that someone had deleted it).
     */
    template <typename Group, typename ...Args>
    iterator<Group> insert(Group grp, Args ... args) const {
        Ptr index = dh.insert_cell(grp, key, false, args...);
        return iterator<Group>(grp, &dh, key, index);
    }

    template <typename Group, typename ...Args>
    iterator<Group> insert_unique(Group grp, Args ... args) const {
        Ptr index = dh.insert_cell(grp, key, true, args...);
        return iterator<Group>(grp, &dh, key, index);
    }

    /** Return true if index was deleted, false if it was not present.
     *
     *  Internally, we set the `key` to `null_ptr`
     *  for deleted keys. These are skipped over
     *  during bucket iteration.
     *
     *  This happens via a call to DeviceHash::erase_key, which
     *  has release memory-ordering semantics.
     *
     *  Regardless of the output of this call,
     *  your cursor is now invalid.
     *
     * .. warning::
     * 
     *     If value accesses, inserting and erasing are all happening
     *     concurrently by different threads, it is up to you to stop other
     *     readers & writers of the cell's value from accessing it.
     *     You must ensure this before you delete it!
     *     Those other potential accessors include everyone with
     *     an iterator pointing at this same position (since
     *     the iterator can be dereferenced).
     *
     *     Otherwise, you run the risk that the cell may be allocated
     *     again (potentially with a different key), and written
     *     into.
     *
     *     Technically, erasing without any concurrent insertions
     *     would leave the value dedicated, but no longer referenced.
     */
    template <typename Group>
    bool erase(iterator<Group> position) const {
        return position.erase();
    }
};

/** Device side data structure for hash table.
 *
 * Max capacity = 1<<size_expt key/value pairs.
 *
 * Requires size_expt < 32
 *
 * @tparam T type of values held in the table.
 * @tparam Mode access mode for DeviceHash
 */
template <typename T,
          int search_width,
          sycl::access::mode Mode,
          sycl::access::target accessTarget
              = sycl::access::target::global_buffer>
class DeviceHash {
    template <typename,int, sycl::access::mode, sycl::access::target>
    friend class DeviceHash;

    sycl::accessor<Ptr, 1, Mode, accessTarget> keys;  // key for each cell
    sycl::accessor<T, 1, Mode, accessTarget>   cell;

    //const DeviceAlloc<Mode,accessTarget> alloc;

    /*  Attempt to reserve the key at index (by overwriting `was`
     *  with reserved using relaxed semantics)
     *
     *  Sets `was` to the old value of the key on return
     *  @return true if successful, false if no change
     */
    bool reserve_key(Ptr index, Ptr &was, Ptr key) const {
        ADDRESS_CHECK(index, size_expt);
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        if(!v.compare_exchange_weak(was, reserved)) {
                            //sycl::memory_order::acquire))
            return false;
        }
        return true;
    }

  public:
    using BucketT = Bucket<T,search_width,Mode,accessTarget>;
    static constexpr sycl::access::target Target = accessTarget;
    static constexpr int width = search_width;
    const int size_expt;

    /** Construct from the host Hash class.
     *
     * @arg h host hash class
     * @arg cgh SYCL handler
     */
    DeviceHash(Hash<T,search_width> &h, sycl::handler &cgh)
        : keys(h.keys, cgh)
        , cell(h.cell, cgh)
        //, alloc(h.alloc, cgh)
        , size_expt(h.size_expt)
        //, count(h.cell.get_count()) // max capacity
        { }

    template <typename Device>
    DeviceHash(Hash<T,search_width> &h, sycl::handler &cgh, const Device &)
        : DeviceHash(h, cgh) { }

    template <bool use=true, std::enable_if_t< use &&
              accessTarget == sycl::access::target::host_buffer,bool> = true>
    DeviceHash(Hash<T,search_width> &h)
        : keys(h.keys)
        , cell(h.cell)
        , size_expt(h.size_expt)
        { }

    /** Increment index in a pseudo-random way
     *
     * Implementation note: Consumers of this function
     * require it to make progress even for accum == 0.
     *
     * This is a simple congruential generator with
     * the property that it is full-period for any
     * power of 2 modulus.
     */
    uint32_t next_hash(uint32_t seed) const {
        const uint32_t a = 1664525;
        const uint32_t c = 1; //1013904223;
        seed = (uint32_t) ( (uint64_t)(a)*(uint64_t)mod_w(seed>>search_width) + c );
        return mod_w(seed) << search_width;
    }

    /** Apply the function to every key,value pair.
     *  Each key is processed by a whole group.
     *
     *  The function should have type:
     *
     *  void fn(sycl::nd_item<Dim> it, Ptr key, T &value, Args ... args);
     *
     *  A generic nd_range<1>(1024, 32) is recommended for looping
     *  over its key-space.
     */
    template <int Dim, typename Fn, typename ...Args>
    void parallel_for(sycl::handler &cgh,
                      sycl::nd_range<Dim> rng,
                      Fn fn, Args ... args) const {
        const size_t count = 1 << size_expt;

        cgh.parallel_for(rng, [=, keys=this->keys, cell=this->cell]
                (sycl::nd_item<Dim> it) {
            sycl::group<Dim> g = it.get_group();
            for(size_t i = g.get_group_linear_id()
               ;       i < count
               ;       i += g.get_group_linear_range()) {
                //Ptr key = sycl::select_from_group(g, keys[i], 0);
                Ptr key = keys[i];
                if((key>>31) & 1) continue;
                fn(it, key, cell[i], args ...);
            }
        });
    }

    /** Apply the function to every key,value pair.
     *  Each key is processed by a whole group.
     *
     *  The function should have type:
     *
     *  R fn(sycl::nd_item<Dim> it, Ptr key, T &value);
     *
     *  A generic nd_range<1>(1024, 32) is recommended for looping
     *  over its key-space.
     */
    template <typename R, int Dim, typename Fn, typename ...Args>
    void parallel_for(sycl::handler &cgh,
                      sycl::nd_range<Dim> rng,
                      sycl::buffer<R,1> &ret,
                      Fn fn, Args ... args) const {
        sycl::accessor<R, 1>  d_ret(ret, cgh, sycl::read_write);
        const size_t count = 1 << size_expt;

        cgh.parallel_for(rng,
                sycl::reduction(d_ret, sycl::plus<R>()),
                [=, keys=this->keys, cell=this->cell]
                (sycl::nd_item<Dim> it, auto &ans) {
            sycl::group<Dim> g = it.get_group();
            for(size_t i = g.get_group_linear_id()
               ;       i < count
               ;       i += g.get_group_linear_range()) {
                //Ptr key = sycl::select_from_group(g, keys[i], 0);
                Ptr key = keys[i];
                if((key>>31) & 1) continue;
                R tmp = fn(it, key, cell[i]); //, args ...);
                ans += tmp;
            }
        });
    }

    template <typename U, int Dim, sycl::access::mode Mode2, typename Fn,
              typename ...Args>
    void map(sycl::handler &cgh,
             sycl::nd_range<Dim> rng,
             const DeviceHash<U, search_width, Mode2, accessTarget> &out,
             Fn fn, Args ... args) const {
        sycl::accessor<T, 1, Mode>    cell(this->cell);
        sycl::accessor<U, 1, Mode2>   cell2(out.cell);
        sycl::accessor<Ptr, 1, Mode>  keys(this->keys);
        sycl::accessor<Ptr, 1, Mode2> keys2(out.keys);
        const size_t count = 1 << size_expt;

        cgh.parallel_for(rng,
                [=](sycl::nd_item<Dim> it) {
            sycl::group<Dim> g = it.get_group();
            const int ngrp = g.get_group_linear_range();
            for(size_t i = g.get_group_linear_id()
               ;       i < count
               ;       i += ngrp) {
                Ptr key = keys[i];
                keys2[i] = key;
                if((key>>31) & 1) continue;
                fn(it, key, cell[i], cell2[i], args ...);
            }
        });
    }

    /** Wrap the index into the valid range, [0, 2**size_expt).
     */
    Ptr mod(Ptr index) const {
        return index & ((1<<size_expt) - 1);
    }

    /** Wrap the super-index into the valid range, [0, 2**(size_expt-width)).
     */
    Ptr mod_w(Ptr si) const {
        return si & ((1<<(size_expt-search_width)) - 1);
    }

    /// bucket = (key % N) is the first index.
    BucketT operator[](Ptr key) const {
        return BucketT(*this, key);
    }

    /** Convenience function to insert `value`
     * to the :class:`Bucket` at the given `key`
     *
     * @arg uniq true if keys are uniq (ignores value
     *                if key is found)
     */
    template <typename Group>
    typename BucketT::template iterator<Group>
    insert(Group g, Ptr key, const T&value, bool uniq) const {
        BucketT bucket = (*this)[key];
        return bucket.insert(g, value, uniq);
    }

    /** Low-level function inserting a key,value pair
     *  and returning the Ptr index.
     *
     *  @arg args value to set on insert -- either value or empty
     *  @returns Ptr index where insertion took place / key was found
     *           or null_ptr if table is full
     */
    template <typename Group, typename ...Args>
    Ptr insert_cell(Group g, Ptr key, bool uniq, Args ... args) const {
        Ptr index = mod(key);
        if(! run_op(g, key, index, uniq, [=](Ptr i1, Ptr k1) {
#           ifdef DEBUG_SYCLHASH
            printf("inserting %u at %u (found %u)\n", key, i1, k1);
#           endif
            // Note: we can't over-write erased keys
            // in uniq mode (since insert_uniq k2, delete k2, insert_uniq k1)
            // when k2 collides with k1 would leave an erased slot in front.
            while(k1 == null_ptr || (!uniq && k1 == erased)) {
                if(set_key(i1, k1, key, args...)) {
                    return Step::Complete;
                }
#               ifdef DEBUG_SYCLHASH
                printf("failed set_key (found %u)\n", k1);
#               endif
            }
            if(uniq && k1 == key) {
                return Step::Complete;
            }
            return Step::Continue;
        })) {
            return null_ptr;
        }
        return index;
    }

    /** Run through the `canonical` sequence of keys,
     * searching for null_ptr, deleted, or
     * (if uniq == true) key.
     *
     * If found, runs `Step op(Ptr index, Ptr key)`
     * on the index,keys[index] pair.
     *
     * That returns false of `Stop`, true on `Complete`, or,
     * on `Continue`, continues searching until all keys are exhausted,
     * then return false.
     */
    template <typename Group, typename Fn>
    bool run_op(Group g,
               Ptr key,
               Ptr &index,
               bool uniq,
               Fn op) const {
        Ptr i0 = (index >> search_width) << search_width; // align reads

        uint32_t cap = (1 << (size_expt-search_width)) + (i0 != index);

        // It would be better if we could fix the group size
        // to (1<<width).  However, this setup mimicks
        // that by disabling threads past (1<<width).
        const int tid = g.get_local_linear_id();
        const int ntid = g.get_local_linear_range();
        // Max number of usable threads in this group.
        const int ngrp = ntid < (1<<search_width)
                       ? ntid : (1<<search_width);
        const int max_sz = //ngrp * ceil( (1<<width)/ngrp)
                           ngrp*( ((1<<search_width)+ngrp-1)/ngrp );
        for(int trials = 0
           ; trials < cap
           ; ++trials, i0 = next_hash(i0)) {
            // read next `2**width` keys
            for(int i = tid; i < max_sz; i += ngrp) {
                bool check = false;
                Ptr ahead;
                //if(i0+i < (1<<size_expt))
                if(i < (1<<search_width)) {
                    ahead = keys[i0+i];
                    check = ahead == null_ptr
                          || ahead == erased
                          || (uniq && ahead == key);
                    // ensure we move forward of index on trial 0
                    if(trials == 0) check = check && i0+i >= index;
                }
                warpMaskT mask = ballot(g, check);

                for(int winner=0; (mask>>winner) > 0; winner++) {
                    if(((mask>>winner)&1) == 0) continue;
                    Ptr found = sycl::select_from_group(g, ahead, winner);

                    int idx = i0+i-tid+winner;
                    Step step;
                    apply_leader(step, g, op(idx, found));
                    switch(step) {
                    case Step::Stop:
                        index = idx;
                        return false;
                    case Step::Complete:
                        index = idx;
                        return true;
                    case Step::Continue:
                        break;
                    }
                }
            }
        }
        return false;
    }

    /// Low-level function used to read a cell value using a Ptr index.
    std::conditional_t<Mode == sycl::access::mode::read,
                       const T&, T& >
    get_cell(Ptr loc) const {
        ADDRESS_CHECK(loc, size_expt);
        return cell[loc];
    }

    /** Where key was null_ptr, set to key
     *  and fill its value atomically.
     *
     * @arg index hash-table index where key is set
     * @return true if set is successful, false otherwise
     */
    bool set_key(Ptr index, Ptr &was, Ptr key, const T &value) const {
        ADDRESS_CHECK(index, size_expt);
        if(!reserve_key(index, was, key)) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = reserved;
        cell[index] = value;
        while(!v.compare_exchange_strong(val, key,
                            sycl::memory_order::release)) {
            // this is an error
        };
        return true;
    }

    bool set_key(Ptr index, Ptr &was, Ptr key) const {
        ADDRESS_CHECK(index, size_expt);
        if(!reserve_key(index, was, key)) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = reserved;
        while(!v.compare_exchange_strong(val, key)) {
            // this is an error
        };
        return true;
    }

    /*  Erase the key at index (by overwriting key with `erased`)
     *
     *  @return true if successful, false if no change
     */
    bool erase_key(Ptr index, Ptr key) const {
        if(index == null_ptr) return false;
        ADDRESS_CHECK(index, size_expt);
#       ifdef DEBUG_SYCLHASH
        printf("Erasing %u at %u\n", key, index);
#       endif
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::release,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = key;
        return v.compare_exchange_strong(val, erased);
    }
};

template<typename T, int width, class Descriptor>
DeviceHash(Hash<T,width>&,
           sycl::handler&,
           Descriptor)
    -> DeviceHash<T, width, Descriptor::mode, Descriptor::target>;

template <typename T, int search_width, sycl::access::mode Mode>
class HostHash : public DeviceHash<T, search_width, Mode,
                                   sycl::access::target::host_buffer>
{
  public:
    template <typename ...Args>
    HostHash(Hash<T,search_width> &hash, Args... deduction_helpers)
        : DeviceHash<T,search_width,Mode,sycl::access::target::host_buffer>(hash)
        {}
};

template<typename T, int width, class Descriptor>
HostHash(Hash<T,width>&, Descriptor)
    -> HostHash<T, width, Descriptor::mode>;
}
