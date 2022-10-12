#pragma once

#include <stdexcept>

#include <syclhash/base.hpp>
#include <syclhash/alloc.hpp>

namespace syclhash {

template <typename T, sycl::access::mode Mode>
class DeviceHash;

template <typename T>
class Hash {
    Alloc                     alloc;
    sycl::buffer<T,1>         cell;
    sycl::buffer<Ptr, 1>      keys; // key for each cell
    sycl::buffer<Ptr, 1>      next; // `next` pointer for each cell

  public:
    const int size_expt;

    Hash(int size_expt, sycl::queue &queue)
        : alloc(size_expt, queue)
        , cell(1 << size_expt)
        , keys(1 << size_expt)
        , next(1 << size_expt)
        , size_expt(size_expt) {

        if(size_expt >= 32) {
            throw std::invalid_argument("2**size_expt is too large.");
        }

        queue.submit([&](sycl::handler &cgh){
            sycl::accessor K{keys, cgh, sycl::write_only, sycl::no_init};
            cgh.fill(K, null_ptr);
        });
        queue.submit([&](sycl::handler &cgh){
            sycl::accessor N{next, cgh, sycl::write_only, sycl::no_init};
            cgh.fill(N, null_ptr);
        });
    }

    template <typename U, sycl::access::mode Mode>
    friend class DeviceHash;
};

/** Iterate through a linked list of cells.
 *
 * Example:
 *
 *     for(T &val : dh.get(key)) {
 *         printf("%lu %lu\n", key, val);
 *     } printf("\n"); }
 *
 */
template<class T, sycl::access::mode Mode>
class Bucket {
    DeviceHash<T,Mode> &dh;

  public:
    const Ptr key;

    Bucket(DeviceHash<T,Mode> &dh, Ptr key)
            : dh(dh), key(key) {}

    /** iterator = a cursor pointing at a specific cell in a linked list.
     */
    class iterator {
        friend class Bucket<T,Mode>;
        DeviceHash<T,Mode> &dh;
        Ptr index;
    
        /** Only friends can construct iterators.
         */
        iterator(DeviceHash<T,Mode> &dh, Ptr key, Ptr index)
                : dh(dh), index(index), key(key) {
            seek();
        }

        /** Seek forward to the next valid index --
         * where dh.keys[index] == key
         * or index == null_ptr
         */
        void seek() {
            for(; index != null_ptr && dh.keys[index] != key
                ; index = dh.next[index] );
        }

      public:
        const Ptr key;

        using iterator_category = std::forward_iterator_tag;
        using difference_type = size_t;

        using value_type = T;
        using reference = T &;
        using pointer = T *;

        //< Is this cursor over an empty cell?
        // (due to potential parallel acccess,
        //  this function would not be stable)
        //bool is_empty() {
        //    return dh.keys[index] == null_ptr;
        //}
        //< Is this cursor a null-pointer?
        bool is_null() const {
            return index == null_ptr;
        }

        reference operator *() const {
            return dh.cell[index];
        }

        // pre-increment
        iterator &operator++() {
            if(index != null_ptr) {
                index = dh.next[index];
                seek();
            }
            return *this;
        }
        // post-increment
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

    iterator begin() {
        /* This special case is correct, but not needed.
         * Because unoccupied cells must have next set to null_ptr,
         * seek() will find next[key] == null_ptr.
        if(!dh.alloc.occupied(key)) {
            return end();
        }*/
        return iterator(dh, key, dh.mod(key));
    }
    iterator end()   { return iterator(dh, key, null_ptr); }

    /** Put a new value into this bucket.
     *
     *  Return the Cursor where value was successfully placed.
     *  Note - cursor will be == end(), a null_ptr, on error.
     */
    template <typename Group>
    iterator insert(Group g, const T& value) {
        Ptr index = dh.mod(key);
        // special case -- first insertion
        if(dh.keys[index] == null_ptr && dh.alloc.try_alloc(g, index)) {
            while(!dh.set_key(g, index, key, value)) {}
            return begin();
        }
        // below: canonical index is already allocated somewhere.

        while(1) { // Loop here because link-cell-by-CAS may not succeed.
            index = dh.mod(key);

            // seek to end of linked-list
            for(; dh.keys[index] != null_ptr && dh.next[index] != null_ptr
                ; index = dh.next[index]) { }

            if(dh.keys[index] == null_ptr) {
                // empty slot: re-use it
                if(dh.set_key(g, index, key, value)) {
                    return iterator(dh, key, index);
                }
            } else if(dh.next[index] == null_ptr) {
                // reached end: allocate a new slot
                Ptr loc = dh.alloc.alloc(g, key);
                if(loc == null_ptr) { // memory is full
                    return end();
                }
                dh.keys[loc] = key;
                dh.cell[loc] = value;
                // now that we have it, we must put loc at the end
                while( !dh.set_next(g, index, loc)) {
                    index = dh.mod(key);
                    for(; dh.next[index] != null_ptr
                        ; index = dh.next[index]) { }
                }
                return iterator(dh, key, loc);
            }
        };
    }

    /** Return true if index was deleted, false if it was not present.
     *
     * Caution: Regardless of the output of this call,
     *          your cursor is now invalid.
     *
     * Note: internally, we set the `key` to `null_ptr`
     * for deleted keys.
     * These are skipped over during bucket iteration.
     */
    bool erase(iterator position) {
        if(position.index == null_ptr) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::release,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    dh.keys[position.index]);
        Ptr val = key;
        return v.compare_exchange_strong(val, null_ptr);
    }

    //> Have the group leader call erase() and report back to the group.
    template <typename Group>
    bool erase(Group g, iterator position) {
        bool result = false;
        if(g.get_local_linear_id() == 0) {
            result = del(position);
        }
        return sycl::select_from_group(g, result, 0);
    }
};

/** Device side data structure for hash table.
 *
 * Max capacity = 1<<size_expt key/value pairs.
 *
 * Requires size_expt < 32
 */
template <typename T, sycl::access::mode Mode>
class DeviceHash {
    friend class Bucket<T,Mode>;

    DeviceAlloc<Mode>                alloc;
    sycl::accessor<T, 1, Mode>        cell;
    sycl::accessor<Ptr, 1, Mode>      keys;  // key for each cell
    sycl::accessor<Ptr, 1, Mode>      next; // `next` pointer for each cell
  public:
    const int size_expt;

    DeviceHash(Hash<T> &h, sycl::handler &cgh)
        : alloc(h.alloc, cgh)
        , cell(h.cell, cgh)
        , keys(h.keys, cgh)
        , next(h.next, cgh)
        , size_expt(h.size_expt)
        //, count(h.cell.get_count()) // max capacity
        { }

    /** Wrap the index into the range of valid keys, [0, 2**size_expt).
     */
    Ptr mod(Ptr index) const {
        return index & ((1<<(size_expt+1)) - 1);
    }

    // bucket = (key % N) is the first index.
    Bucket<T,Mode> operator[](Ptr key) {
        return Bucket<T,Mode>(*this, key);
    }

    /** Convenience function to insert a key,value pair.
     */
    template <typename Group>
    typename Bucket<T,Mode>::iterator
    insert(Group g, Ptr key, const T&value) {
        Bucket<T,Mode> bucket = (*this)[key];
        return bucket.insert(g, value);
    }

    /** Where key was null_ptr, set to key
     *  and fill its value atomically.
     */
    bool set_key(Ptr index, Ptr key, const T &value) {
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = null_ptr;
        if(!v.compare_exchange_weak(val, null_ptr-1,
                            sycl::memory_order::acquire)) {
            return false;
        }
        cell[index] = value;
        val = null_ptr-1;
        while(!v.compare_exchange_strong(val, key,
                            sycl::memory_order::release)) {
            // this is an error
        };
        return true;
    }

    //> Have the group leader call set_key and report the result.
    template <typename Group>
    bool set_key(Group g, Ptr index, Ptr key, const T &value) {
        bool result = false;
        if(g.get_local_linear_id() == 0) {
            result = set_key(index, key, value);
        }
        return sycl::select_from_group(g, result, 0);
    }

    //> Link index -> loc by setting next[index]
    bool set_next(Ptr index, Ptr loc) {
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    next[index]);
        Ptr val = null_ptr;
        return(v.compare_exchange_weak(val, loc,
                            sycl::memory_order::release));
    }

    //> Have the group leader call set_next and report the result.
    template <typename Group>
    bool set_next(Group g, Ptr index, Ptr loc) {
        bool result = false;
        if(g.get_local_linear_id() == 0) {
            result = set_next(index, loc);
        }
        return sycl::select_from_group(g, result, 0);
    }
};

}

    /** Increment index in a pseudo-random way
     * that causes different groups to diverge in their
     * search sequence.
     *
     * Always returns an index within the addressable range.
     *
     * Implementation Note: must make progress even for gid == 0.
    Ptr next_hash(Ptr &seed, Ptr id) {
        return mod(murmur3(seed, id));
    }
     */

    /** Low-level function to allocate an empty key/value pair.
     *
     * TODO: add an initialization function for cell[index] here.
    template <typename Group>
    Ptr alloc(Group g, Ptr key) {
        Ptr index = mod(key);
        const Ptr gid = g.get_group_linear_id();
        const int tid = g.get_local_linear_id();
        const int gsize = g.get_local_linear_range();
        const Ptr id = gid*gsize + tid;

        bool did_set = false;
        Ptr winner = gsize; // max potential tid value == no winner

        if(tid > 0) { // tid 0 tries index, all others diffuse randomly
            index = next_hash(index, id);
        }
        for(int trial=0; trial<10; trial++) {
            auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    dh.keys[index]);
            val = v.compare_exchange_weak(null_ptr, key);
            if(val == null_ptr) { // success
                did_set = true;
                winner = tid;
            }
            winner = sycl::reduce_over_group(g, winner, sycl::min<uint32_t>());
            if(winner != gsize) break;

            index = next_hash(index, id);
        }
        // No memory available.
        if(winner == gsize) {
            return null_ptr;
        }

        // pull the index from the winning tid
        winner = sycl::select_from_group(g, index, winner);
        if(did_set && winner != tid) { // need to put back
            keys[index] = null_ptr;
        }

        // Note: We need a strong memory ordering on `next`
        // key so that next[winner] is null when this
        // cell is first referenced.
        // At present, we assume next[] is set correctly
        // by Hash::Hash and during deletion.
        return winner;
    }
    */

