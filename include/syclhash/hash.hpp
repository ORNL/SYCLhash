#pragma once

#include <stdexcept>

#include <syclhash/base.hpp>
#include <syclhash/alloc.hpp>

namespace syclhash {

template <typename T, sycl::access::mode Mode>
class DeviceHash;

/** Hash table. Store key:value pairs, where key is Ptr type.
 *
 * @tparam T type of values held in the table.
 */
template <typename T>
class Hash {
    template <typename U, sycl::access::mode Mode>
    friend class DeviceHash;

    Alloc                     alloc;
    sycl::buffer<T,1>         cell;
    sycl::buffer<Ptr, 1>      keys; // key for each cell
    sycl::buffer<Ptr, 1>      next; // `next` pointer for each cell

  public:
    const int size_expt; ///< Base-2 log of the max hash-table size.

    /** Allocate and initialize space for max 2**size_expt items.
     *
     * @arg size_expt exponent of the allocator's size
     * @arg queue SYCL queue to use when initializing free_list
     */
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
 * @tparam Mode accessor mode for interacting with the bucket
 *
 */
template<class T, sycl::access::mode Mode>
class Bucket {
    DeviceHash<T,Mode> &dh;

  public:
    const Ptr key;
    using value_type = T;

    /** Construct the Bucket for `dh`
     * that points at `key`.
     */
    Bucket(DeviceHash<T,Mode> &dh, Ptr key)
            : dh(dh), key(key) {}

    /** A cursor pointing at a specific cell in the Bucket.
     */
    class iterator {
        friend class Bucket<T,Mode>;
        DeviceHash<T,Mode> &dh;
        Ptr index;
        sycl::group<1> grp;
    
        /** Only friends can construct iterators.
         */
        iterator(sycl::group<1> g, DeviceHash<T,Mode> &dh, Ptr key, Ptr index)
                : dh(dh), index(index), key(key), grp(g) {
            seek(g, false);
        }

        /** Seek forward to the next valid index --
         * where dh.keys[index] == key
         * or index == null_ptr
         *
         * if fwd == true, then seek will start by advancing
         * to next[index]
         */
        template <typename Group>
        void seek(Group g, bool fwd) {
            bool ret;
            apply_leader(ret, g, dh.seek(index, key, fwd));
            if( ret ) { //apply_leader<Ptr>(g, dh.seek, index, key)
                index = sycl::select_from_group(g, index, 0);
            } else {
                index = null_ptr;
            }
        }

      public:
        const Ptr key;

        using iterator_category = std::forward_iterator_tag;
        using difference_type = size_t;

        using value_type = T;
        using reference = T &;
        using pointer = T *;

        // Is this cursor over an empty cell?
        // (due to potential parallel acccess,
        //  this function would not be stable)
        //bool is_empty() {
        //    return dh.keys[index] == null_ptr;
        //}

        /// Is this cursor a null-pointer?
        bool is_null() const {
            return index == null_ptr;
        }

        /// Access the value this iterator refers to.
        reference operator *() const {
            return dh.get_cell(index);
        }

        /// Erase the key:value pair this iterator refers to.
        bool erase() {
            bool ret;
            //return apply_leader<bool>(grp, dh.erase_key, index, key);
            apply_leader(ret, grp, dh.erase_key(index, key));
            return ret;
        }

        /// pre-increment
        iterator &operator++() {
            if(index != null_ptr) {
                seek(grp, true);
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
    iterator begin(sycl::group<1> grp) {
        /* This special case is correct, but not needed.
         * Because unoccupied cells must have next set to null_ptr,
         * seek() will find next[key] == null_ptr.
        if(!dh.alloc.occupied(key)) {
            return end(grp);
        }*/
        return iterator(grp, dh, key, dh.mod(key));
    }

    /** Create an iterator pointing to the end
     *  of this Bucket.
     *
     * @arg grp collective group that will access the iterator
     */
    iterator end(sycl::group<1> grp) {
        return iterator(grp, dh, key, null_ptr);
    }

    /** Put a new value into this bucket.
     *
     *  This is implemented via a call to DeviceHash::set_key,
     *  which acquires the key (replacing it with `null_ptr-1`),
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
    iterator insert(sycl::group<1> grp, const T& value) {
        Ptr index = dh.mod(key);
        // special case -- first insertion
        //if(dh.keys[index] == null_ptr && dh.alloc.try_alloc(grp, index)) {
        if(dh.alloc.try_alloc(grp, index)) {
            bool ret;
            do {
                apply_leader(ret, grp, dh.set_key(index, key, value));
            } while(!ret);
            //while(!apply_leader<bool>(grp, dh.set_key, index, key, value)) {}
            return begin(grp);
        }
        // below: canonical index is already allocated somewhere.

        while(1) { // Loop here because link-cell-by-CAS may not succeed.
            index = dh.mod(key);

            // seek to end of linked-list
            bool empty; //= apply_leader<bool>(grp, dh.seek, index, null_ptr);
            apply_leader(empty, grp, dh.seek(index, null_ptr, false));

            if(empty) {
                // leader's index is an empty slot: re-use it
                index = sycl::select_from_group(grp, index, 0);
                bool ok;
                apply_leader(ok, grp, dh.set_key(index, key, value));
                //if( apply_leader<bool>(grp, dh.set_key, index, key, value) ) {
                if(ok) {
                    return iterator(grp, dh, key, index);
                }
            } else {
                // reached end: allocate a new slot
                Ptr loc = dh.alloc.alloc(grp, key);
                if(loc == null_ptr) { // memory is full
                    return end(grp);
                }
                if(grp.get_local_linear_id() == 0) {
                    dh.set_key_unsafe(loc, key, value);
                    // now that we have it, we must put loc at the end
                    dh.set_last(key, loc);
                }
                return iterator(grp, dh, key, loc);
            }
        };
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
    bool erase(iterator position) {
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
template <typename T, sycl::access::mode Mode>
class DeviceHash {
    //friend class Bucket<T,Mode>;

    sycl::accessor<T, 1, Mode>        cell;
    sycl::accessor<Ptr, 1, Mode>      keys;  // key for each cell
    sycl::accessor<Ptr, 1, Mode>      next; // `next` pointer for each cell
  public:
    DeviceAlloc<Mode>                alloc;
    const int size_expt;

    /** Construct from the host Hash class.
     *
     * @arg h host alloc class
     * @arg cgh SYCL handler
     */
    DeviceHash(Hash<T> &h, sycl::handler &cgh)
        : cell(h.cell, cgh)
        , keys(h.keys, cgh)
        , next(h.next, cgh)
        , alloc(h.alloc, cgh)
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

    /** Convenience function to insert `value`
     * to the :class:`Bucket` at the given `key`
     */
    template <typename Group>
    typename Bucket<T,Mode>::iterator
    insert(Group g, Ptr key, const T&value) {
        Bucket<T,Mode> bucket = (*this)[key];
        return bucket.insert(g, value);
    }

    /// Set the key (assumes no potential contention on key)
    void set_key_unsafe(Ptr loc, Ptr key, const T &value) {
        keys[loc] = key;
        cell[loc] = value;
    }

    /// Used to read a cell value.
    T &get_cell(Ptr loc) {
        return cell[loc];
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

    /// Erase the key at index (by overwriting key with null_ptr)
    bool erase_key(Ptr index, Ptr key) {
        if(index == null_ptr) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::release,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    keys[index]);
        Ptr val = key;
        return v.compare_exchange_strong(val, null_ptr);
    }


    /// Link key -> loc by setting last next-ptr from key
    void set_last(Ptr key, Ptr loc) {
        Ptr val = null_ptr;
        while(1) {
            Ptr index = seek_end(mod(key));
            auto v = sycl::atomic_ref<
                                Ptr, sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
                                        next[index]);
            if(v.compare_exchange_weak(val, loc, sycl::memory_order::release))
                break;
        }
    }

    /** Seek to the next slot with key == search
     *
     * updates index as it moves forwards
     * returns true if keys[index] == search
     * false if keys[index] != search OR if index == null_ptr
     * (in which case next[index] == null_ptr)
     *
     * .. note::
     *
     *     Different concurrent workers may not traverse
     *     the list in the same way!  Use within `apply_leader`
     *     if you need consistency.
     */
    bool seek(Ptr &index, Ptr search, bool fwd) {
        if(fwd) {
            index = next[index];
        }
        if(index == null_ptr) return false;
        while(keys[index] != search) {
            Ptr mv = next[index];
            if(mv == null_ptr) return false;
            index = mv;
        };
        return true;
    }

    /** Seek to the end of the linked-list starting at `index`.
     *
     * .. note::
     *
     *     Different concurrent workers may not traverse
     *     the list in the same way!  Use within `apply_leader`
     *     if you need consistency.
     */
    Ptr seek_end(Ptr index) {
        // seek to end of linked-list
        while(1) {
            Ptr mv = next[index];
            if(mv == null_ptr) return index;
            index = mv;
        }
        return null_ptr;
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

