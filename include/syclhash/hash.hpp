#pragma once

#include <stdexcept>

#include <syclhash/base.hpp>
#include <syclhash/alloc.hpp>

namespace syclhash {

template <typename T, sycl::access::mode Mode>
struct DeviceHash;

template <typename T>
class Hash {
    Alloc                     alloc;
    sycl::buffer<T,1>         cell;
    sycl::buffer<Ptr, 1>      keys; // key for each cell
    sycl::buffer<Ptr, 1>      next; // `next` pointer for each cell

  public:
    const int size_expt;

    Hash(int size_expt, sycl::queue &queue)
        : alloc(1 << size_expt, queue)
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

    template <sycl::access::mode Mode>
    friend class DeviceHash;
};

/** Cursor pointing at a specific cell in a linked
 *  list.
 */
template<class T, sycl::access::mode Mode>
class Cursor {
    DeviceHash<T,Mode> &dh;
    Ptr index;

  public:
    const Ptr key;

    using iterator_category = std::forward_iterator_tag;
    using difference_type = size_t;

    using value_type = T;
    using reference = T &;
    using pointer = T *;

    Cursor(DeviceHash<T,Mode> &dh, Ptr key, Ptr index)
            : dh(dh), index(index), key(key) {
        seek();
    }

    //< Is this cursor over an empty cell?
    bool is_empty() {
        return dh.keys[index] == null_ptr;
    }
    //< Is this cursor a null-pointer?
    bool is_null() {
        return index == null_ptr;
    }

    /** Seek forward to the next valid index --
     * where dh.keys[index] == key
     * or index == null_ptr
     */
    void seek() {
        for(; index != null_ptr && dh.keys[index] != key
            ; index = dh.next[index] );
    }
    reference operator *() {
        return dh.cell[index];
    }
    //pointer operator ->() {
    //    return dh.cell[index];
    //}
    // pre-increment
    Cursor& operator ++() {
        index = dh.next[index];
        seek();
        return *this;
    }
    // post-increment
    Cursor operator++(int) {
        Cursor tmp = *this;
        ++(*this);
        return tmp;
    }
    friend bool operator== (const Cursor& a, const Cursor& b) {
        return a.index == b.index;
    };
    friend bool operator!= (const Cursor& a, const Cursor& b) {
        return a.index != b.index;
    };

    /** Append the given `next` index to the end of this bucket.
     */
    Cursor<T,Mode> append(Ptr next) {
        // Special case: initial key is always a bucket member.
        if(dh.mod(key) == next) return Cursor<T,Mode>(dh, key, next);

        Ptr val = next;
        //dh.next[next] = null_ptr;

        while(val != null_ptr) {
            Ptr cur = index; // Start at index.
            // Seek to the end of the bucket.
            // TODO (optimization): Collapse pointer chain here.
            for(; dh.next[cur] != null_ptr
                ; cur = dh.next[cur]);
        
            auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    dh.next[cur]);
            val = null_ptr;
            if(v.compare_exchange_weak(val, next)) {
                break;
            }
        }
            
        return Cursor<T,Mode>(dh, key, next);
    }

    /** Return true if index was deleted, false if it was not present.
     *
     * side-effect: Invalidates the cursor.
     *
     * Note: internally, we set the `key` to `null_ptr`
     * for deleted keys.
     * These are skipped over during bucket iteration.
     */
    bool del() {
        if(index == null_ptr) return false;
        auto v = sycl::atomic_ref<
                            Ptr, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                                    dh.keys[index]);
        Ptr val = key;
        bool ret = v.compare_exchange_strong(val, null_ptr);

        // Whatever the result, this cursor is now invalid.
        index = null_ptr;
        return ret;
    }
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
    Cursor<T,Mode> begin() { return Cursor<T,Mode>(dh, key, dh.mod(key)); }
    Cursor<T,Mode> end()   { return Cursor<T,Mode>(dh, key, null_ptr); }

    /** Put a new value into this bucket.
     *
     *  Return the Cursor where value was successfully placed.
     *  Note - cursor will be == end(), a null_ptr, on error.
     */
    template <typename Group>
    Cursor<T,Mode> put(Group g) {
        // 1. allocate a free hash-index
        Ptr next = 0; //dh.alloc(g, key);
        // 2. link the next value onto the bucket
        return begin().append(next);
    }

};

/** Device side data structure for hash table.
 *
 * Max capacity = 1<<size_expt key/value pairs.
 *
 * Requires size_expt < 32
 */
template <typename T, sycl::access::mode Mode>
struct DeviceHash {
    DeviceAlloc<Mode>                alloc;
    sycl::accessor<T, 1, Mode>        cell;
    sycl::accessor<Ptr, 1, Mode>      keys;  // key for each cell
    sycl::accessor<Ptr, 1, Mode>      next; // `next` pointer for each cell
  //private:
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
    Ptr mod(Ptr index) {
        return index & ((1<<(size_expt+1)) - 1);
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

    // bucket = (key % N) is the first index.
    Bucket<T,Mode> get(Ptr key) {
        return Bucket<T,Mode>(*this, key);
    }

    /** Convenience function to add one instance of `key`.
     */
    template <typename Group>
    Cursor<T,Mode> put(Group g, Ptr key) {
        Bucket<T,Mode> bucket = get(key);
        return bucket.put(key);
    }
};

}
