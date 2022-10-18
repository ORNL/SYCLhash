#pragma once

#include <syclhash/config.hpp>

namespace syclhash {

typedef uint32_t Ptr; ///< Pointers are 32-bit unsigned ints.
static const Ptr null_ptr = ~(Ptr)0; ///< Addressable space is 2**32-2
static const Ptr reserved = (~(Ptr)0) ^ 1; ///< mark reserved
static const Ptr erased   = (~(Ptr)0) ^ 3; ///< mark erased

inline int ctz(uint32_t x) {
    int ans = 0;
    int c = 32; // c will be the number of zero bits on the right
    x &= -int32_t(x);
    if (x & 0x0000FFFF) c -= 16;
    if (x & 0x00FF00FF) c -= 8;
    if (x & 0x0F0F0F0F) c -= 4;
    if (x & 0x33333333) c -= 2;
    if (x & 0x55555555) c -= 1;
    return c;
}

inline int ctz(const uint64_t x) {
    int ans = 0;
    if(x == 0) return 64;

    // Binary search to find first 1 set.
    for(int level=32; level>0; level=level/2) {
        // low-order "level" number of bits
        uint64_t mask = (~(uint64_t)0) >> (64-level);
        if(((x>>ans) & mask) == 0) { // not in low-order "level" bits
            ans += level;
        }
    }
    return ans;
}

/// have only the leader call fn(args), all other id-s return same result
#define apply_leader(ans, g, call) { \
    if(g.get_local_linear_id() == 0) { \
        ans = call; \
    } \
    ans = sycl::select_from_group(g, ans, 0); \
}

/* This approach doesn't work.  It confuses the compiler with:
 * error: reference to non-static member function must be called
 *
template <typename Ret, typename Group, typename F, typename... Args>
Ret apply_leader(Group g, F fn, Args... args) {
    Ret ans;
    if(g.get_local_linear_id() == 0) {
        ans = fn(args...);
    }
    return sycl::select_from_group(g, ans, 0);
}*/

/// rotate x left by r bits
inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

/// Finalization mix - force all bits of a hash block to avalanche
inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

/** updates seed, returns a value derived from the output seed
 *
 * @arg h1 seed
 * @arg k1 value to accumulate into seed
 */
inline uint32_t murmur3(uint32_t *seed, uint32_t k1) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;

    uint32_t h1 = *seed;
    h1 ^= k1;
    h1 = rotl32(h1,13);
    h1 = h1*5+0xe6546b64;

    *seed = h1;
    return fmix32(h1);
}

}
