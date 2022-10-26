Memory Allocator
################

The memory allocator is based on the concepts from
`SlabHash <https://github.com/owensgroup/SlabHash>`_.
An entire group works synchronously, so every operation
is called by every index in the group, but only one logical
allocation/de-allocation takes place during a call.

The allocator is only designed to allocate/de-allocate
one data address at a time.  This is because its intended
use is for hash-tables, where adding a key to a bucket
requires one new slot.

It maintains a *free-list* of `uint32_t` bitmasks.  Each bit corresponds to
a cell.  If a bit is on, then the cell is occupied.

Memory addresses (type `uin32_t n`), index cells,
and are decoded as you might expect::

    // Pseudo-code (ignoring race-conditions).
    // Try allocation at n.

    bool occupied = (free_list[n/32]>>(n%32)) & 1; // bit n of free_list

    if( ! occupied ) {
        free_list[n/32] |= 1<<(n%32);
        return cell[n];
    } else {
        return "Failed Allocation at n";
    }

Each call to allocate() is a group collective - where the
group works together to allocate the same block.

The first unallocated cell is determined
by scanning ``free_list`` -- reading one group's-worth
of ``uint32_t-s`` at a time.
As memory fills up, the search sequence utilizes
a pseudo-random search order based on the 
``nd_item.get_group_linear_id()``::

    N = pow(2, size_expt);               // number of addressable cells
    n_0     = group_linear_id;           // first index linearly
    n_{i+1} = murmur3(n_i, n_0);         // move based on group id.

For practicality, the maximum addressable space is limited to
`2^31` cells (so, please choose `size_expt <= 31`).

Allocate just returns an un-allocated address.  If you want it to actually
reference something, you'll need to declare your actual data space somewhere
on your own.

.. doxygenclass:: syclhash::Alloc
   :members:

.. doxygenclass:: syclhash::DeviceAlloc
   :members:
