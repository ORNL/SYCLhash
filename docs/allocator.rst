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

It maintains 2 data structures:

  * A giant vector of `Cell` data.

  * A *free-list* of `uint32_t` bitmasks.  Each bit corresponds to
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

    N = sizeof(cell / sizeof(Cell));     // allocatable cells
    n_0     = group_linear_id;           // first index linearly
    n_{i+1} = searchNext(n_i,n_0,N);

    where searchNext(n_i,n_0,N) = hash(n_i | (n_0<<16)) % N

The hash output space is 2^16.  This limits the maximum addressable space to
N <= 2^16 cells.

.. doxygenclass:: syclhash::Alloc
   :members:

.. doxygenclass:: syclhash::DeviceAlloc
   :members:
