SYCL hash
=========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installing

   allocator
   hash

SYCL hash is a lock-free concurrent hash
table written in the SYCL programming model.

Buckets are implemented using a linked-cell
list data structure. Cells are allocated
using an allocator that maintains a free-list
as a bit vector (1 = occupied).

---

:ref:`genindex`

