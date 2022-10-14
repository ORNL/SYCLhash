Hash Table
##########

This hash table works similarly to `syclhash::Alloc`.
Although it provides lookup, and `insert` functions directly,

.. doxygenfunction:: syclhash::DeviceHash::operator[]

.. doxygenfunction:: syclhash::DeviceHash::insert

most operations on the hash-table are accomplished using
`Bucket`-s,

.. doxygenclass:: syclhash::Bucket
   :members:

These are collective calls, so each group must call add/del/next
with the same argument.

.. doxygenclass:: syclhash::Hash
   :members:

.. doxygenclass:: syclhash::DeviceHash
   :members:
