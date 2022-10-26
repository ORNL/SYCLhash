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

The iterators over each Bucket are collective, since each group must
call add/del/next with the same argument.

.. doxygenclass:: syclhash::Hash
   :members:

.. doxygenclass:: syclhash::DeviceHash
   :members:

Because groups don't exist in host code, the `HostHash` class
has considerably less useful functionality than `DeviceHash`.

See the examples inside `tests/` for ideas.
