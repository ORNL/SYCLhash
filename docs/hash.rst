Hash Table
##########

The hash table is very similar to `syclhash::Alloc`.
It provides three additional operations,

  * add(grp, k, data): add a data point to the cell index `k`

  * del(grp, k, ptr): delete the data point from index `k`

  * next(grp, k, ptr): iterate over the data points inside `k`

These are collective calls, so each group must call add/del/next
with the same argument.

.. doxygenclass:: syclhash::Hash
   :members:
