# SAL 2.0 - a more effecient movable memory system

Experiments have shown that cold cache lines is the number one
cause of slowness, and movable node objects introduce a level
of indriection between every level of the tree. 

A related issue is that reference counting causes massive number
of cold cache hits during COW operations; this impacts insert
performance, but it also impacts release costs. 

An alternative to reference counting is versioned epochs,
   where for each object we have created + destoryed numbers.

Today SAL allocates 32 MB segments, each segment has a 12 byte
header which includes type + size + ptr_address + sequence, checksum, and
the fact that a value contains a pointer back to its own
ptr_address is how we detect "live vs moved" objects, and
the size field helps us with rounding down from cacheline level.


The allocator in each segment is a bump allocator which makes
it fast, easy to undo, and no overhead searching... but freed
space within a transaction is lost until compaction.

Key Insights
------------
 - the goal of the allocator is to separate stale/cold data
   from hot data so that data sets that are large than ram
   don't end up paging (or worse, writing out) cold data
   just because it is next to hot and or changing data.

 - ref counts don't need to be on the indirection control_block,
   which means they can be externalized to their own u16 array,
   but the control block still needs 1 bit of reference counting
   which is the "ref" owned by the u16 count. 

 - the 12 byte header places a massive burden on types that need
   128 byte alignment themselves.. but without this header, the
   compactor has no way to walk the data and decide what to move
   and when.

 - all state must be outside of transient fields...









