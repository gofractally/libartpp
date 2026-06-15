# Alt Node Design

The biggest win at scale for ART is to minimize the number of cachelines one
must load. The next is to utilize direct indexing and branchless designs. .
What this ammounts to is making the "pointer" the header that points to the 
next level of the tree. 

For a given node, direct indexing is fastest for query, but
if the node is too sparse then linear scans for lower bound / iteration
slow down and you are wasting memory.

Path compression is critical for reducing hops and therefore we should prefer
to use the extra space in a cacheline for a prefix rather than expand out to
a prefix node.


P

## Set List Node
 - 1 byte + sizeof(pointer) per branch, optional(pointer) for terminal here.
 - 8 byte pointers give a max branching factor of ~13 
 - any unused space is available for a common prefix 
 - some pointers get tagged as "inline" which means a single cacheline
   may hold 2-5 small setlist nodes, they can be made real nodes if
   they get too big.  
 
## Full Subrange 
 - 16 8 byte pointers per cacheline  

This means that if all 13 setlist nodes fall within the same subrange, 
     then the same state could be represented as a Full Subrange.



## Dense Subranges 
     - when max-min > 16 you have 2+ cachelines... with densly packed pointers

## Sparse Subranges
     - You have > 13 branches, but they are spread out across the entire
       range... currently subdive it into two 128 bit ranges and use
       a bitset to identify presense... but with 8 byte pointers, 
       it might as well be two setlist nodes rather than a bitset..
       the 128 bit bitset introduces branches, and small setlists can be
       SIMD or ILP to non branching direct calc. Plus it adds another
       code path.  
     - A full node is 16 cachelines, thus, by pidgeon hole principle, 
       random distirutions would tend to put less than 2 per cacheline
       in full mode.  Thus growing from setlist to full is wasteful,
       but going to a C2 node... has its own problems if all 15 branches
       all fall in the same half of the tree...  C2 really needs to 
       be two setlist nodes with a btree style division of the space,
       such that when a setlist splits both cachelines end up half
       full.  And a headerless pointer would need to know:
          1. that we have a setlist destination
          2. how many clines there are
          3. what is the dividing byte 

        Lets call this a S2 node... the pointer needs type + divider
        The next is an S3 node which requires 2 dividers + type,
           and can handle up to 42 branches 
        Now if you use 4 byte offsets then you can get up to
        5 implicit btree nodes each with ~14 branches.. going to 70..

Now by the time you have 70 children, pidgeon hole principle says
that a full node would be about 25% full if evenly distributed; however,
density increases dramatically if they are concentrated, for example,
        with text the tree is effectively 7bit thus 128 range, and
        only 8 of 16 of the full node's branches will contain something.

Thus... with 8 byte pointers, you use 16 bits for the bitset presense,
    the tag tells you it is a sparse_fullnode..

Node Types:
   1. setlist - 16 bits represent 16 dividers.. a 1 means there is
                a divider there; thus popcount + 1 = number of cachelines 
   2. sparse_full - 16 bits is present vector of full ranges
   3. full        - identical to 0xffff sparse_full.. 

Now we have the issue of path-compression and "small setlists" and terminal values.

A setlist node with dividers can easily place the terminal value in 
the left most cacheline and setlist header byte listing numbranches can use
a bit to indicate if the first branch is terminal value..

One additional type of node that we can have is 
   4. prefix_ - has a common prefix, optional terminal value, followed by
      one of the other node types all in one allocation. 


A setlist node:
    u16 prefixlen:6 
    u16 has_term:1
    u16 nbranch:4 // max 14
    u16 inline_children:5 // presense vector of branches 
    u8  present[nbranch]
    u64 branches[nbranch+has_term] // branchN = branches[N+has_term]
    u8  unused space, 
    u8  prefix[prefixlen];  // aligned to end, grows toward start
   // total <= 128 bytes

The case above is simple for fixed memory structures,
    but requires some tweeks for ref counted shared memory.

SAL Control Blocks
 - null means ref count 0.

Paralell 16 bit ref counter array:
  - most refcount traffic does not need the 6 bytes 
       - this allows us to get 4x the density on the
         cacheline sharing in nodes...
  - however shared vs unique needs count + offset...
       - thus the control block must maintain a state bit
       that indicates if it is shared or unique, but the
       ref counts can be elsewhere, and going from 1 to 2
       forces you to +1 the shared bit, and 2 to 1, to
       -1 the shared bit.. may have to CAS 


Better Garbage Collection
 - reference counting is expensive because of eager updates,
 every copy, clone, requires updating many counts. The alternative
 is garbage collection... and that requires either a full traversal or
 epic based free.

I am going to propose the following:

1. memory is orgnaized into allocation blocks...
2. versions (commits) are number that monotoncically increment,
     and each version tracks the N allocation blocks that it
     created new objects in. 
3. Each block has a bitset that tracks which cachelines are free/used
4. It also has a parallel array of {created,destroyed} versions per
   cacheline.  
5. there exists an array of pinned versions... each version has
its own reference count, and when it goes to 0.. we go to the
blocks that it wrote into and see what, if anything, can be freed.

When a node is allocated, it gets stamped with the created time
     when a node is replaced by a newer version, it gets stamped
     with a destroyed time. 

A node/cline is valid if there exists a pinned version >= its
created time and < its destroyed time. 

Now a garbage collector/background thread scanns through this
block-by-block... every time a version is released it
scanns those blocks, but it ultaimtely has to scan all blocks
that have objects created after the version that just
retired. 

historically we put the born/destroyed in the node; however,
that makes scanning expensive. The reality is we can find 
free space on block at a time... 

Now if we were to do ref counting, it would be on the
block basis. Every alloc in the block would inc++,
      every clone of any object on the block would inc++..
      every destroy would decrement the blocks count..

Same number of inc/dec, but fewer cachelines... but this accounting
is likely overkill when lazy GC poked by version expiration,
   and memory pressure can scan the dense version data.


















   




















