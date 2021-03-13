Rabbit Order
============

- This is an implementation of the algorithm proposed in the following paper:
    - J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, and S. Iwamura.
      Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis.
      IEEE International Parallel and Distributed Processing Symposium (IPDPS),
      2016.
- Please read `license.txt` before reading or using the files.
- Note that some graph datasets are already reordered, and so Rabbit Order will
  not show significant performance improvement on those graphs.
    - For example, [Laboratory for Web Algorithmics](http://law.di.unimi.it/)
      reorders graphs using the Layered Label Propagation algorithm.
    - Web graphs are sometimes reordered by URL. This ordering is known to show
      good locality.


How to use
----------

`demo/reorder.cc` is a sample program that reorders graphs by using Rabbit
Order.
Type `make` in the `demo/` directory to build the program.

### Requirements

- g++ (4.9.2)
- Boost C++ library (1.58.0)
- libnuma (2.0.9)
- libtcmalloc\_minimal in google-perftools (2.1)

Numbers in each parenthesis are the oldest versions that we used to test this
program.

### Usage

    Usage: reorder [-c] GRAPH_FILE
      -c    Print community IDs instead of a new ordering

If flag `-c` is given, this program runs in the *clustering mode*
(described later); otherwise, it runs in the *reordering mode*.

### Input

`GRAPH_FILE` has to be an edge-list file like the following:

    14 10
    2 194
    14 1
    89 20
    ...

Each line represents an edge.
The first number is a source vertex ID, and the second number is a target
vertex ID.
Edges do not need to be sorted in some orderings, but **vertex IDs should be
zero-based contiguous numbers (i.e., 0, 1, 2, ...)**; otherwise, this demo
program may consume more memory and show lower performance.

Many edge-list files in this format are found in
[Stanford Large Network Dataset Collection] (http://snap.stanford.edu/data/index.html).

### Output (reordering mode)

The program prints a new ordering as follows:

    8
    16
    1
    4
    ...

These lines represent the following permutation:

    Vertex 0 ==> Vertex 8
    Vertex 1 ==> Vertex 16
    Vertex 2 ==> Vertex 1
    Vertex 3 ==> Vertex 4
    ...

### Output (clustering mode)

The program prints a clustering result as follows:

    1
    5
    1
    5
    5
    9
    ...

These lines represent the following classification:

    Vertex 0 ==> Cluster 1
    Vertex 1 ==> Cluster 5
    Vertex 2 ==> Cluster 1
    Vertex 3 ==> Cluster 5
    Vertex 4 ==> Cluster 5
    Vertex 5 ==> Cluster 9
    ...

Note that the cluster IDs may be non-contiguous.

