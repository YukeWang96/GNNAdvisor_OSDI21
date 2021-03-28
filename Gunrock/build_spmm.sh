rm spmm
rm -rf gunrock/
# git submodule add git@github.com:gunrock/gunrock.git
git submodule init
git submodule update
cp -r app/spmm gunrock/gunrock/app/
cp -r examples/spmm gunrock/examples
cp CMakeLists.txt gunrock/examples

cd gunrock
mkdir build && cd build
cmake .. && make spmm -j8

cd bin
cp spmm ../../../