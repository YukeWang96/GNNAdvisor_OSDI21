for grh in cora citeseer pubmed
do
for k in 32 64 128
do
build/bin/spmm market ./../../data/misc/${grh}.mtx --num-runs=200 --feature-len=$k &>> gr_test.txt
done
done


OVCAR-8H
./spmm market /home/yuke/.graphs/orig-mtx/OVCAR-8H.mtx --num-runs=200 --feature-len=16
# ./spmm market /home/yuke/.graphs/orig-mtx/amazon0505.mtx --num-runs=200 --feature-len=16