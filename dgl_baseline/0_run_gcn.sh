mkdir logs
mv *.csv logs/
mv *.log logs/
./0_bench_dgl_gcn.py| tee dgl_gcn.log
./1_log2csv.py dgl_gcn.log