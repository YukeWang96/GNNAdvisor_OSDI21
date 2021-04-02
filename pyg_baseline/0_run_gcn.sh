mkdir logs
mv *.csv logs/
mv *.log logs/
./0_bench_pyg_gcn.py| tee pyg_gcn.log
./1_log2csv.py pyg_gcn.log