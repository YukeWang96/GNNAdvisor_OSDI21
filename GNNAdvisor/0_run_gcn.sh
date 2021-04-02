mkdir logs
mv *.csv logs/
mv *.log logs/
./0_bench_GNNA_GCN.py| tee GNNA_GCN.log
./1_log2csv.py GNNA_GCN.log