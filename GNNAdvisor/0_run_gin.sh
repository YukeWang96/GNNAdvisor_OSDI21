mkdir logs
./0_bench_GNNA_GIN.py| tee GNNA_GIN.log
./1_log2csv.py GNNA_GIN.log