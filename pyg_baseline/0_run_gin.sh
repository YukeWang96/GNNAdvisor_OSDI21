mkdir logs
./0_bench_pyg_gin.py| tee pyg_gin.log
./1_log2csv.py pyg_gin.log