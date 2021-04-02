mkdir logs
./0_bench_dgl_gin.py| tee dgl_gin.log
./1_log2csv.py dgl_gin.log