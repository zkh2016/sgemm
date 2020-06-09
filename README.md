# sgemm
The implementation method refer to the [maxas](https://github.com/NervanaSystems/maxas/wiki/SGEMM).

# performance
1. The test environment： ubuntu18.04, cuda10, 1080ti
2. The code only supports limited input matrix, not universal adaptation, only for learning. Here is the GFLOP for testing different size matrices

 N | cublas |  sgemm  | sgemm/cublas |
-|-|-
512 | 4451.6069 | 3587.3280 |  80% |
1024 | 7856.5241 | 6640.6945 | 84% |
2048 | 9409.4447 | 8769.9500 | 93% |
4096 | 10180.4288 | 9708.4873 | 95% |
