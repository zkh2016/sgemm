all: sgemm

sgemm:sgemm.cu
	nvcc -O3 sgemm.cu -o sgemm -arch sm_61 -lcublas
