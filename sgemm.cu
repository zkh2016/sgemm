#include <cuda.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
using namespace std;

const int ITER = 10;

inline __device__ void calc_sum(float4 a, float4 b, float* sum){
  sum[0] += a.x * b.x;
  sum[1] += a.x * b.y;
  sum[2] += a.x * b.z;
  sum[3] += a.x * b.w;

  sum[4] += a.y * b.x;
  sum[5] += a.y * b.y;
  sum[6] += a.y * b.z;
  sum[7] += a.y * b.w;

  sum[8] += a.z * b.x;
  sum[9] += a.z * b.y;
  sum[10] += a.z * b.z;
  sum[11] += a.z * b.w;

  sum[12] += a.w * b.x;
  sum[13] += a.w * b.y;
  sum[14] += a.w * b.z;
  sum[15] += a.w * b.w;
}

texture<float4, 1, cudaReadModeElementType> texRefA;
texture<float4, 1, cudaReadModeElementType> texRefB;
__global__ void kernel_sgemm(float *C, const int M, const int K, const int N){
  __shared__ float4 cache[512];
  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int blk = tid >= 32 ? bx : by;
  int ldx = tid >= 32 ? N/4 : M/4;
  texture<float4, 1, cudaReadModeElementType> tex = tid >= 32 ? texRefB : texRefA;
  int tid2 = (tid >> 4) & 1; //split warp to two sub warp
  int tid15 = tid & 15;

  int track0 = blk * 16 + tid15 + (ldx * tid2);
  int track2 = track0 + ldx*2;
  int track4 = track0 + ldx*4;
  int track6 = track0 + ldx*6;

  int end = track0 + (K-8)*ldx;

  int writeS = tid15 + tid2*16;
  writeS += tid>=32 ? 128 : 0;

  int readAs = ((tid >> 1) & 7);
  int readBs = (((tid & 0x30) >> 3) | (tid & 1)) + 128;
  float sum[64] = {0};
  float4 j0Ax00, j0Ax32, j0By00, j0By32, j1Ax00, j1Ax32, j1By00, j1By32;

  while(track0 <= end){
    float4 loadX0 = tex1Dfetch(tex, track0);
    float4 loadX2 = tex1Dfetch(tex, track2);
    float4 loadX4 = tex1Dfetch(tex, track4);
    float4 loadX6 = tex1Dfetch(tex, track6);

    cache[writeS] = loadX0;
    cache[writeS + 2*16] = loadX2;
    cache[writeS + 4*16] = loadX4;
    cache[writeS + 6*16] = loadX6;

    __syncthreads();

    track0 += 8*ldx;
    track2 += 8*ldx;
    track4 += 8*ldx;
    track6 += 8*ldx;
    writeS ^= 16*16;

#pragma unroll
    for(int j = 0; j < 8; j++){
      int prefetch = (j+1)%8;
      if(j&1){
        j0Ax00 = cache[readAs + prefetch*16];
        j0By00 = cache[readBs + prefetch*16];
        j0Ax32 = cache[readAs + prefetch*16+8];
        j0By32 = cache[readBs + prefetch*16+8];
        calc_sum(j0Ax00, j0By00, sum);
        calc_sum(j0Ax00, j0By32, &sum[16]);
        calc_sum(j0Ax32, j0By00, &sum[32]);
        calc_sum(j0Ax32, j0By32, &sum[48]);
      }
      else{
        j1Ax00 = cache[readAs + prefetch*16];
        j1By00 = cache[readBs + prefetch*16];
        j1Ax32 = cache[readAs + prefetch*16+8];
        j1By32 = cache[readBs + prefetch*16+8];
        calc_sum(j1Ax00, j1By00, sum);
        calc_sum(j1Ax00, j1By32, &sum[16]);
        calc_sum(j1Ax32, j1By00, &sum[32]);
        calc_sum(j1Ax32, j1By32, &sum[48]);
      }
    }
    readAs ^= 16*16;
    readBs ^= 16*16;
  }

  readAs = ((tid >> 1) & 7);
  readBs = (((tid &0x30) >> 3) | (tid & 1));

  for(int i = 0; i < 4; i++){
    for(int j= 0; j < 4; j++){
      C[(by*64 + readAs * 4 + i)*N + bx*64 + readBs * 4 + j] = sum[i*4+j];
    }
  }
  for(int i = 0; i < 4; i++){
    for(int j= 0; j < 4; j++){
      C[(by*64 + readAs * 4 + i)*N + bx*64 + readBs * 4 + j + 32] = sum[16+i*4+j];
    }
  }
  for(int i = 0; i < 4; i++){
    for(int j= 0; j < 4; j++){
      C[(by*64 + readAs * 4 + i + 32)*N + bx*64 + readBs * 4 + j] = sum[32+i*4+j];
    }
  }
  for(int i = 0; i < 4; i++){
    for(int j= 0; j < 4; j++){
      C[(by*64 + readAs * 4 + i + 32)*N + bx*64 + readBs * 4 + j + 32] = sum[48+i*4+j];
    }
  }
}

void sgemm(float *A, float *B, float *C, const int M, const int K, const int N){
  float* tmpA = new float[M*K];
  for(int i = 0; i < M; i++){
    for(int j = 0; j < K; j++){
      tmpA[j * M + i] = A[i*K+j];
    }
  }

  float *dev_A, *dev_B, *dev_C;
  cudaMalloc((void**)&dev_A, sizeof(float)*M*K);
  cudaMalloc((void**)&dev_B, sizeof(float)*N*K);
  cudaMalloc((void**)&dev_C, sizeof(float)*N*M);

  cudaMemcpy(dev_A, tmpA, sizeof(float)*M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, sizeof(float)*N*K, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);
  cudaBindTexture(0, texRefA, (float4*)dev_A);
  cudaBindTexture(0, texRefB, (float4*)dev_B);

  for(int i = 0; i < ITER; i++)
    kernel_sgemm<<<dim3(N/64, M/64), dim3(64,1,1)>>>(dev_C, M, K, N);

  cudaEventRecord(end, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(end);
  float time_elapsed = 0;
  cudaEventElapsedTime(&time_elapsed, start, end);
  double ops = 1.0e-9f * M*N*K*2;
  double flops = ops / (time_elapsed / 1000.0f/ITER); 
  printf("time = %.4f, gflops=%.4f\n", time_elapsed, flops);
  cudaMemcpy(C, dev_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
}

void cublas_sgemm(float *a, float *b, float *c, int M, int K, int N){
 cublasHandle_t handle;
 cublasCreate(&handle);
 float *da, *db;
 float *dc;
 cudaMalloc((void**)&da, M*K*sizeof(float));
 cudaMalloc((void**)&db, K*N*sizeof(float));
 cudaMalloc((void**)&dc, M*N*sizeof(float));
 cudaMemcpy(da, a, M*K*sizeof(float), cudaMemcpyHostToDevice);
 cudaMemcpy(db, b, N*K*sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

 float alpha = 1, beta = 0;
 for(int i = 0; i < ITER; i++)
 cublasGemmEx(handle,
     CUBLAS_OP_N,
     CUBLAS_OP_N,
     N, M, K,
     &alpha, db, CUDA_R_32F, N,
     da, CUDA_R_32F, K, &beta,
     dc, CUDA_R_32F, N,
     CUDA_R_32F,
     CUBLAS_GEMM_DFALT);
 //cudaThreadSynchronize();
  cudaEventRecord(end, 0);
  cudaEventSynchronize(start);
  cudaEventSynchronize(end);
  float time_elapsed = 0;
  cudaEventElapsedTime(&time_elapsed, start, end);
  double ops = 1.0e-9f * M*N*K*2;
  double flops = ops / (time_elapsed / 1000.0f/ITER); 
  printf("time = %.4f, gflops=%.4f\n", time_elapsed, flops);
 cudaMemcpy(c, dc, M*N*sizeof(float), cudaMemcpyDeviceToHost);
 cudaFree(da);
 cudaFree(db);
 cudaFree(dc);
}

void cpu(float *A, float *B, float *C, const int M, const int K, const int N){
  memset(C, 0.0f, sizeof(float)*M*N);
  for(int i = 0; i < M; i++){
    for(int k = 0; k < K; k++){
      float a = A[i*K + k];
      for(int j = 0; j < N; j++){
        C[i*N+j] += a * B[k*N + j]; 
      }
    }
  }
}

void verify(float *c1, float *c2, const int M, const int N){
  for(int i = 0; i < M*N; i++){
    if(fabs(c1[i] - c2[i]) > 0.000001){
      printf("%d %f %f\n", i, c1[i], c2[i]);
      return;
    }
  }
  printf("verify success\n");
}

int main(int argc, char**argv){
  int M = atoi(argv[1]);//512;
  int K = atoi(argv[2]);//512;
  int N = atoi(argv[3]);//512;
  float *A = new float[M*K];
  float *B = new float[K*N];
  float *C = new float[M*N];

  srand((unsigned)time(0));
  for(int i = 0; i< M * K; i++){
    A[i] = (float)(rand() % 100);
  }
  for(int i = 0; i < K*N; i++){
    B[i] = (float)(rand() % 100);
  }

  cpu(A, B, C, M, K, N);

  printf("cublas:\n");
  float *blas_C = new float[M*N];
  cublas_sgemm(A, B, blas_C, M, K, N);
  verify(C, blas_C, M, N);
  delete blas_C;

  printf("sgemm:\n");
  float *sgemm_C = new float[M*N];
  sgemm(A, B, sgemm_C, M, K, N);
  verify(C, sgemm_C, M, N);
  delete sgemm_C;

  delete A;
  delete B;
  delete C;
  return 0;
}
