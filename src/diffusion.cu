%%writefile diffusion.cu

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int N = 0;
#define D 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0

__global__ void diff_eq(double * vet, double * reductionMatrix, int num, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x > 0 && x < N-1 && y > 0 && y < N-1){
        int z_e = ((num+1)%2);
        int z_l = ((num)%2);

        vet[x + y * N + z_e * N * N] = vet[x + y * N + z_l * N * N]
            + D * DELTA_T *
            (
                (
                    vet[(x+1) + y * N + z_l * N * N] +
                    vet[(x-1) + y * N + z_l * N * N] +
                    vet[x + (y+1) * N + z_l * N * N] +
                    vet[x + (y-1) * N + z_l * N * N] -
                    4 * vet[x + y * N + z_l * N * N]
                ) /
                (DELTA_X * DELTA_X)
            );
        reductionMatrix[x + y * N] += fabs(vet[x + y * N + z_e * N * N] - vet[x + y * N + z_l * N * N]);
    }
}

__global__ void reduceSum(double *input, double *output, int size) {
    extern __shared__ double sharedData[];

    // Índices
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Carregar elementos da matriz na memória compartilhada
    sharedData[tid] = (index < size) ? input[index] : 0.0f;
    __syncthreads();

    // Redução paralela na memória compartilhada
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // O primeiro thread de cada bloco armazena o resultado parcial
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main(int argc, char ** argv)
{
    if(argc < 3){
        printf("./diffusion tamanho_matriz n_iteracoes\n");
        exit(0);
    }

    int h_num = atoi(argv[2]);
    N = atoi(argv[1]);
    int depth = 2;
    int size = N * N * depth * sizeof(double);

    char arquivo_matriz[200];
    sprintf(arquivo_matriz, "results/cuda/matriz/%d_%d.csv", N, h_num);
    FILE * saida_matriz = fopen(arquivo_matriz, "w+");
    char arquivo_diff[200];
    sprintf(arquivo_diff, "results/cuda/diff/%d_%d.csv", N, h_num);
    FILE * saida_diff = fopen(arquivo_diff, "w+");
    FILE * saida_tempo = fopen("results/cuda/time.csv", "a+");

    // Alocação no Host
    double *h_C = (double *)malloc(size);
    double *h_partialSums = (double *)malloc(ceil((N*N)/1024) * sizeof(double));

    //Alocação no device
    double *d_C, *d_reductionMatrix, *d_partialSums;
    cudaMalloc((void **)&d_C, size);
    cudaMalloc((void **)&d_reductionMatrix, N * N *sizeof(double));
    cudaMalloc((void **)&d_partialSums, ceil((N*N)/1024) * sizeof(double));

    for (int i = 0; i < N * N * depth; i++) {
        h_C[i] = 0.0; // Inicialização com valores de exemplo
    }

    h_C[N/2 + N/2 * N + 0 * N * N] = 1.0;

    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid(ceil((double)N/threadsPerBlock.x), ceil((double)N/threadsPerBlock.y));
    int threadsPerBlockReduce = 1024;
    int blocksPerGridReduce = ceil((N*N)/threadsPerBlockReduce);
    size_t sharedMemSize = threadsPerBlockReduce * sizeof(double);

    clock_t start = clock();

    for(int t = 0; t < h_num; t++){

        diff_eq<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_reductionMatrix, t, N);
        cudaDeviceSynchronize();

        if(t%100 == 0){
            reduceSum<<<blocksPerGridReduce, threadsPerBlockReduce, sharedMemSize>>>(d_reductionMatrix, d_partialSums, N*N);
            cudaMemcpy(h_partialSums, d_partialSums, blocksPerGridReduce * sizeof(double), cudaMemcpyDeviceToHost);

            double diffMedio = 0.0;
            for(int i = 0; i < blocksPerGridReduce; i++){
                diffMedio+=h_partialSums[i];
            }
            fprintf(saida_diff, "%d,%g\n", t, diffMedio/((N-2)*(N-2)));

            printf("Iteração %d - Diferença média: %g\n", t, diffMedio / ((N - 2) * (N - 2)));
        }

    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost );

    clock_t end = clock();

    fprintf(saida_tempo,"%d,%d,%g\n", N,h_num,((double)(end - start)) / CLOCKS_PER_SEC);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int index = ((h_num%2) * N * N) + (i * N) + j;
            if(j==N-1){
                fprintf(saida_matriz, "%f\n", h_C[index]);
            }
            else{
                fprintf(saida_matriz, "%f,", h_C[index]);
            }
        }
    }

    printf("Concentração final no centro: %f\n", h_C[N/2 + N/2 * N + (h_num%2) * N * N]);

    printf("Tempo de execução: %.2f segundos\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaFree(d_C);
    cudaFree(d_reductionMatrix);
    cudaFree(d_partialSums);
    free(h_C);
    free(h_partialSums);
    fclose(saida_diff);
    fclose(saida_matriz);
}
