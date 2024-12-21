#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 2000
#define D 0.1
#define DELTA_T 0.01
#define DELTA_X 1.0

__global__ void diff_eq(double * vet, int * num)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x > 0 && x < N-1 && y > 0 && y < N-1){    
        int z_e = ((*num+1)%2);
        int z_l = ((*num)%2);

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
    }
}

int main(int argc, char ** argv)
{
    int width = N, height = N, depth = 2;
    int size = width * height * depth * sizeof(double);
    int h_num = atoi(argv[1]);
    int *d_num;

    char arquivo[100];
    sprintf(arquivo, "%d_cuda.csv", h_num);
    FILE * saida = fopen(arquivo, "w+");

    // Alocação no Host
    double *h_C = (double *)malloc(size);
    double *d_C;

    cudaMalloc((void **)&d_C, size);
    cudaMalloc((void **)&d_num, sizeof(int));

    for (int i = 0; i < width * height * depth; i++) {
        h_C[i] = 0.0; // Inicialização com valores de exemplo
    }
    
    h_C[N/2 + N/2 * width + 0 * N * N] = 1.0;

    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid(ceil((double)N/threadsPerBlock.x), ceil((double)N/threadsPerBlock.y));
    
    for(int t = 0; t < h_num; t++){
        
        cudaMemcpy(d_num, &t, sizeof(int), cudaMemcpyHostToDevice);
        diff_eq<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_num);
        cudaDeviceSynchronize();
    
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost );


    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = ((h_num%2) * height * width) + (i * width) + j;
            if(j==N-1){
                fprintf(saida, "%f\n", h_C[index]);
            }
            else{
                fprintf(saida, "%f,", h_C[index]);
            }
        }
    }

    cudaFree(d_C);
    free(h_C);

} 