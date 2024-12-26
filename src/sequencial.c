# include <stdio.h>
# include <stdlib.h>
# include <omp.h>
# include<math.h>

//# define N 2000 // Tamanho da grade
int N = 0;
# define D 0.1 // Coeficiente de difusão
# define DELTA_T 0.01
# define DELTA_X 1.0

FILE * saida_diff;

void diff_eq(double ***C, int T) {
    double difmedio = 0.0;
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[(t+1)%2][i][j] = C[t%2][i][j] + D * DELTA_T * ((C[t%2][i+1][j] + C[t%2][i-1][j] + C[t%2][i][j+1] + C[t%2][i][j-1] - 4 * C[t%2][i][j]) / (DELTA_X));
                difmedio += fabs(C[(t + 1) % 2][i][j] - C[t % 2][i][j]);
            }
        }
        if ((t%100) == 0)
          fprintf(saida_diff, "%d,%g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main(int argc, char ** argv) {
    if(argc < 3){
        printf("./sequencial n_iteracoes tamanho_matriz\n");
        exit(0);
    }
    int T = atoi(argv[1]);
    N = atoi(argv[2]);

    char arquivo_matriz[100];
    char arquivo_diff[100];
    sprintf(arquivo_matriz, "/home/belos/Documents/ContaminantDiffusionCuda/results/seq/matriz/%d_%d.csv", N, T);
    sprintf(arquivo_diff, "/home/belos/Documents/ContaminantDiffusionCuda/results/seq/diff/%d_%d.csv", N, T);
    FILE * saida_matriz = fopen(arquivo_matriz, "w+");
    saida_diff = fopen(arquivo_diff, "w+");

    double ***C = (double ***)malloc(2 * sizeof(double **));
    for(int t = 0; t<2; t++){
        C[t] = (double **)malloc(N * sizeof(double *));
        for (int i = 0; i < N; i++) {
            C[t][i] = (double *)malloc(N * sizeof(double));
        }
    }
    // double C[2][N][N] = {0}; // Concentração inicial

    for (int t = 0; t < 2; t++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[t][i][j] = 0.0;

    C[0][N/2][N/2] = 1.0; // Inicializar uma concentração alta no centro
    
    diff_eq(C, T);// Executar a equação de difusão
    
    
    printf("Concentração final no centro: %f\n", C[T%2][N/2][N/2]); // Exibir resultado para verificação

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if(j == N-1){
                fprintf(saida_matriz, "%f\n", C[T%2][i][j]);
            }
            else{
                fprintf(saida_matriz, "%f,", C[T%2][i][j]);
            }
        }
    }

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < N; i++) {
            free(C[t][i]);
        }
        free(C[t]);
    }
    free(C);
    fclose(saida_matriz);
    fclose(saida_diff);

    return 0;
}
