import os
import subprocess

# Configurações
inicio_T = 0
fim_T = 500
passo_T = 10
valores_T = list(range(inicio_T, fim_T + 1, passo_T))
N = 2000  # Tamanho fixo da matriz
threads_per_block = (8, 8)  # Threads por bloco em cada dimensão
blocks_per_grid = (N // threads_per_block[0], N // threads_per_block[1])
num_threads_totais = threads_per_block[0] * threads_per_block[1] * blocks_per_grid[0] * blocks_per_grid[1]

# Arquivos e executáveis
arquivo_cuda = "diffusion.cu"
executavel_cuda = "diffusion"
resultados_cuda = "tempos_execucao_cuda.txt"

# Compilar o programa CUDA
print("Compilando o programa CUDA...")
subprocess.run(["nvcc", arquivo_cuda, "-o", executavel_cuda], check=True)

# Criar diretório de saída
os.makedirs("results/cuda", exist_ok=True)

# Executar o programa CUDA e salvar tempos
tempos_cuda = []

with open(resultados_cuda, "w") as f:
    for T in valores_T:
        # Executar programa CUDA
        print(f"Executando o programa CUDA com T={T}...")
        resultado_cuda = subprocess.run([f"./{executavel_cuda}", str(N), str(T)], capture_output=True, text=True)
        if resultado_cuda.returncode == 0:
            tempo_cuda = float(resultado_cuda.stdout.strip().split("\n")[-1].split(":")[-1].strip().split()[0])
            tempos_cuda.append(tempo_cuda)
            f.write(f"{T},{tempo_cuda:.6f}\n")
            print(f"Tempo CUDA: {tempo_cuda:.2f} segundos")
        else:
            print(f"Erro ao executar o CUDA com T={T}: {resultado_cuda.stderr}")

print("Execuções CUDA concluídas e tempos salvos.")

