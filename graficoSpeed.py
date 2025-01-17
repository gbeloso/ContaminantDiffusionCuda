import matplotlib.pyplot as plt

# Função para calcular o speedup e a eficiência
def calcular_speedup_eficiencia(tempos_sequenciais, tempos_cuda, num_threads):
    speedup = [t_seq / t_cuda for t_seq, t_cuda in zip(tempos_sequenciais, tempos_cuda)]
    eficiencia = [sp / num_threads for sp in speedup]
    return speedup, eficiencia

# Configurações
arquivo_sequencial_resultados = "tempos_execucao_sequencial.txt"
arquivo_cuda_resultados = "tempos_execucao_cuda.txt"
threads_per_block = (8, 8)  # Threads por bloco em cada dimensão
blocks_per_grid = (2000 // threads_per_block[0], 2000 // threads_per_block[1])
num_threads_totais = threads_per_block[0] * threads_per_block[1] * blocks_per_grid[0] * blocks_per_grid[1]

# Ler os tempos sequenciais do arquivo
tempos_sequenciais = []
valores_T = []
with open(arquivo_sequencial_resultados, "r") as f:
    for linha in f:
        if "," in linha:
            T, tempo_seq = linha.strip().split(",")
            valores_T.append(int(T))
            tempos_sequenciais.append(float(tempo_seq))

# Ler os tempos CUDA do arquivo
tempos_cuda = []
with open(arquivo_cuda_resultados, "r") as f:
    for linha in f:
        if "," in linha:
            _, tempo_cuda = linha.strip().split(",")
            tempos_cuda.append(float(tempo_cuda))

# Calcular speedup e eficiência
speedup, eficiencia = calcular_speedup_eficiencia(tempos_sequenciais, tempos_cuda, num_threads_totais)

# Gerar gráficos
plt.figure(figsize=(14, 7))

# Gráfico de Speedup
plt.subplot(1, 2, 1)
plt.plot(valores_T, speedup, marker='o', label="Speedup Real", color='blue')
plt.plot(valores_T, [min(T, num_threads_totais) for T in valores_T], linestyle='--', label="Speedup Linear (Ideal)", color='red')
plt.title("Gráfico de Speedup")
plt.xlabel("Número de Iterações (T)")
plt.ylabel("Speedup")
plt.grid(True)
plt.legend()

# Gráfico de Eficiência
plt.subplot(1, 2, 2)
plt.plot(valores_T, eficiencia, marker='o', label="Eficiência Real", color='blue')
plt.axhline(y=1, color='red', linestyle='--', label="Eficiência Linear")
plt.title("Gráfico de Eficiência")
plt.xlabel("Número de Iterações (T)")
plt.ylabel("Eficiência")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("comparacao_speedup_eficiencia.png")
plt.show()

print("Gráficos gerados e salvos como 'comparacao_speedup_eficiencia.png'.")

