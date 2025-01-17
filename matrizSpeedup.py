import pandas as pd

def ler_tempos_arquivo(nome_arquivo):
    """Lê os tempos do arquivo e retorna uma lista."""
    tempos = []
    try:
        with open(nome_arquivo, 'r') as arquivo:
            for linha in arquivo:
                if linha.strip():
                    try:
                        tempos.append(float(linha.strip()))
                    except ValueError:
                        print(f"Linha ignorada: {linha.strip()}")
    except FileNotFoundError:
        print(f"Arquivo {nome_arquivo} não encontrado.")
    return tempos

def calcular_speedup_500_interacoes(tempos_sequencial, tempos_openmp, tempos_cuda, threads):
    """Calcula o speedup para cada abordagem considerando 500 interações."""
    index_500 = 49  # Índice correspondente às 500 interações (0, 10, ..., 500)

    speedups = {}

    # Speedup para CUDA
    speedups['CUDA'] = tempos_sequencial[index_500] / tempos_cuda[index_500] if tempos_cuda[index_500] != 0 else 0

    # Speedups para OpenMP com várias threads
    for t in threads:
        speedups[f'OpenMP ({t} threads)'] = (
            tempos_sequencial[index_500] / tempos_openmp[t][index_500]
            if tempos_openmp[t][index_500] != 0 else 0
        )

    return speedups

def main():
    # Arquivos de entrada
    arquivo_sequencial = "tempos_execucao_sequencial (1).txt"
    arquivos_openmp = {
        2: "tempos_execucao_pararelo2.txt",
        4: "tempos_execucao_pararelo4.txt",
        8: "tempos_execucao_pararelo8.txt",
        16: "tempos_execucao_pararelo16.txt"
    }
    arquivo_cuda = "tempos_execucao_cuda.txt"

    threads = [2, 4, 8, 16]  # Número de threads para OpenMP

    # Ler dados dos arquivos
    tempos_sequencial = ler_tempos_arquivo(arquivo_sequencial)
    tempos_openmp = {}
    for t, arquivo in arquivos_openmp.items():
        tempos_openmp[t] = ler_tempos_arquivo(arquivo)
    tempos_cuda = ler_tempos_arquivo(arquivo_cuda)

    # Verificar se os arquivos têm pelo menos 50 entradas
    if not (len(tempos_sequencial) > 49 and len(tempos_cuda) > 49 and all(len(tempos_openmp[t]) > 49 for t in threads)):
        print("Erro: Os arquivos não possuem dados suficientes para 500 interações.")
        return

    # Calcular os speedups para 500 interações
    speedups = calcular_speedup_500_interacoes(tempos_sequencial, tempos_openmp, tempos_cuda, threads)

    # Exibir resultados
    print("Speedups para 500 interações:")
    for abordagem, valor in speedups.items():
        print(f"{abordagem}: {valor:.2f}")

    # Salvar resultados em CSV
    df_speedups = pd.DataFrame(list(speedups.items()), columns=["Abordagem", "Speedup"])
    df_speedups.to_csv("speedups_500_interacoes.csv", index=False)
    print("Resultados salvos em 'speedups_500_interacoes.csv'.")

if __name__ == "__main__":
    main()

