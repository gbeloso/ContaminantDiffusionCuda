import matplotlib.pyplot as plt

def ler_dados_arquivo(nome_arquivo):
    """Lê os dados do arquivo e retorna duas listas: interações e tempos."""
    interacoes = []
    tempos = []
    try:
        with open(nome_arquivo, 'r') as arquivo:
            for linha in arquivo:
                if linha.strip():
                    partes = linha.strip().split(',')
                    if len(partes) == 2:
                        try:
                            interacoes.append(int(partes[0]))
                            tempos.append(float(partes[1]))
                        except ValueError:
                            print(f"Linha ignorada devido a formato incorreto: {linha.strip()}")
    except FileNotFoundError:
        print(f"Erro: Arquivo {nome_arquivo} não encontrado.")
    return interacoes, tempos

def plotar_grafico(interacoes_cuda, tempos_cuda, interacoes_seq, tempos_seq):
    plt.figure(figsize=(10, 6))

    tempos_cuda_ajustados = [tempo * 10 for tempo in tempos_cuda]  # Multiplicando por 10

    # Plot CUDA
    plt.plot(interacoes_cuda, tempos_cuda_ajustados, marker='o', label='CUDA (x10)', linestyle='--')

    # Plot Sequencial
    plt.plot(interacoes_seq, tempos_seq, marker='s', label='Sequencial')

    # Configurações do gráfico
    plt.title("Análise de Tempo de Execução")
    plt.xlabel("Número de Interações")
    plt.ylabel("Tempo de Execução (segundos)")
    plt.legend()
    plt.grid(True)

    # Salvar e exibir gráfico
    plt.savefig("analise_tempo_execucao.png")
    plt.show()

def main():
    arquivo_cuda = "tempos_execucao_cuda.txt"
    arquivo_sequencial = "tempos_execucao_sequencial (1).txt"

    # Ler dados dos arquivos
    interacoes_cuda, tempos_cuda = ler_dados_arquivo(arquivo_cuda)
    interacoes_seq, tempos_seq = ler_dados_arquivo(arquivo_sequencial)

    if not interacoes_cuda or not interacoes_seq:
        print("Erro: Não foi possível carregar os dados dos arquivos fornecidos.")
        return

    # Plotar gráfico
    plotar_grafico(interacoes_cuda, tempos_cuda, interacoes_seq, tempos_seq)

if __name__ == "__main__":
    main()

