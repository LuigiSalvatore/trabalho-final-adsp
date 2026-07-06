"""
Script principal de execução do Projeto ADSP - Processamento Digital de Sinais.
Extração automática de velocidade em fotos de medidores/radares (LABELO PUCRS).
"""
import os
import sys
import warnings

# Forçar UTF-8 no stdout/stderr para suportar caracteres especiais (█, ░, ã, etc.)
# no terminal do Windows que usa cp1252 por padrão.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Suprimir avisos do PyTorch e outros antes de qualquer import pesado
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def main():
    from src.export_results import process_directory, save_results

    print("=========================================================")
    print("  PROJETO ADSP - AUTOMATIZAÇÃO DE TRIAGEM DE INFRAÇÕES  ")
    print("  Cliente: LABELO PUCRS                                 ")
    print("=========================================================\n")

    dataset_dir = os.path.join(os.path.dirname(__file__), "Fotos_Medidor de velocidade")
    if not os.path.exists(dataset_dir):
        print(f"Erro: Pasta de fotos não encontrada em '{dataset_dir}'")
        sys.exit(1)

    print(f"Iniciando processamento de amostras em: {dataset_dir}")

    # Flags de controle do terminal
    reset = "--reset" in sys.argv
    process_all = "--all" in sys.argv

    # Número de workers: lê --workers N se fornecido, senão usa cpu_count - 6
    n_workers = None
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        try:
            n_workers = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Aviso: flag --workers requer um número inteiro. Usando default.")

    cpu_count = os.cpu_count() or 2
    effective_workers = n_workers if n_workers is not None else max(1, cpu_count - 6)

    if process_all:
        print(f"Modo COMPLETO ativado: Processando TODAS as imagens do repositório...")
        print(f"Workers paralelos: {effective_workers} de {cpu_count} disponíveis")
        max_samples = None
    else:
        print("Modo DEMONSTRAÇÃO ativado: Processando amostra de 20 imagens espalhadas pelas pastas...")
        print(f"Workers paralelos: {effective_workers} de {cpu_count} disponíveis")
        print("(Para processar todas as imagens, execute: python main.py --all)")
        print("(Para reiniciar o progresso, adicione: --reset)")
        print("(Para definir workers manualmente: --workers N)\n")
        max_samples = 20

    df_results = process_directory(
        dataset_dir,
        max_total_samples=max_samples,
        reset=reset,
        n_workers=n_workers,
    )

    print("--- RESUMO DA EXTRAÇÃO ---")
    total = len(df_results)
    if total > 0:
        sucessos = df_results['Velocidade_Extraida_kmh'].notnull().sum()
        print(f"Total de imagens processadas nesta rodada: {total}")
        print(f"Leituras de velocidade extraídas com sucesso: {sucessos} / {total} ({sucessos/total*100:.1f}%)")

    save_results(df_results)

    print("\nAmostra dos resultados obtidos:")
    print(df_results[['Arquivo', 'Velocidade_Extraida_kmh', 'Metodo_OCR']].head(10).to_string())
    print("\nProcessamento concluído com sucesso!")


# Guard obrigatório para multiprocessing no Windows (spawn)
if __name__ == "__main__":
    main()
