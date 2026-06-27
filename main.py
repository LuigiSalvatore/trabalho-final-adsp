"""
Script principal de execução do Projeto ADSP - Processamento Digital de Sinais.
Extração automática de velocidade em fotos de medidores/radares (LABELO PUCRS).
"""
import os
import sys
from src.export_results import process_directory, save_results

def main():
    print("=========================================================")
    print("  PROJETO ADSP - AUTOMATIZAÇÃO DE TRIAGEM DE INFRAÇÕES  ")
    print("  Cliente: LABELO PUCRS                                 ")
    print("=========================================================\n")
    
    dataset_dir = os.path.join(os.path.dirname(__file__), "Fotos_Medidor de velocidade")
    if not os.path.exists(dataset_dir):
        print(f"Erro: Pasta de fotos não encontrada em '{dataset_dir}'")
        sys.exit(1)
        
    print(f"Iniciando processamento de amostras em: {dataset_dir}")
    
    # Processa amostras representativas para demonstração e geração de estatísticas
    df_results = process_directory(dataset_dir, max_total_samples=20)
    
    print("\n--- RESUMO DA EXTRAÇÃO ---")
    total = len(df_results)
    if total > 0:
        sucessos = df_results['Velocidade_Extraida_kmh'].notnull().sum()
        print(f"Total de imagens processadas nesta rodada: {total}")
        print(f"Leituras de velocidade extraídas com sucesso: {sucessos} / {total} ({sucessos/total*100:.1f}%)")
    
    output_path = "resultados_extracao_velocidade.csv"
    save_results(df_results, output_path)
    
    print("\nAmostra dos resultados obtidos:")
    print(df_results[['Arquivo', 'Velocidade_Extraida_kmh', 'Metodo_OCR']].head(10).to_string())
    print("\nProcessamento concluído com sucesso!")

if __name__ == "__main__":
    main()
