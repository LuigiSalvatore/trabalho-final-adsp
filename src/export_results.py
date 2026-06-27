"""
Módulo para processamento em lote de imagens, exportação de resultados e geração de relatórios.
"""
import os
import glob
import pandas as pd
from src.speed_extractor import SpeedExtractor

def process_directory(base_dir, max_samples_per_folder=10):
    """
    Varre os diretórios de amostras e extrai as velocidades.
    max_samples_per_folder: limite de amostras por pasta para execução rápida de demonstração.
    """
    extractor = SpeedExtractor(use_easyocr=True)
    results = []
    
    # Encontrar todas as imagens (.jpg, .png, .jpeg)
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    
    for root, dirs, files in os.walk(base_dir):
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(root, ext)))
            
        if not image_files:
            continue
            
        # Ordenar e limitar amostras
        image_files = sorted(image_files)[:max_samples_per_folder]
        
        for img_path in image_files:
            rel_path = os.path.relpath(img_path, base_dir)
            folder_name = os.path.basename(os.path.dirname(img_path))
            filename = os.path.basename(img_path)
            
            try:
                speed, raw_text, method = extractor.process_image(img_path)
            except Exception as e:
                speed, raw_text, method = None, f"Erro: {str(e)}", "Falha"
                
            results.append({
                "Caminho_Relativo": rel_path,
                "Pasta_Amostra": folder_name,
                "Arquivo": filename,
                "Velocidade_Extraida_kmh": speed,
                "Metodo_OCR": method,
                "Texto_Reconhecido": raw_text
            })
            
    df = pd.DataFrame(results)
    return df

def save_results(df, output_csv="resultados_extracao_velocidade.csv"):
    """Salva os resultados em formato CSV estruturado."""
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Resultados salvos com sucesso em: {output_csv}")
