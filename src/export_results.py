"""
Módulo para processamento em lote de imagens, exportação de resultados e geração de relatórios.
"""
import os
import glob
import pandas as pd
from src.speed_extractor import SpeedExtractor

def process_directory(base_dir, max_total_samples=30):
    """
    Varre os diretórios de amostras e extrai as velocidades.
    max_total_samples: limite total de imagens para demonstração rápida.
    """
    extractor = SpeedExtractor(use_easyocr=True)
    print(f"Dispositivo de processamento OCR detectado: {extractor.hardware_device}")
    results = []
    
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    all_image_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for ext in extensions:
            for filepath in glob.glob(os.path.join(root, ext)):
                all_image_files.append(filepath)
                
    # Ordenar e selecionar amostras representativas espalhadas pelas pastas
    all_image_files = sorted(all_image_files)
    if max_total_samples is not None:
        step = max(1, len(all_image_files) // max_total_samples) if all_image_files else 1
        selected_files = all_image_files[::step][:max_total_samples]
    else:
        selected_files = all_image_files
    
    print(f"Total de {len(all_image_files)} imagens encontradas. Processando {len(selected_files)} imagens...")
    
    for idx, img_path in enumerate(selected_files, 1):
        rel_path = os.path.relpath(img_path, base_dir)
        parts = rel_path.split(os.sep)
        amostra_folder = parts[0] if len(parts) > 1 else "Raiz"
        filename = os.path.basename(img_path)
        
        print(f"[{idx}/{len(selected_files)}] Processando: {filename} ({amostra_folder})...")
        try:
            speed, raw_text, method = extractor.process_image(img_path)
        except Exception as e:
            speed, raw_text, method = None, f"Erro: {str(e)}", "Falha"
            
        results.append({
            "Caminho_Relativo": rel_path,
            "Pasta_Amostra": amostra_folder,
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
