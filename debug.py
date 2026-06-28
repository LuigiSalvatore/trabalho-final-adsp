"""
Utilitário de Depuração para validação das velocidades extraídas no Projeto ADSP.
Suporta análise via CSV, análise direta por OCR de arquivos específicos (-f) ou listas de arquivos (-l).
"""
import os
import sys
import re
import warnings
import argparse
import pandas as pd

# Silencia avisos do PyTorch e do sistema
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

CSV_FILE = "resultados_extracao_velocidade.csv"
DATASET_DIR = "Fotos_Medidor de velocidade"

# Valores padrão esperados por Amostra: (VelMax, VelMed, VelConsid) ou (VelMax, VelMed)
EXPECTED_VALUES = {
    "Amostra 1": [
        (40, 342, 318),
        (40, 90, 83)
    ],
    "Amostra 2": [
        (20, 40)
    ],
    "Amostra 3": [
        (30, 72, 65)
    ],
    "Amostra 4": [
        (40, 150, 140),
        (40, 125, 116)
    ]
}

def find_file_by_name(base_dir, filename):
    """Procura recursivamente um arquivo pelo nome base e retorna seu caminho completo."""
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def extract_values_from_text(text, amostra):
    """
    Usa regex para buscar os valores de velocidade no texto reconhecido.
    Retorna uma tupla com os valores numéricos encontrados de acordo com a amostra.
    """
    if pd.isna(text) or not isinstance(text, str):
        return (None, None, None) if amostra != "Amostra 2" else (None, None)

    # Encontrar todas as ocorrências de números no texto (limitados a valores razoáveis de velocidade < 500)
    numbers = [int(val) for val in re.findall(r'\b\d+\b', text)]
    candidates = [n for n in numbers if 5 <= n < 500]  # ignora números muito pequenos ou grandes (anos/IDs)
    
    v_max = None
    v_med = None
    v_cons = None
    
    # Regex para VelMax
    max_match = re.search(r'vel[^\d]*max[^\d]*[:\.\s]+(\d+)', text, re.IGNORECASE)
    if max_match:
        v_max = int(max_match.group(1))
        
    # Regex para VelMed / Medida
    med_match = re.search(r'vel[^\d]*(?:med|real)[^\d]*[:\.\s]+(\d+)', text, re.IGNORECASE)
    if med_match:
        v_med = int(med_match.group(1))
        
    # Regex para VelConsid
    cons_match = re.search(r'vel[^\d]*cons[^\d]*[:\.\s]+(\d+)', text, re.IGNORECASE)
    if cons_match:
        v_cons = int(cons_match.group(1))

    # Fallbacks e regras específicas por amostra se o OCR falhar nas marcações específicas
    if amostra == "Amostra 2":
        if v_max is None and len(candidates) >= 1:
            v_max = candidates[0]
        if v_med is None and len(candidates) >= 2:
            v_med = candidates[1]
        return (v_max, v_med)
        
    elif amostra == "Amostra 3":
        reg_match = re.search(r'reg[^\d]*(\d+)', text, re.IGNORECASE)
        med3_match = re.search(r'medida[^\d]*(\d+)', text, re.IGNORECASE)
        cons3_match = re.search(r'considerada[^\d]*(\d+)', text, re.IGNORECASE)
        
        v_reg = int(reg_match.group(1)) if reg_match else None
        v_med = int(med3_match.group(1)) if med3_match else None
        v_cons = int(cons3_match.group(1)) if cons3_match else None
        
        if len(candidates) >= 3:
            if v_reg is None: v_reg = candidates[-3]
            if v_med is None: v_med = candidates[-2]
            if v_cons is None: v_cons = candidates[-1]
        return (v_reg, v_med, v_cons)
        
    else: # Amostra 1 e 4 (esperam 3 números: Max, Med, Consid)
        if len(candidates) >= 3:
            if v_max is None: v_max = candidates[0]
            if v_med is None: v_med = candidates[1]
            if v_cons is None: v_cons = candidates[2]
        return (v_max, v_med, v_cons)

def analyze_csv():
    """Lê o arquivo CSV de resultados e analisa as anomalias instantaneamente."""
    print(f"Lendo resultados do CSV '{CSV_FILE}'...")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")
        return False

    anomalias = []
    verificados = 0

    for idx, row in df.iterrows():
        amostra_folder = row["Pasta_Amostra"]
        filename = row["Arquivo"]
        full_text = row["Texto_Reconhecido"]
        
        if amostra_folder not in EXPECTED_VALUES:
            continue
            
        verificados += 1
        val_tuple = extract_values_from_text(full_text, amostra_folder)
        
        targets = EXPECTED_VALUES[amostra_folder]
        matched = False
        for target in targets:
            if val_tuple == target:
                matched = True
                break
                
        if not matched:
            anomalias.append({
                "Arquivo": filename,
                "Amostra": amostra_folder,
                "Valores_Encontrados": val_tuple,
                "Valores_Esperados": targets,
                "Texto_Completo": full_text
            })
            print(f"[AVISO] ANOMALIA em {filename} ({amostra_folder}): Encontrado {val_tuple} | Esperado {targets}")

    print("\n=========================================================")
    print("  RELATÓRIO DE ANOMALIAS (ANÁLISE VIA CSV)               ")
    print("=========================================================")
    print(f"Total de registros analisados no CSV: {verificados}")
    print(f"Total de anomalias encontradas: {len(anomalias)}")
    
    if len(anomalias) > 0:
        print("\nLista de imagens com valores discrepantes:")
        for an in anomalias:
            print(f"- [{an['Amostra']}] {an['Arquivo']} -> Encontrado: {an['Valores_Encontrados']} | Esperado: {an['Valores_Esperados']}")
    else:
        print("\n[OK] Sucesso! Nenhuma anomalia de velocidade encontrada nos dados do CSV.")
    return True

def process_single_image(reader, img_path):
    """Processa uma única imagem com OCR e extrai valores."""
    from src.image_processing import crop_metadata_region, enhance_text_region, load_image
    
    filename = os.path.basename(img_path)
    rel_path = os.path.relpath(img_path, DATASET_DIR)
    parts = rel_path.split(os.sep)
    amostra_folder = parts[0] if len(parts) > 1 else "Desconhecido"
    
    img = load_image(img_path)
    region = 'top' if amostra_folder == "Amostra 3" else 'bottom'
    cropped = crop_metadata_region(img, region=region, ratio=0.35)
    
    results = reader.readtext(cropped, detail=0)
    full_text = " ".join(results)
    
    val_tuple = extract_values_from_text(full_text, amostra_folder)
    targets = EXPECTED_VALUES.get(amostra_folder, [])
    
    matched = False
    for target in targets:
        if val_tuple == target:
            matched = True
            break
            
    status = "[OK]" if matched else "[ANOMALIA]"
    print(f"\n{status} Arquivo: {filename} ({amostra_folder})")
    print(f"  -> Valores Extraídos: {val_tuple}")
    print(f"  -> Valores Esperados: {targets}")
    print(f"  -> Texto OCR: {full_text}")
    return matched

def main():
    parser = argparse.ArgumentParser(description="Programa de Depuração de Velocidades (ADSP/LABELO).")
    parser.add_argument("-f", "--file", help="Nome do arquivo de imagem específico para analisar diretamente via OCR.")
    parser.add_argument("-l", "--list", help="Caminho do arquivo TXT contendo a lista de arquivos de imagem para analisar.")
    args = parser.parse_args()

    print("=========================================================")
    print("  PROGRAMA DE DEBURAÇÃO - VALIDAÇÃO DE VELOCIDADES (ADSP)")
    print("=========================================================")

    # Se foi solicitada análise de imagem específica ou lista de imagens
    if args.file or args.list:
        import easyocr
        import torch
        
        has_gpu = torch.cuda.is_available()
        print(f"Inicializando EasyOCR no dispositivo: {'GPU (CUDA)' if has_gpu else 'CPU'}...")
        reader = easyocr.Reader(['pt', 'en'], gpu=has_gpu, verbose=False)

        if args.file:
            print(f"\nBuscando arquivo '{args.file}' no dataset...")
            img_path = find_file_by_name(DATASET_DIR, args.file)
            if not img_path:
                print(f"Erro: Arquivo '{args.file}' não foi encontrado em '{DATASET_DIR}'.")
                sys.exit(1)
            process_single_image(reader, img_path)

        elif args.list:
            if not os.path.exists(args.list):
                print(f"Erro: Arquivo de lista '{args.list}' não encontrado.")
                sys.exit(1)
                
            print(f"\nLendo lista de arquivos de '{args.list}'...")
            with open(args.list, 'r', encoding='utf-8') as f:
                filenames = [line.strip() for line in f if line.strip()]
                
            print(f"Iniciando varredura OCR para {len(filenames)} arquivos...")
            success_count = 0
            for idx, fname in enumerate(filenames, 1):
                img_path = find_file_by_name(DATASET_DIR, fname)
                if not img_path:
                    print(f"\n[ERRO] Arquivo [{idx}/{len(filenames)}] '{fname}' não encontrado.")
                    continue
                matched = process_single_image(reader, img_path)
                if matched:
                    success_count += 1
            print(f"\nVerificação concluída: {success_count}/{len(filenames)} arquivos correspondentes.")
        return

    # Modo padrão: análise rápida via CSV ou fallback de OCR completo
    if os.path.exists(CSV_FILE):
        success = analyze_csv()
        if success:
            return

    # Fallback OCR completo nas imagens se o CSV não existir
    print(f"Aviso: Arquivo '{CSV_FILE}' não encontrado.")
    print("Executando análise varrendo e rodando OCR diretamente nas imagens...")
    
    import easyocr
    import torch
    from src.image_processing import crop_metadata_region, load_image
    import glob

    has_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(['pt', 'en'], gpu=has_gpu, verbose=False)
    
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    all_files = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for ext in extensions:
            for filepath in glob.glob(os.path.join(root, ext)):
                all_files.append(filepath)
                
    all_files = sorted(all_files)
    print(f"Total de {len(all_files)} imagens encontradas para verificação.")
    
    anomalias = 0
    for img_path in all_files:
        rel_path = os.path.relpath(img_path, DATASET_DIR)
        parts = rel_path.split(os.sep)
        amostra_folder = parts[0] if len(parts) > 1 else "Desconhecido"
        if amostra_folder not in EXPECTED_VALUES:
            continue
            
        matched = process_single_image(reader, img_path)
        if not matched:
            anomalias += 1
            
    print(f"\nVarredura finalizada. Anomalias totais: {anomalias}")

if __name__ == "__main__":
    main()
