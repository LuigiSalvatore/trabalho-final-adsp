"""
Utilitário de Depuração para validação das velocidades extraídas no Projeto ADSP.
Verifica se existem imagens com velocidades diferentes dos padrões esperados de cada Amostra.
"""
import os
import sys
import re
import warnings

# Silencia avisos do PyTorch e do sistema
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import easyocr
import torch
from src.image_processing import load_image, crop_metadata_region, enhance_text_region

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

def extract_values_from_text(text, amostra):
    """
    Usa regex para buscar os valores de velocidade no texto reconhecido.
    Retorna uma tupla com os valores numéricos encontrados de acordo com a amostra.
    """
    # Encontrar todas as ocorrências de números no texto (limitados a valores razoáveis de velocidade < 500)
    numbers = [int(val) for val in re.findall(r'\b\d+\b', text)]
    candidates = [n for n in numbers if 5 <= n < 500]  # ignora números muito pequenos ou grandes (anos/IDs)
    
    # Extração baseada em padrões para garantir exatidão
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
        # Amostra 2 geralmente tem apenas Max e Med (ex: VELMAX: 20, VELMED: 40)
        if v_max is None and len(candidates) >= 1:
            v_max = candidates[0]
        if v_med is None and len(candidates) >= 2:
            v_med = candidates[1]
        return (v_max, v_med)
        
    elif amostra == "Amostra 3":
        # Amostra 3: Vel. Reg km/h Vel. Medida km/h Vel.Considerada
        # Procura por números próximos a padrões específicos
        reg_match = re.search(r'reg[^\d]*(\d+)', text, re.IGNORECASE)
        med3_match = re.search(r'medida[^\d]*(\d+)', text, re.IGNORECASE)
        cons3_match = re.search(r'considerada[^\d]*(\d+)', text, re.IGNORECASE)
        
        v_reg = int(reg_match.group(1)) if reg_match else None
        v_med = int(med3_match.group(1)) if med3_match else None
        v_cons = int(cons3_match.group(1)) if cons3_match else None
        
        # Fallback para os 3 números da tarja de tabela se a detecção de palavra quebrar
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

def main():
    print("=========================================================")
    print("  PROGRAMA DE DEBURAÇÃO - VALIDAÇÃO DE VELOCIDADES (ADSP)")
    print("=========================================================\n")
    
    dataset_dir = "Fotos_Medidor de velocidade"
    if not os.path.exists(dataset_dir):
        print(f"Erro: Pasta '{dataset_dir}' não encontrada.")
        sys.exit(1)
        
    # Inicializa EasyOCR
    has_gpu = torch.cuda.is_available()
    print(f"Inicializando EasyOCR no dispositivo: {'GPU (CUDA)' if has_gpu else 'CPU'}...")
    reader = easyocr.Reader(['pt', 'en'], gpu=has_gpu, verbose=False)
    
    # Coletar todas as imagens do repositório
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    all_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for ext in extensions:
            for filepath in glob.glob(os.path.join(root, ext)):
                all_files.append(filepath)
                
    all_files = sorted(all_files)
    print(f"Total de {len(all_files)} imagens encontradas para verificação.")
    print("Iniciando busca por anomalias (valores diferentes dos esperados)...\n")
    
    anomalias = []
    verificados = 0
    
    for idx, img_path in enumerate(all_files, 1):
        rel_path = os.path.relpath(img_path, dataset_dir)
        parts = rel_path.split(os.sep)
        amostra_folder = parts[0] if len(parts) > 1 else "Desconhecido"
        filename = os.path.basename(img_path)
        
        if amostra_folder not in EXPECTED_VALUES:
            continue
            
        verificados += 1
        
        # Carregar e processar imagem
        try:
            img = load_image(img_path)
            # Amostra 3 tem os metadados no topo, as outras no rodapé
            region = 'top' if amostra_folder == "Amostra 3" else 'bottom'
            cropped = crop_metadata_region(img, region=region, ratio=0.35)
            enhanced, thresh = enhance_text_region(cropped)
            
            # Executar OCR
            results = reader.readtext(cropped, detail=0)
            full_text = " ".join(results)
            
            # Extrair valores
            val_tuple = extract_values_from_text(full_text, amostra_folder)
            
            # Validar contra os padrões esperados
            targets = EXPECTED_VALUES[amostra_folder]
            matched = False
            for target in targets:
                # Compara se a tupla coincide com alguma das tuplas esperadas
                if val_tuple == target:
                    matched = True
                    break
                    
            if not matched:
                anomalias.append({
                    "Arquivo": filename,
                    "Caminho": rel_path,
                    "Amostra": amostra_folder,
                    "Valores_Encontrados": val_tuple,
                    "Valores_Esperados": targets,
                    "Texto_Completo": full_text
                })
                print(f"⚠️ ANOMALIA em {filename} ({amostra_folder}): Encontrado {val_tuple} | Esperado {targets}")
                
        except Exception as e:
            print(f"❌ Erro ao processar {filename}: {e}")
            
    print("\n=========================================================")
    print("  RELATÓRIO FINAL DE VERIFICAÇÃO DE ANOMALIAS            ")
    print("=========================================================")
    print(f"Total de imagens analisadas: {verificados}")
    print(f"Total de anomalias encontradas: {len(anomalias)}")
    
    if len(anomalias) > 0:
        print("\nLista de imagens com valores discrepantes:")
        for anom in anomalias:
            print(f"- [{anom['Amostra']}] {anom['Arquivo']} -> Encontrado: {anom['Valores_Encontrados']} | Esperado: {anom['Valores_Esperados']}")
    else:
        print("\n✅ Sucesso! Nenhuma anomalia de velocidade encontrada no dataset.")
        
if __name__ == "__main__":
    import glob
    main()
