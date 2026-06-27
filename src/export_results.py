"""
Módulo para processamento em lote de imagens com suporte a pause/resume e barra de progresso com ETA/ETC.
"""
import os
import sys
import glob
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from src.speed_extractor import SpeedExtractor

STATE_FILE = "process_state.json"
OUTPUT_CSV = "resultados_extracao_velocidade.csv"

def save_state(results, pending_files, max_total_samples):
    """Salva o progresso atual em um arquivo JSON."""
    state = {
        "max_total_samples": max_total_samples,
        "results": results,
        "pending_files": pending_files
    }
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_state():
    """Carrega o progresso anterior se existir."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Aviso: Erro ao carregar arquivo de estado: {e}")
    return None

def clear_state():
    """Remove o arquivo de estado temporário."""
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception as e:
            print(f"Erro ao remover arquivo de estado: {e}")

def process_directory(base_dir, max_total_samples=30, reset=False):
    """
    Varre os diretórios de amostras e extrai as velocidades com suporte a pause/resume.
    """
    if reset:
        clear_state()
        
    state = load_state()
    extractor = SpeedExtractor(use_easyocr=True)
    print(f"Dispositivo de processamento OCR detectado: {extractor.hardware_device}")
    
    results = []
    pending_files = []
    total_to_process = 0
    
    if state is not None:
        print("\n=== [RESUME] Estado anterior detectado! Retomando progresso... ===")
        results = state.get("results", [])
        pending_files = state.get("pending_files", [])
        total_to_process = len(results) + len(pending_files)
        print(f"Progresso: {len(results)} concluídos, {len(pending_files)} restantes de um total de {total_to_process}.")
    else:
        # Fazer varredura inicial se for uma nova execução
        extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
        all_image_files = []
        for root, dirs, files in os.walk(base_dir):
            for ext in extensions:
                for filepath in glob.glob(os.path.join(root, ext)):
                    all_image_files.append(filepath)
                    
        all_image_files = sorted(all_image_files)
        
        if max_total_samples is not None:
            step = max(1, len(all_image_files) // max_total_samples) if all_image_files else 1
            selected_files = all_image_files[::step][:max_total_samples]
        else:
            selected_files = all_image_files
            
        pending_files = selected_files
        total_to_process = len(selected_files)
        print(f"Total de {len(all_image_files)} imagens encontradas. Iniciando processamento de {total_to_process} imagens...")
        save_state(results, pending_files, max_total_samples)

    if not pending_files:
        print("Tudo já foi processado anteriormente.")
        df = pd.DataFrame(results)
        return df

    start_time = time.time()
    processed_in_this_run = 0
    total_completed = len(results)
    
    # Listas auxiliares para calcular médias móveis de tempo (s/it)
    durations = []
    
    try:
        while pending_files:
            img_path = pending_files[0]
            rel_path = os.path.relpath(img_path, base_dir)
            parts = rel_path.split(os.sep)
            amostra_folder = parts[0] if len(parts) > 1 else "Raiz"
            filename = os.path.basename(img_path)
            
            img_start = time.time()
            
            # Executar extração
            try:
                speed, raw_text, method = extractor.process_image(img_path)
            except Exception as e:
                speed, raw_text, method = None, f"Erro: {str(e)}", "Falha"
                
            img_duration = time.time() - img_start
            durations.append(img_duration)
            
            # Registrar resultado
            results.append({
                "Caminho_Relativo": rel_path,
                "Pasta_Amostra": amostra_folder,
                "Arquivo": filename,
                "Velocidade_Extraida_kmh": speed,
                "Metodo_OCR": method,
                "Texto_Reconhecido": raw_text
            })
            
            # Remover da fila
            pending_files.pop(0)
            processed_in_this_run += 1
            total_completed = len(results)
            
            # Salvar estado atual no disco (salva após cada arquivo para não perder se fechar o terminal)
            save_state(results, pending_files, max_total_samples)
            
            # Exportar CSV incremental
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            
            # --- CÁLCULO DE PROTÓTIPO DE PROGRESSO, ETA E ETC ---
            elapsed = time.time() - start_time
            avg_duration = sum(durations) / len(durations)
            remaining_count = len(pending_files)
            eta_seconds = remaining_count * avg_duration
            
            # Formatação do ETA (Tempo Restante)
            eta_str = str(timedelta(seconds=int(eta_seconds)))
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            
            # Formatação do ETC (Horário de Conclusão)
            etc_time = datetime.now() + timedelta(seconds=eta_seconds)
            etc_str = etc_time.strftime("%H:%M:%S")
            
            # Barra de progresso visual
            percent = (total_completed / total_to_process) * 100
            bar_length = 20
            filled_length = int(bar_length * total_completed // total_to_process)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Imprimir estatísticas formatadas de forma fixa na tela
            sys.stdout.write(f"\r\033[KProcessando: {filename} ({amostra_folder})\n")
            sys.stdout.write(
                f"\r\033[K[{bar}] {percent:.1f}% | {total_completed}/{total_to_process} | "
                f"Tempo: {elapsed_str} | ETA: {eta_str} | Conclusão esperada: {etc_str}"
            )
            # Retornar o cursor uma linha acima para manter o efeito fixo
            sys.stdout.write("\033[A")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        sys.stdout.write("\n\n\033[KProcessamento pausado pelo usuário (Ctrl+C).\n")
        sys.stdout.write("O progresso foi salvo. Execute novamente para continuar de onde parou.\n")
        sys.exit(0)
        
    # Limpar cursor e linha no final da execução completa
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    clear_state()
    
    df = pd.DataFrame(results)
    return df

def save_results(df, output_csv=OUTPUT_CSV):
    """Salva os resultados finais em formato CSV estruturado."""
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Resultados consolidados salvos com sucesso em: {output_csv}")
