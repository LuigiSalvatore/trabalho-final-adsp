"""
Módulo para processamento em lote de imagens com suporte a pause/resume,
barra de progresso com ETA/ETC e processamento paralelo via multiprocessing.
"""
import os
import sys
import glob
import json
import time
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

STATE_FILE = "process_state.json"
OUTPUT_CSV = "resultados_extracao_velocidade.csv"

# ---------------------------------------------------------------------------
# Funções de estado (salvar / carregar / limpar)
# ---------------------------------------------------------------------------

def save_state(results, pending_files, max_total_samples):
    """Salva o progresso atual em um arquivo JSON."""
    state = {
        "max_total_samples": max_total_samples,
        "results": results,
        "pending_files": list(pending_files),
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


# ---------------------------------------------------------------------------
# Worker — executado em cada processo filho
# ---------------------------------------------------------------------------

_worker_extractor = None

def _init_worker():
    """
    Inicializa os recursos do worker uma única vez quando o processo filho é criado.
    Carrega o modelo EasyOCR na memória para ser reaproveitado em todas as tarefas subsequentes.
    """
    global _worker_extractor
    import warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Configurações para evitar thrashing de CPU e uso excessivo de recursos
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass

    # Import local e instanciação única do extrator (carrega pesos do EasyOCR do disco apenas 1 vez)
    from src.speed_extractor import SpeedExtractor
    _worker_extractor = SpeedExtractor(use_easyocr=True)


def _worker_process_image(args):
    """
    Função de topo de nível executada pelo pool de workers.
    Recebe (img_path, base_dir) e retorna um dicionário com o resultado.
    """
    global _worker_extractor
    img_path, base_dir = args

    # Fallback caso a inicialização por algum motivo não tenha rodado
    if _worker_extractor is None:
        _init_worker()

    rel_path = os.path.relpath(img_path, base_dir)
    parts = rel_path.split(os.sep)
    amostra_folder = parts[0] if len(parts) > 1 else "Raiz"
    filename = os.path.basename(img_path)

    try:
        speed, raw_text, method = _worker_extractor.process_image(img_path)
    except Exception as e:
        speed, raw_text, method = None, f"Erro: {str(e)}", "Falha"

    return {
        "Caminho_Relativo": rel_path,
        "Pasta_Amostra": amostra_folder,
        "Arquivo": filename,
        "Velocidade_Extraida_kmh": speed,
        "Metodo_OCR": method,
        "Texto_Reconhecido": raw_text,
    }


# ---------------------------------------------------------------------------
# Barra de progresso
# ---------------------------------------------------------------------------

def _print_progress(filename, amostra_folder, total_completed, total_to_process,
                    elapsed, avg_duration, n_workers):
    """Imprime a barra de progresso com ETA e ETC no terminal."""
    remaining_count = total_to_process - total_completed
    eta_seconds = remaining_count * avg_duration / max(n_workers, 1)

    eta_str = str(timedelta(seconds=int(eta_seconds)))
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    etc_time = datetime.now() + timedelta(seconds=eta_seconds)
    etc_str = etc_time.strftime("%H:%M:%S")

    percent = (total_completed / total_to_process) * 100
    bar_length = 25
    filled_length = int(bar_length * total_completed // total_to_process)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    sys.stdout.write(f"\r\033[KProcessando: {filename} ({amostra_folder})\n")
    sys.stdout.write(
        f"\r\033[K[{bar}] {percent:.1f}% | {total_completed}/{total_to_process} | "
        f"Workers: {n_workers} | Tempo: {elapsed_str} | ETA: {eta_str} | Conclusão: {etc_str}"
    )
    sys.stdout.write("\033[A")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Função principal de processamento
# ---------------------------------------------------------------------------

def process_directory(base_dir, max_total_samples=30, reset=False, n_workers=None):
    """
    Varre os diretórios de amostras e extrai as velocidades com suporte a pause/resume
    e processamento paralelo via ProcessPoolExecutor.

    Args:
        base_dir: Diretório raiz das amostras.
        max_total_samples: Número máximo de imagens a processar (None = todas).
        reset: Se True, ignora estado salvo e reinicia do zero.
        n_workers: Número de processos paralelos. Default: os.cpu_count() - 2 (mínimo 1).
    """
    if reset:
        clear_state()

    # Determinar número de workers
    if n_workers is None:
        cpu_count = os.cpu_count() or 2
        n_workers = max(1, cpu_count - 2)

    state = load_state()
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
        # Varredura inicial (evita duplicatas causadas pelo glob case-insensitive no Windows)
        all_image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    all_image_files.append(os.path.join(root, file))

        all_image_files = sorted(list(set(all_image_files)))

        if max_total_samples is not None:
            step = max(1, len(all_image_files) // max_total_samples) if all_image_files else 1
            selected_files = all_image_files[::step][:max_total_samples]
        else:
            selected_files = all_image_files

        pending_files = selected_files
        total_to_process = len(selected_files)
        print(
            f"Total de {len(all_image_files)} imagens encontradas. "
            f"Iniciando processamento de {total_to_process} imagens com {n_workers} workers paralelos..."
        )
        save_state(results, pending_files, max_total_samples)

    if not pending_files:
        print("Tudo já foi processado anteriormente.")
        return pd.DataFrame(results)

    # Intervalo de salvamento: a cada ~5% do total
    save_interval = max(1, total_to_process // 20)
    start_time = time.time()
    processed_in_this_run = 0
    # Janela deslizante de durações para cálculo de ETA mais preciso
    durations = deque(maxlen=50)

    # Preparar argumentos para os workers
    worker_args = [(img_path, base_dir) for img_path in pending_files]
    results_in_order = list(results)  # Cópia para acumular resultados

    print("\nInicializando workers paralelos e carregando modelos OCR na memória...")
    print("Isso pode levar de 30 a 90 segundos na primeira execução. Por favor, aguarde...\n")
    sys.stdout.flush()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_process_image, arg): arg for arg in worker_args}

            for future in as_completed(futures):
                img_path, _ = futures[future]
                filename = os.path.basename(img_path)
                rel_path = os.path.relpath(img_path, base_dir)
                parts = rel_path.split(os.sep)
                amostra_folder = parts[0] if len(parts) > 1 else "Raiz"

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "Caminho_Relativo": rel_path,
                        "Pasta_Amostra": amostra_folder,
                        "Arquivo": filename,
                        "Velocidade_Extraida_kmh": None,
                        "Metodo_OCR": "Falha",
                        "Texto_Reconhecido": f"Erro no worker: {str(e)}",
                    }

                results_in_order.append(result)
                processed_in_this_run += 1

                # Calcular tempo médio por imagem com janela deslizante
                elapsed = time.time() - start_time
                # Tempo médio real: elapsed / processed * n_workers ≈ tempo por imagem
                avg_duration = elapsed / processed_in_this_run

                durations.append(avg_duration)
                smooth_avg = sum(durations) / len(durations)

                total_completed = len(results_in_order)

                # Atualizar display de progresso
                _print_progress(
                    filename, amostra_folder,
                    total_completed, total_to_process,
                    elapsed, smooth_avg, n_workers
                )

                # Salvar estado e CSV a cada save_interval imagens
                remaining_pending = [
                    arg[0] for fut, arg in futures.items() if not fut.done()
                ]
                if (processed_in_this_run % save_interval == 0) or (processed_in_this_run == len(worker_args)):
                    save_state(results_in_order, remaining_pending, max_total_samples)
                    df_temp = pd.DataFrame(results_in_order)
                    df_temp.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    except KeyboardInterrupt:
        # Salvar o que já foi processado
        remaining_pending = [
            arg[0] for fut, arg in futures.items() if not fut.done()
        ]
        save_state(results_in_order, remaining_pending, max_total_samples)
        df_temp = pd.DataFrame(results_in_order)
        df_temp.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        sys.stdout.write("\n\n\033[KProcessamento pausado pelo usuário (Ctrl+C).\n")
        sys.stdout.write("O progresso foi salvo. Execute novamente para continuar de onde parou.\n")
        sys.exit(0)

    # Limpar cursor e linha no final da execução completa
    sys.stdout.write("\n\n")
    sys.stdout.flush()
    clear_state()

    df = pd.DataFrame(results_in_order)
    return df


# ---------------------------------------------------------------------------

def save_results(df, output_csv=OUTPUT_CSV):
    """Salva os resultados finais em formato CSV estruturado."""
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Resultados consolidados salvos com sucesso em: {output_csv}")
