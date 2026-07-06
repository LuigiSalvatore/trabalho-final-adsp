"""
Módulo de Extração de Velocidade utilizando OCR e Expressões Regulares (Regex).
Otimizado para uso em ambiente multiprocessing (lazy init do EasyOCR).
"""
import re
import cv2
from src.image_processing import load_image, crop_metadata_region, enhance_text_region


class SpeedExtractor:
    def __init__(self, use_easyocr=True):
        self.use_easyocr = use_easyocr
        self.reader = None
        self.hardware_device = "Nenhum"
        # A inicialização pesada do EasyOCR é feita de forma lazy (sob demanda)
        # para compatibilidade com multiprocessing no Windows.

        # Padrões Regex para capturar velocidade medida (ordem de especificidade)
        self.patterns = [
            # Ex: "Vel. med: 99km/h", "Vel. med.: 99.5 km/h", "VEL.MED: 40 km/h"
            re.compile(r'vel[^\d]*med[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Ex: "Vel. real: 99.12km/h"
            re.compile(r'vel[^\d]*real[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Ex: "Vel. cons.: 92km/h"
            re.compile(r'vel[^\d]*cons[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Ex: "Velocidade 60 km/h"
            re.compile(r'velocidade[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Genérico: "(\\d+) km/h" ou "(\\d+)km/h"
            re.compile(r'(\d+[\.,]?\d*)\s*km/?h', re.IGNORECASE),
        ]

    def _ensure_reader(self):
        """
        Inicializa o EasyOCR de forma lazy. Chamado apenas quando necessário,
        dentro do processo worker. Isso evita problemas de fork/spawn do multiprocessing.
        """
        if self.reader is not None:
            return

        if not self.use_easyocr:
            return

        try:
            import easyocr
            # Forçar uso de CPU para evitar conflitos de inicialização de CUDA
            # no multiprocessing do Windows e economizar VRAM/Recursos.
            self.hardware_device = "CPU (Forçado)"
            self.reader = easyocr.Reader(['pt', 'en'], gpu=False, verbose=False)
        except Exception as e:
            print(f"Aviso: EasyOCR nao pode ser carregado: {e}")
            self.use_easyocr = False
            self.hardware_device = "Nenhum"

    def get_hardware_device(self):
        """Retorna o dispositivo de hardware ativo (inicializa se necessário)."""
        self._ensure_reader()
        return self.hardware_device

    def extract_from_text(self, text):
        """Aplica os padrões de regex no texto extraído via OCR."""
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                val_str = match.group(1).replace(',', '.')
                try:
                    val = float(val_str)
                    return val
                except ValueError:
                    continue
        return None

    def process_image(self, image_path):
        """
        Carrega a imagem, recorta faixas de metadados, roda OCR e extrai a velocidade.
        Retorna (velocidade_extraida, texto_ocr_completo, metodo)

        Estratégia de tentativas (short-circuit ao primeiro sucesso):
          1. bottom + cropped
          2. bottom + enhanced
          3. top + cropped
          4. top + enhanced
        A região 'full' foi removida pois os metadados estão sempre no topo ou rodapé.
        """
        self._ensure_reader()

        img = load_image(image_path)

        # Regiões a testar: bottom e top apenas (padrão das amostras)
        regions_to_test = ['bottom', 'top']

        for region in regions_to_test:
            cropped = crop_metadata_region(img, region=region, ratio=0.3)
            # Upscaling 2x com interpolação cúbica para aumentar fontes pequenas
            cropped = cv2.resize(cropped, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            enhanced, _ = enhance_text_region(cropped)

            if self.reader:
                # Short-circuit: testa versão original primeiro, depois realçada
                for target_img in [cropped, enhanced]:
                    results = self.reader.readtext(target_img, detail=0)
                    full_text = " ".join(results)
                    speed = self.extract_from_text(full_text)
                    if speed is not None:
                        return speed, full_text, f"EasyOCR ({region})"

        return None, "", "Nenhum resultado"
