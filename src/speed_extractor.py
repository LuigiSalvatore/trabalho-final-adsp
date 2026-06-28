"""
Módulo de Extração de Velocidade utilizando OCR e Expressões Regulares (Regex).
"""
import re
import cv2
from src.image_processing import load_image, crop_metadata_region, enhance_text_region

class SpeedExtractor:
    def __init__(self, use_easyocr=True):
        self.use_easyocr = use_easyocr
        self.reader = None
        if use_easyocr:
            try:
                import easyocr
                import torch
                has_gpu = torch.cuda.is_available()
                device_str = "GPU (CUDA)" if has_gpu else "CPU"
                # Inicializa EasyOCR para Português e Inglês com detecção dinâmica de hardware
                self.reader = easyocr.Reader(['pt', 'en'], gpu=has_gpu, verbose=False)
                self.hardware_device = device_str
            except Exception as e:
                print(f"Aviso: EasyOCR nao pode ser carregado: {e}")
                self.use_easyocr = False
                self.hardware_device = "Nenhum"

        # Padrões Regex para capturar velocidade medida
        self.patterns = [
            # Ex: "Vel. med: 99km/h", "Vel. med.: 99.5 km/h", "VEL.MED: 40 km/h"
            re.compile(r'vel[^\d]*med[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Ex: "Vel. real: 99.12km/h"
            re.compile(r'vel[^\d]*real[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Ex: "Vel. cons.: 92km/h"
            re.compile(r'vel[^\d]*cons[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Ex: "VEL.MED 40km/h" ou "VELOCIDADE 60 km/h"
            re.compile(r'velocidade[^\d]*[:\.\s]+(\d+[\.,]?\d*)', re.IGNORECASE),
            # Genérico: "(\d+) km/h" ou "(\d+)km/h"
            re.compile(r'(\d+[\.,]?\d*)\s*km/?h', re.IGNORECASE),
        ]

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
        """
        img = load_image(image_path)
        
        # Testar regiões (rodapé primeiro, depois topo, depois imagem inteira)
        regions_to_test = ['bottom', 'top', 'full']
        
        for region in regions_to_test:
            cropped = crop_metadata_region(img, region=region, ratio=0.3)
            # Redimensiona por 2x com interpolação cúbica para aumentar fontes pequenas e melhorar o OCR
            cropped = cv2.resize(cropped, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            enhanced, thresh = enhance_text_region(cropped)
            
            # Executar EasyOCR se disponível
            if self.reader:
                # Testar imagem original recortada e imagem realçada
                for target_img in [cropped, enhanced]:
                    results = self.reader.readtext(target_img, detail=0)
                    full_text = " ".join(results)
                    speed = self.extract_from_text(full_text)
                    if speed is not None:
                        return speed, full_text, f"EasyOCR ({region})"
                        
        return None, "", "Nenhum resultado"
