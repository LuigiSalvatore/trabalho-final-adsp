"""
Módulo de Processamento Digital de Imagens (PDI/PDS) para detecção e realce de metadados.
"""
import cv2
import numpy as np

def load_image(image_path):
    """Carrega uma imagem a partir do caminho especificado."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")
    return img

def crop_metadata_region(img, region='bottom', ratio=0.25):
    """
    Recorta a faixa de metadados da imagem.
    region: 'bottom', 'top', ou 'full'
    ratio: proporção da altura a recortar (ex: 0.25 = 25% inferiores/superiores)
    """
    h, w = img.shape[:2]
    if region == 'bottom':
        return img[int(h * (1 - ratio)):h, 0:w]
    elif region == 'top':
        return img[0:int(h * ratio), 0:w]
    return img

def enhance_text_region(img):
    """
    Aplica técnicas de PDI para realçar caracteres em faixas de metadados:
    - Conversão para escala de cinza
    - Deteção e ajuste de contraste (CLAHE)
    - Binarização adaptativa
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Binarização adaptativa (Otsu e Threshold adaptativo)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return enhanced, thresh
