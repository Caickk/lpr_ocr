"""
Bloco A — Pré-processamento da imagem da placa antes do OCR.

Recebe o recorte (numpy array BGR) vindo do módulo de detecção (YOLO)
e prepara a imagem para maximizar a acurácia do PaddleOCR.
"""
import cv2
import numpy as np

TARGET_HEIGHT = 64  # px — altura mínima recomendada para OCR de texto curto


def resize_plate(img: np.ndarray, target_height: int = TARGET_HEIGHT) -> np.ndarray:
    """
    Redimensiona a imagem para a altura mínima recomendada, preservando
    o aspect ratio. Não reduz imagens que já estão acima do alvo.
    """
    h, w = img.shape[:2]
    if h >= target_height:
        return img

    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converte BGR para escala de cinza. Idempotente se já vier em cinza."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                 tile_grid_size: tuple[int, int] = (4, 4)) -> np.ndarray:
    """Normaliza contraste local — mais robusto que equalizeHist global."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binariza via threshold adaptativo — tolera iluminação desigual."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )


def preprocess(img: np.ndarray) -> np.ndarray:
    """Pipeline completo do Bloco A: resize -> cinza -> CLAHE -> threshold."""
    resized = resize_plate(img)
    gray = to_grayscale(resized)
    enhanced = apply_clahe(gray)
    return binarize(enhanced)
