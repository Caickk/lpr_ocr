import cv2
import numpy as np

def preprocess(img: np.ndarray) -> np.ndarray:
    # A1 — Redimensionamento: garante altura mínima mantendo proporção
    height, width = img.shape[:2]
    if height < 60:
        scale = 60 / height
        img = cv2.resize(img, (int(width * scale), 60), interpolation=cv2.INTER_CUBIC)

    # A2 — Deskew: corrige perspectiva se a placa vier inclinada
    # (no MVP pode pular — só adicionar se o OCR errar muito em placas anguladas)

    # A3 — Escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A4 — CLAHE: normaliza contraste local (melhor que equalizeHist global)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # A5 — Threshold adaptativo: binariza mesmo com iluminação desigual
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    return binary