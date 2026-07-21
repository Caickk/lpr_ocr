'''
Pode ser necessário desativar o MKLDNN no Colab para evitar erros de segmentação.
import os
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
'''
from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(
    text_recognition_model_name='PP-OCRv5_mobile_rec',  # ou PP-OCRv6 quando estável no seu ambiente
    use_angle_cls=True,
    lang='en',
    det=False,       # você já entrega o recorte — não precisa do modelo de detecção
    use_gpu=False,   # muda para True se confirmar GPU disponível depois
    enable_mkldnn=True, #Quando true, pode dar problema no Colab.
    show_log=False,
    device = 'cpu' 
)

def read_plate(img: np.ndarray) -> tuple[str, float] | None:
    """Retorna (texto_bruto, confiança_média) ou None se abaixo do limiar."""
    result = ocr.ocr(img, cls=True)

    if not result or not result[0]:
        return None

    texts, scores = [], []
    for line in result[0]:
        text, score = line[1]
        texts.append(text)
        scores.append(score)

    raw_text = ' '.join(texts)
    avg_conf = sum(scores) / len(scores)

    # B2 — Descarta leituras com confiança baixa
    if avg_conf < 0.6:
        return None

    return raw_text, avg_conf