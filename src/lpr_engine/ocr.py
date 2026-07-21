'''
Pode ser necessário desativar o MKLDNN no Colab para evitar erros de segmentação.
import os
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
'''

from paddleocr import PaddleOCR
import numpy as np
class PlateOCR:
    def __init__(self, min_confidence: float = 0.6):
        self._min_conf = min_confidence

        #use_mkldnn = os.getenv("LPR_ENV", "dev") == "production"

        self._engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",            # Detecção: mantida ligada para tolerar recortes imperfeitos do YOLO
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",    # ou PP-OCRv6 quando estável no seu ambiente
            use_doc_orientation_classify=False,
            use_doc_unwarping=True,                                     # Retificação de perspectiva: substitui deskew manual via OpenCV,
            use_textline_orientation=False,                             # Orientação de linha (rotação 180°): desativado.
            enable_mkldnn=False,                                        # Aceleração Intel/AMD: só ativa em produção (máquina física),
            device="cpu",                                               # Hardware ainda não confirmado — CPU como baseline seguro e reprodutível em qualquer máquina
        )
    
    def read(self, img: np.ndarray) -> tuple[str, float] | None:
        """Retorna (texto_bruto, confiança_média) ou None se abaixo do limiar."""
        result = self._engine.ocr(img)

        if not result or not result[0]:
            return None

        candidates = result[0]
        best = max(candidates, key=lambda x: x[1])
        text, score = best

        if score < self._min_conf:
            return None

        return text, round(float(score), 4)