"""
Bloco B — Leitura de texto via PaddleOCR, configurado para o contexto
do projeto (recortes de placa vindos do YOLO, câmeras fixas, hardware
a confirmar).
"""
import os

import numpy as np
from paddleocr import PaddleOCR


class PlateOCR:
    def __init__(self, min_confidence: float = 0.6):
        self._min_conf = min_confidence

        use_mkldnn = os.getenv("LPR_ENV", "dev") == "production"

        self._engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=True,
            use_textline_orientation=False,
            enable_mkldnn=use_mkldnn,
            device="cpu",
        )

    def read(self, img: np.ndarray) -> tuple[str, float] | None:
        """
        Retorna (texto_bruto, confiança) ou None se não houver leitura
        ou se a confiança estiver abaixo do limiar configurado.
        """
        result = self._engine.ocr(img)

        if not result or not result[0]:
            return None

        candidates = result[0]
        best = max(candidates, key=lambda c: c[1])
        text, score = best

        if score < self._min_conf:
            return None

        return text, round(float(score), 4)
