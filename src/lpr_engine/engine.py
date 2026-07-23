"""
Orquestrador do pipeline: recorte da placa -> pré-processamento -> OCR
-> pós-processamento/validação -> deduplicação -> evento de leitura.
"""
from datetime import datetime

import numpy as np

from .deduplicator import PlateDeduplicator
from .ocr import PlateOCR
from .postprocessor import validate
from .preprocessor import preprocess

MIN_SCORE_CLEAN = 0.85
MIN_SCORE_SOFT = 0.70


def _quality_label(score: float) -> str:
    if score >= MIN_SCORE_CLEAN:
        return 'alta'
    if score >= MIN_SCORE_SOFT:
        return 'media'
    return 'baixa'


class LPREngine:
    def __init__(self, ocr: PlateOCR | None = None,
                 dedup: PlateDeduplicator | None = None):
        # Permite injeção de dependências nos testes, evitando carregar
        # o modelo real do PaddleOCR.
        self._ocr = ocr or PlateOCR()
        self._dedup = dedup or PlateDeduplicator(cooldown_seconds=30)

    def process(self, frame_crop: np.ndarray) -> dict | None:
        img = preprocess(frame_crop)

        ocr_result = self._ocr.read(img)
        if ocr_result is None:
            return None

        raw_text, conf = ocr_result

        validated = validate(raw_text)
        if validated is None:
            return None

        plate, pattern = validated

        if not self._dedup.should_emit(plate):
            return None

        return {
            'placa': plate,
            'confianca_ocr': conf,
            'qualidade': _quality_label(conf),
            'padrao': pattern,
            'timestamp': datetime.now().isoformat(),
        }
