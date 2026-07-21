import numpy as np
from datetime import datetime
from .preprocessor import preprocess
from .ocr import read_plate
from .postprocessor import validate
from .deduplicator import PlateDeduplicator

dedup = PlateDeduplicator(cooldown_seconds=30)

def process(frame_crop: np.ndarray) -> dict | None:
    """
    Entrada:  recorte numpy BGR da placa
    Saída:    dict com evento de leitura, ou None se descartado
    """
    img = preprocess(frame_crop)

    ocr_result = read_plate(img)
    if ocr_result is None:
        return None

    raw_text, conf = ocr_result

    validated = validate(raw_text)
    if validated is None:
        return None

    plate, pattern = validated

    if not dedup.should_emit(plate):
        return None  # duplicata dentro da janela de tempo

    return {
        'placa': plate,
        'confianca_ocr': round(conf, 4),
        'padrao': pattern,
        'timestamp': datetime.now().isoformat(),
    }