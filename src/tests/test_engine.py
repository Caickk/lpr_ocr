from unittest.mock import MagicMock

import numpy as np
import pytest

from lpr_engine.deduplicator import PlateDeduplicator
from lpr_engine.engine import LPREngine


@pytest.fixture
def fake_crop() -> np.ndarray:
    """Simula um recorte de placa vindo do YOLO (BGR, pequeno)."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 255, size=(20, 80, 3), dtype=np.uint8)


def make_engine(ocr_read_return, dedup: PlateDeduplicator | None = None) -> LPREngine:
    """
    Monta a LPREngine com um PlateOCR mockado, via injeção de dependência
    -- não carrega o modelo real do PaddleOCR.
    """
    mock_ocr = MagicMock()
    mock_ocr.read.return_value = ocr_read_return
    return LPREngine(ocr=mock_ocr, dedup=dedup or PlateDeduplicator(cooldown_seconds=30))


class TestLPREngineProcess:
    def test_returns_none_when_ocr_finds_nothing(self, fake_crop):
        engine = make_engine(ocr_read_return=None)
        assert engine.process(fake_crop) is None

    def test_returns_none_when_ocr_text_fails_validation(self, fake_crop):
        engine = make_engine(ocr_read_return=("!!!!!!!", 0.9))
        assert engine.process(fake_crop) is None

    def test_returns_event_dict_for_valid_mercosul_plate(self, fake_crop):
        engine = make_engine(ocr_read_return=("ABC1D23", 0.91))
        event = engine.process(fake_crop)

        assert event is not None
        assert event["placa"] == "ABC1D23"
        assert event["padrao"] == "mercosul"
        assert event["confianca_ocr"] == 0.91
        assert "timestamp" in event

    def test_returns_event_dict_for_valid_old_pattern_plate(self, fake_crop):
        engine = make_engine(ocr_read_return=("ABC1234", 0.88))
        event = engine.process(fake_crop)

        assert event is not None
        assert event["padrao"] == "antigo"

    @pytest.mark.parametrize("score,expected_label", [
        (0.90, "alta"),
        (0.75, "media"),
        (0.55, "baixa"),
    ])
    def test_quality_label_matches_score_range(self, fake_crop, score, expected_label):
        engine = make_engine(ocr_read_return=("ABC1D23", score))
        event = engine.process(fake_crop)
        assert event["qualidade"] == expected_label

    def test_second_reading_of_same_plate_within_cooldown_is_suppressed(self, fake_crop):
        dedup = PlateDeduplicator(cooldown_seconds=30)
        engine = make_engine(ocr_read_return=("ABC1D23", 0.91), dedup=dedup)

        first = engine.process(fake_crop)
        second = engine.process(fake_crop)

        assert first is not None
        assert second is None

    def test_different_plates_are_not_deduplicated_against_each_other(self, fake_crop):
        dedup = PlateDeduplicator(cooldown_seconds=30)

        engine_a = make_engine(ocr_read_return=("ABC1D23", 0.91), dedup=dedup)
        engine_b = make_engine(ocr_read_return=("XYZ9K87", 0.91), dedup=dedup)

        assert engine_a.process(fake_crop) is not None
        assert engine_b.process(fake_crop) is not None
