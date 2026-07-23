from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lpr_engine.ocr import PlateOCR


@pytest.fixture
def fake_image() -> np.ndarray:
    return np.zeros((64, 200), dtype=np.uint8)


def make_plate_ocr_with_mocked_engine(ocr_return_value):
    """
    Cria um PlateOCR com o PaddleOCR real substituído por um mock,
    para testar a lógica de filtragem sem carregar modelos.
    """
    with patch("lpr_engine.ocr.PaddleOCR") as mock_paddle_cls:
        mock_engine = MagicMock()
        mock_engine.ocr.return_value = ocr_return_value
        mock_paddle_cls.return_value = mock_engine
        plate_ocr = PlateOCR(min_confidence=0.6)
    return plate_ocr


class TestPlateOCRRead:
    def test_returns_none_when_result_is_none(self, fake_image):
        ocr = make_plate_ocr_with_mocked_engine(None)
        assert ocr.read(fake_image) is None

    def test_returns_none_when_result_is_empty_list(self, fake_image):
        ocr = make_plate_ocr_with_mocked_engine([])
        assert ocr.read(fake_image) is None

    def test_returns_none_when_first_region_is_empty(self, fake_image):
        ocr = make_plate_ocr_with_mocked_engine([[]])
        assert ocr.read(fake_image) is None

    def test_returns_text_and_score_above_threshold(self, fake_image):
        ocr = make_plate_ocr_with_mocked_engine([[("ABC1D23", 0.94)]])
        result = ocr.read(fake_image)
        assert result == ("ABC1D23", 0.94)

    def test_returns_none_when_score_below_threshold(self, fake_image):
        ocr = make_plate_ocr_with_mocked_engine([[("ABC1D23", 0.42)]])
        assert ocr.read(fake_image) is None

    def test_picks_highest_confidence_among_multiple_candidates(self, fake_image):
        candidates = [[
            ("ABC1D2E", 0.55),
            ("ABC1D23", 0.91),
            ("XYZ9999", 0.30),
        ]]
        ocr = make_plate_ocr_with_mocked_engine(candidates)
        result = ocr.read(fake_image)
        assert result == ("ABC1D23", 0.91)

    def test_score_is_rounded_to_four_decimals(self, fake_image):
        # score acima do min_confidence (0.6) — o objetivo aqui é
        # validar o arredondamento, não o filtro de confiança
        ocr = make_plate_ocr_with_mocked_engine([[("ABC1D23", 0.912345678)]])
        _, score = ocr.read(fake_image)
        assert score == 0.9123

    def test_custom_min_confidence_is_respected(self, fake_image):
        with patch("lpr_engine.ocr.PaddleOCR") as mock_paddle_cls:
            mock_engine = MagicMock()
            mock_engine.ocr.return_value = [[("ABC1D23", 0.75)]]
            mock_paddle_cls.return_value = mock_engine
            ocr = PlateOCR(min_confidence=0.8)

        assert ocr.read(fake_image) is None
