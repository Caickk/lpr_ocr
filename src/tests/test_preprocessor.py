import numpy as np
import pytest

from lpr_engine.preprocessor import (
    TARGET_HEIGHT,
    apply_clahe,
    binarize,
    preprocess,
    resize_plate,
    to_grayscale,
)


def make_bgr_image(height: int, width: int) -> np.ndarray:
    """Gera uma imagem BGR sintética com algum padrão (não uniforme),
    para que CLAHE/threshold tenham o que processar."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


class TestResizePlate:
    def test_upscales_image_below_target_height(self):
        img = make_bgr_image(20, 80)
        result = resize_plate(img)
        assert result.shape[0] == TARGET_HEIGHT

    def test_preserves_aspect_ratio_when_upscaling(self):
        img = make_bgr_image(20, 80)  # aspect ratio 4:1
        result = resize_plate(img)
        expected_width = int(80 * (TARGET_HEIGHT / 20))
        assert result.shape[1] == expected_width

    def test_does_not_resize_image_already_at_target(self):
        img = make_bgr_image(TARGET_HEIGHT, 200)
        result = resize_plate(img)
        assert result.shape == img.shape

    def test_does_not_resize_image_above_target(self):
        img = make_bgr_image(120, 400)
        result = resize_plate(img)
        assert result.shape == img.shape

    def test_respects_custom_target_height(self):
        img = make_bgr_image(10, 40)
        result = resize_plate(img, target_height=32)
        assert result.shape[0] == 32


class TestToGrayscale:
    def test_converts_bgr_to_single_channel(self):
        img = make_bgr_image(64, 200)
        gray = to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (64, 200)

    def test_is_idempotent_on_already_gray_image(self):
        img = make_bgr_image(64, 200)
        gray_once = to_grayscale(img)
        gray_twice = to_grayscale(gray_once)
        assert np.array_equal(gray_once, gray_twice)


class TestApplyClahe:
    def test_output_has_same_shape_as_input(self):
        img = make_bgr_image(64, 200)
        gray = to_grayscale(img)
        enhanced = apply_clahe(gray)
        assert enhanced.shape == gray.shape

    def test_output_dtype_is_uint8(self):
        img = make_bgr_image(64, 200)
        gray = to_grayscale(img)
        enhanced = apply_clahe(gray)
        assert enhanced.dtype == np.uint8


class TestBinarize:
    def test_output_only_contains_binary_values(self):
        img = make_bgr_image(64, 200)
        gray = to_grayscale(img)
        binary = binarize(gray)
        unique_values = set(np.unique(binary).tolist())
        assert unique_values.issubset({0, 255})

    def test_output_shape_matches_input(self):
        img = make_bgr_image(64, 200)
        gray = to_grayscale(img)
        binary = binarize(gray)
        assert binary.shape == gray.shape


class TestPreprocessPipeline:
    def test_full_pipeline_returns_binary_single_channel_image(self):
        img = make_bgr_image(20, 80)
        result = preprocess(img)
        assert result.ndim == 2
        assert result.shape[0] == TARGET_HEIGHT
        unique_values = set(np.unique(result).tolist())
        assert unique_values.issubset({0, 255})

    def test_pipeline_handles_already_sized_image(self):
        img = make_bgr_image(100, 300)
        result = preprocess(img)
        assert result.shape[0] == 100  # não deve ter sido redimensionada

    @pytest.mark.parametrize("height,width", [(1, 5), (5, 1), (64, 64)])
    def test_pipeline_does_not_crash_on_edge_shapes(self, height, width):
        img = make_bgr_image(height, width)
        result = preprocess(img)
        assert result is not None
