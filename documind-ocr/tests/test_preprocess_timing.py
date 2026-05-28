"""Tests for preprocessing pipeline timing and overhead reduction claims."""

import time
import numpy as np
import pytest

from app.pipelines.preprocess import (
    grayscale,
    denoise,
    clahe_enhance,
    threshold_adaptive,
    morphological_clean,
    correct_skew,
    preprocess_pipeline,
)


def _make_bgr_image(h=200, w=300) -> np.ndarray:
    """Create a synthetic BGR document image."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 240
    # Add horizontal black bars (simulated text lines)
    for y in range(20, h - 20, 25):
        img[y : y + 8, 20 : w - 20] = 30
    return img


class TestPreprocessSteps:
    def test_grayscale_shape(self):
        img = _make_bgr_image()
        gray = grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (200, 300)

    def test_clahe_shape(self):
        gray = grayscale(_make_bgr_image())
        enhanced = clahe_enhance(gray)
        assert enhanced.shape == gray.shape
        assert enhanced.dtype == np.uint8

    def test_threshold_binary(self):
        gray = grayscale(_make_bgr_image())
        thresh = threshold_adaptive(gray)
        unique_vals = set(np.unique(thresh))
        assert unique_vals.issubset({0, 255}), "Threshold output should be binary"

    def test_morphological_clean_shape(self):
        gray = grayscale(_make_bgr_image())
        thresh = threshold_adaptive(gray)
        cleaned = morphological_clean(thresh)
        assert cleaned.shape == thresh.shape

    def test_correct_skew_returns_image(self):
        img = _make_bgr_image()
        result = correct_skew(img)
        assert result.shape == img.shape


class TestPreprocessPipeline:
    def test_output_type_and_shape(self):
        img = _make_bgr_image()
        processed, debug = preprocess_pipeline(img)
        assert isinstance(processed, np.ndarray)
        assert processed.shape[2] == 3, "Should return BGR colour image"

    def test_debug_info_keys(self):
        img = _make_bgr_image()
        _, debug = preprocess_pipeline(img)
        assert "total_preprocessing_ms" in debug
        assert "overhead_reduction_pct" in debug
        assert "steps" in debug

    def test_overhead_reduction_gte_85pct(self):
        """
        The automated pipeline must achieve ≥ 85% overhead reduction vs.
        the manual 30-second baseline per document.
        """
        img = _make_bgr_image()
        _, debug = preprocess_pipeline(img)
        reduction = debug["overhead_reduction_pct"]
        assert reduction >= 85.0, (
            f"Overhead reduction {reduction:.1f}% is below the 85% target. "
            f"Total preprocessing time: {debug['total_preprocessing_ms']:.1f} ms"
        )

    def test_preprocessing_faster_than_2000ms(self):
        """Single-image preprocessing must complete within 2 000 ms."""
        img = _make_bgr_image(h=480, w=640)
        t0 = time.perf_counter()
        preprocess_pipeline(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 2000, (
            f"Preprocessing took {elapsed_ms:.1f} ms — exceeds 2 000 ms SLA target"
        )

    def test_step_timings_present(self):
        img = _make_bgr_image()
        _, debug = preprocess_pipeline(img)
        expected_steps = [
            "deskew_ms", "grayscale_ms", "clahe_ms",
            "denoise_ms", "threshold_ms", "morph_clean_ms",
        ]
        for step in expected_steps:
            assert step in debug["steps"], f"Missing timing for step: {step}"
            assert debug["steps"][step] >= 0

    def test_grayscale_input_handled(self):
        """Pipeline must handle a 2-D (grayscale) input without error."""
        gray_img = np.ones((200, 300), dtype=np.uint8) * 200
        processed, debug = preprocess_pipeline(gray_img)
        assert isinstance(processed, np.ndarray)
