"""Tests for datasets/precompute_hf.py"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add datasets dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precompute_hf import (
    _caption_sample_weights,
    _sample_caption,
    parse_args,
)


class TestParseArgs:
    def test_required_dataset(self):
        with patch.object(sys, "argv", ["precompute_hf.py", "--dataset", "user/dataset"]):
            args = parse_args()
            assert args.dataset == "user/dataset"

    def test_default_values(self):
        with patch.object(sys, "argv", ["precompute_hf.py", "--dataset", "user/dataset"]):
            args = parse_args()
            assert args.savedir == ""
            assert args.resolution == 512
            assert args.image_column == "image"
            assert args.prompt_column == "caption"
            assert args.condition_column is None
            assert args.streaming is False

    def test_column_overrides(self):
        with patch.object(
            sys,
            "argv",
            [
                "precompute_hf.py",
                "--dataset",
                "user/ds",
                "--image_column",
                "img",
                "--prompt_column",
                "text",
                "--condition_column",
                "source",
            ],
        ):
            args = parse_args()
            assert args.image_column == "img"
            assert args.prompt_column == "text"
            assert args.condition_column == "source"

    def test_streaming_flag(self):
        with patch.object(
            sys, "argv", ["precompute_hf.py", "--dataset", "user/ds", "--streaming"]
        ):
            args = parse_args()
            assert args.streaming is True


class TestCaptionSampleWeights:
    def test_uniform_when_none(self):
        w = _caption_sample_weights(3, None)
        assert w.shape == (3,)
        np.testing.assert_array_almost_equal(w, [1 / 3] * 3)

    def test_custom_weights(self):
        w = _caption_sample_weights(3, [1.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(w, [0.25, 0.5, 0.25])


class TestSampleCaption:
    def test_picks_with_weights(self):
        rng = np.random.default_rng(42)
        captions = ["a", "b", "c"]
        weights = np.array([1, 0, 0])  # always pick first
        result = _sample_caption(captions, weights, rng, clean=False)
        assert result == "a"
