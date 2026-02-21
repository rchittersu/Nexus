"""Tests for datasets/precompute.py"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False

# Add datasets dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precompute import (
    parse_args,
    _caption_sample_weights,
    _datadir_to_streams,
    _discover_subfolders,
    _partition_subfolders,
    _sample_caption,
)


def _get_precompute_main():
    from precompute import main
    return main


class TestParseArgs:
    def test_required_datadir(self):
        with patch.object(sys, "argv", ["precompute.py", "--datadir", "/path/to/mds"]):
            args = parse_args()
            assert args.datadir == "/path/to/mds"

    def test_default_values(self):
        with patch.object(sys, "argv", ["precompute.py", "--datadir", "/path/to/mds"]):
            args = parse_args()
            assert args.savedir == ""
            assert args.num_proc is None
            assert args.image_resolutions == [512]


class TestDatadirToStreams:
    def test_single_str_becomes_list(self):
        from streaming import Stream

        streams = _datadir_to_streams("/path/to/mds")
        assert len(streams) == 1
        assert isinstance(streams[0], Stream)

    def test_list_of_strs_becomes_list_of_streams(self):
        from streaming import Stream

        streams = _datadir_to_streams(["/a", "/b"])
        assert len(streams) == 2
        assert all(isinstance(s, Stream) for s in streams)


class TestDiscoverSubfolders:
    def test_empty_when_not_dir(self, tmp_path):
        p = tmp_path / "nonexistent"
        assert _discover_subfolders(str(p)) == []

    def test_finds_digit_subfolders(self, tmp_path):
        (tmp_path / "0").mkdir()
        (tmp_path / "1").mkdir()
        (tmp_path / "meta").mkdir()
        result = _discover_subfolders(str(tmp_path))
        assert len(result) == 2
        assert any("0" in r for r in result)
        assert any("1" in r for r in result)


class TestPartitionSubfolders:
    def test_num_proc_geq_subfolders(self):
        sub = ["/a", "/b", "/c"]
        parts = _partition_subfolders(sub, 5)
        assert len(parts) == 5
        assert parts[0] == ["/a"]
        assert parts[1] == ["/b"]
        assert parts[2] == ["/c"]
        assert parts[3] == []
        assert parts[4] == []


class TestCaptionSampleWeights:
    def test_uniform_when_none(self):
        w = _caption_sample_weights(3, None)
        assert w.shape == (3,)
        np.testing.assert_array_almost_equal(w, [1 / 3] * 3)
