"""Tests for nexus.data.stream_utils"""

from types import SimpleNamespace

import pytest

from nexus.data.stream_utils import config_to_streams


class TestConfigToStreams:
    def test_single_stream_single_path(self):
        config = [{"paths": ["/data/mds1"]}]
        streams = config_to_streams(config)
        assert len(streams) == 1
        assert streams[0].local == "/data/mds1"
        assert streams[0].proportion == 1.0

    def test_two_streams_equal_proportion(self):
        config = [
            {"paths": ["/data/a"]},
            {"paths": ["/data/b"]},
        ]
        streams = config_to_streams(config)
        assert len(streams) == 2
        assert streams[0].local == "/data/a"
        assert streams[0].proportion == 0.5
        assert streams[1].local == "/data/b"
        assert streams[1].proportion == 0.5

    def test_explicit_proportions(self):
        config = [
            {"paths": ["/data/a"], "proportion": 0.25},
            {"paths": ["/data/b"], "proportion": 0.75},
        ]
        streams = config_to_streams(config)
        assert len(streams) == 2
        assert streams[0].proportion == 0.25
        assert streams[1].proportion == 0.75

    def test_accepts_simple_namespace(self):
        config = [
            SimpleNamespace(paths=["/data/a"], proportion=0.5),
            SimpleNamespace(paths=["/data/b"], proportion=0.5),
        ]
        streams = config_to_streams(config)
        assert len(streams) == 2
        assert streams[0].local == "/data/a"
        assert streams[1].local == "/data/b"

    def test_single_path_string_becomes_list(self):
        config = [{"paths": "/single/path"}]
        streams = config_to_streams(config)
        assert len(streams) == 1
        assert streams[0].local == "/single/path"

    def test_empty_streams_raises(self):
        with pytest.raises(ValueError, match="streams must be non-empty"):
            config_to_streams([])

    def test_missing_paths_raises(self):
        with pytest.raises(ValueError, match="paths is required"):
            config_to_streams([{"proportion": 1.0}])

    def test_partial_proportion_raises(self):
        with pytest.raises(ValueError, match="Either all streams must set proportion"):
            config_to_streams([
                {"paths": ["/a"], "proportion": 0.5},
                {"paths": ["/b"]},
            ])
