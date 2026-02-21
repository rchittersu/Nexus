"""Tests for datasets/prepare/dreambooth/prepare.py"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from streaming import StreamingDataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare import (
    _collect_class_items,
    _collect_instance_items,
    main,
    parse_arguments,
)


class TestParseArguments:
    def test_instance_data_dir_and_local_mds_required(self):
        with patch.object(sys, "argv", ["prepare.py", "--instance_data_dir", "/dog"]):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_default_instance_prompt(self):
        with patch.object(
            sys,
            "argv",
            ["prepare.py", "--instance_data_dir", "/dog", "--local_mds_dir", "/out"],
        ):
            args = parse_arguments()
            assert args.instance_prompt == "a photo of sks dog"
            assert args.class_prompt == "a dog"


class TestCollectInstanceItems:
    def test_instance_data_dir_collects_images(self, tmp_path):
        (tmp_path / "img1.png").touch()
        img_path = tmp_path / "img2.png"
        Image.new("RGB", (300, 300), color=(0, 0, 0)).save(img_path)

        class Args:
            dataset_name = None
            download_dir = None
            instance_data_dir = str(tmp_path)
            instance_prompt = "a photo of sks dog"
            min_size = 0
            repeats = 1

        items = _collect_instance_items(Args())
        assert len(items) >= 1
        for path, cap in items:
            assert cap == "a photo of sks dog"
            assert Path(path).exists()

    def test_requires_dataset_name_or_instance_dir(self):
        class Args:
            dataset_name = None
            instance_data_dir = None

        with pytest.raises(ValueError, match="Specify either"):
            _collect_instance_items(Args())


class TestCollectClassItems:
    def test_empty_when_no_class_dir(self):
        class Args:
            class_data_dir = None

        assert _collect_class_items(Args()) == []

    def test_collects_from_class_dir(self, tmp_path):
        Image.new("RGB", (400, 400), color=(1, 2, 3)).save(tmp_path / "dog1.jpg")

        class Args:
            class_data_dir = str(tmp_path)
            class_prompt = "a dog"
            min_size = 0

        items = _collect_class_items(Args())
        assert len(items) == 1
        assert items[0][1] == "a dog"


class TestIntegrationPrepareAndStreamingDataset:
    def test_instance_only_mds_readable(self, tmp_path):
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (512, 512), color=(i, 0, 0)).save(instance_dir / f"img_{i}.png")

        mds_dir = tmp_path / "mds"
        with patch.object(
            sys,
            "argv",
            [
                "prepare.py",
                "--instance_data_dir",
                str(instance_dir),
                "--instance_prompt",
                "a photo of sks dog",
                "--local_mds_dir",
                str(mds_dir),
                "--num_proc",
                "1",
                "--min_size",
                "0",
            ],
        ):
            main()

        assert mds_dir.exists()
        assert (mds_dir / "0").exists()
        assert (mds_dir / "index.json").exists()

        ds = StreamingDataset(
            local=str(mds_dir),
            batch_size=1,
            shuffle=False,
        )
        samples = list(ds)
        assert len(samples) == 3
        for s in samples:
            assert "image" in s
            assert "caption" in s
            assert s["caption"] == "a photo of sks dog"
            assert "width" in s and "height" in s
