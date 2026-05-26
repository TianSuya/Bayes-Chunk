from pathlib import Path

from bayes_chunk.cli import main
from bayes_chunk.config import load_config


def test_config_loads_example_yaml():
    config = load_config(Path("bayes_chunk/configs/llama3_memit_are_bayes.yaml"))

    assert config.algorithm.name == "MEMIT_ARE"
    assert config.segmentation.type == "bayes"
    assert config.segmentation.boundary_stride == 40


def test_segment_dry_run_cli(capsys):
    code = main(["segment", "--config", "bayes_chunk/configs/llama3_memit_are_bayes.yaml", "--dry-run"])

    captured = capsys.readouterr()
    assert code == 0
    assert "boundary_stride" in captured.out


def test_list_algorithms_cli(capsys):
    code = main(["list-algorithms"])

    captured = capsys.readouterr()
    assert code == 0
    assert "MEMIT_ARE" in captured.out
