"""
Training configuration tests.

configs/train.yaml used to be read by nothing: main.py had no argument parsing,
so `--config` was silently discarded by Python and the file never took effect.
These lock in that the flag is parsed and that unknown flags fail loudly.
"""

import pytest

from main import load_train_config, parse_args


def test_load_train_config_reads_the_shipped_file():
    config = load_train_config("configs/train.yaml")
    assert config["num_episodes"] == 10000
    assert config["learning_rate"] == pytest.approx(0.0003)
    assert config["gamma"] == pytest.approx(0.99)


def test_load_train_config_without_a_path_returns_empty():
    assert load_train_config(None) == {}


def test_load_train_config_raises_on_a_missing_file():
    with pytest.raises(FileNotFoundError):
        load_train_config("configs/does-not-exist.yaml")


def test_config_flag_is_parsed_rather_than_discarded():
    args = parse_args(["--config", "configs/train.yaml", "--episodes", "2"])
    assert args.config == "configs/train.yaml"
    assert args.episodes == 2


def test_unknown_flags_fail_loudly():
    """The original defect: an unrecognised flag was silently ignored."""
    with pytest.raises(SystemExit):
        parse_args(["--not-a-real-flag"])


def test_config_values_actually_reach_the_agent(tmp_path):
    """
    Loading the file is not enough; the values must be applied.

    Deliberately uses numbers that differ from the PPOAgent defaults. The
    shipped train.yaml sets learning_rate to 0.0003, which is exactly the
    default, so a run using it cannot distinguish applied from ignored.
    """
    from main import train_and_save

    agent, _ = train_and_save(
        model_path=str(tmp_path / "m.pt"),
        num_episodes=1,
        config={"learning_rate": 0.00711, "gamma": 0.5},
    )
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.00711)
    assert agent.gamma == pytest.approx(0.5)
