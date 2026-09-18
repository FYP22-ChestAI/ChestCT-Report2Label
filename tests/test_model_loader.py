from pathlib import Path

from report2label.models.model_loader import ModelConfig

_MODEL_YAML_TEMPLATE = (
    "base_model: dummy\n"
    "checkpoint_path: {checkpoint_path}\n"
    "num_labels: 18\n"
    "max_length: 512\n"
    "do_lower_case: true\n"
    "device: cpu\n"
    "default_threshold: 0.5\n"
)


def test_relative_checkpoint_path_resolves_against_project_root_not_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    configs_dir = project_root / "configs"
    configs_dir.mkdir(parents=True)
    model_yaml = configs_dir / "model.yaml"
    model_yaml.write_text(_MODEL_YAML_TEMPLATE.format(checkpoint_path="models/RadBertClassifier.pth"))

    # Simulate running from an unrelated working directory, e.g. a notebook
    # executed via nbconvert (kernel cwd = notebooks/, not the project root).
    unrelated_cwd = tmp_path / "somewhere_else"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = ModelConfig.from_yaml(model_yaml)
    assert config.checkpoint_path == str(project_root / "models" / "RadBertClassifier.pth")


def test_absolute_checkpoint_path_is_left_untouched(tmp_path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    model_yaml = configs_dir / "model.yaml"
    absolute_checkpoint = tmp_path / "elsewhere" / "checkpoint.pth"
    model_yaml.write_text(_MODEL_YAML_TEMPLATE.format(checkpoint_path=absolute_checkpoint.as_posix()))

    config = ModelConfig.from_yaml(model_yaml)
    assert Path(config.checkpoint_path) == absolute_checkpoint
