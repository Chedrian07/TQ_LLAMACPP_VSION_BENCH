from __future__ import annotations

from pathlib import Path

from tq_bench.config import load_models, load_profiles


def test_load_profiles_resolves_relative_paths(tmp_path: Path) -> None:
    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text(
        """
profiles:
  colab:
    slot_save_path: ./kvcache
    results_dir: ../results
    cache_ram: 4096
""".strip(),
        encoding="utf-8",
    )

    profiles = load_profiles(profiles_yaml)
    profile = profiles["colab"]

    assert profile.slot_save_path == (tmp_path / "kvcache").resolve()
    assert profile.results_dir == (tmp_path.parent / "results").resolve()
    assert profile.cache_ram == 4096


def test_load_models_parses_download_metadata(tmp_path: Path) -> None:
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        """
models:
  demo:
    family: qwen3-vl
    model_path: ./models/demo/model.gguf
    mmproj_path: ./models/demo/mmproj.gguf
    quantized_model_paths:
      q4_k_m: ./models/demo/demo-q4.gguf
    download:
      repo_id: example/demo
      files:
        bf16:
          filename: Demo-BF16.gguf
          suffix: BF16.gguf
        q4_k_m: demo-q4.gguf
        mmproj:
          filename: mmproj.gguf
""".strip(),
        encoding="utf-8",
    )

    models = load_models(models_yaml)
    model = models["demo"]

    assert model.download is not None
    assert model.download.repo_id == "example/demo"
    assert model.download.files["bf16"].filename == "Demo-BF16.gguf"
    assert model.download.files["bf16"].suffix == "BF16.gguf"
    assert model.download.files["q4_k_m"].suffix == "demo-q4.gguf"
    assert model.download.files["mmproj"].filename == "mmproj.gguf"
