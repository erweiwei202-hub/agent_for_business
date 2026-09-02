from pathlib import Path


def test_grpo_launchers_capture_console_output_in_output_dir():
    root = Path(__file__).resolve().parents[1]

    for launcher in (root / "GRPO_train.sh", root / "scripts" / "GRPO_train.sh"):
        text = launcher.read_text(encoding="utf-8")
        assert 'mkdir -p "$OUTPUT_DIR"' in text
        assert 'exec > >(tee -a "$OUTPUT_DIR/console.log") 2>&1' in text


def test_grpo_launchers_enable_parallel_generation_by_default():
    root = Path(__file__).resolve().parents[1]

    for launcher in (root / "GRPO_train.sh", root / "scripts" / "GRPO_train.sh"):
        text = launcher.read_text(encoding="utf-8")
        assert 'PARALLEL_GENERATION="1"' in text
        assert 'MAX_WORKERS="2"' in text


def test_sft_launcher_uses_qwen_clean_dataset():
    root = Path(__file__).resolve().parents[1]

    text = (root / "scripts" / "sft_train.sh").read_text(encoding="utf-8")

    assert 'DATASET="outputs/sft/accepted-qwen-clean.jsonl"' in text
