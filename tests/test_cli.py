import json
import sys

import pytest

from inference_lab.cli import main


@pytest.mark.parametrize("command", ["inspect", "check", "generate", "benchmark", "profile"])
def test_cli_with_local_checkpoint(tiny_model, tmp_path, monkeypatch, capsys, command):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    checkpoint = tmp_path / "tiny-model"
    tiny_model.save_pretrained(checkpoint)
    vocabulary = {"[UNK]": 0, **{f"token{i}": i for i in range(1, 97)}}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = "{{ messages[0]['content'] }}"
    fast.save_pretrained(checkpoint)
    arguments = ["inference-lab", command, "--model", str(checkpoint), "--device", "cpu", "--dtype", "float32"]
    if command in ("inspect", "check", "profile"):
        arguments += ["--input-length", "4"]
    if command == "check":
        arguments += ["--steps", "2"]
    if command == "generate":
        arguments += ["--prompt", "token3 token4", "--new-tokens", "2"]
    if command == "benchmark":
        arguments += ["--batch-sizes", "1", "--input-lengths", "4", "--new-tokens", "2",
                      "--warmup", "0", "--repeats", "1", "--output", str(tmp_path / "bench.json")]
    if command == "profile":
        arguments += ["--new-tokens", "2", "--output", str(tmp_path / "trace.json")]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    text = capsys.readouterr().out
    if command == "inspect":
        report = json.loads(text)
        assert report["input_shape"] == [1, 4]
        assert report["actual_kv_mib"] == report["expected_kv_mib"]
    elif command == "check":
        assert json.loads(text)["passed"] is True
    elif command == "generate":
        assert '"output_tokens": 2' in text
    elif command == "benchmark":
        assert len(json.loads((tmp_path / "bench.json").read_text())["records"]) == 2
    elif command == "profile":
        trace = json.loads((tmp_path / "trace.json").read_text())
        names = {event.get("name") for event in trace["traceEvents"]}
        assert {"prefill", "decode"} <= names
