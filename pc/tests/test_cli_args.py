import argparse


def test_preparser_does_not_treat_conf_as_config():
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default="")
    args, unknown = pre.parse_known_args(["--conf", "0.20", "--model", "x.onnx"])
    assert args.config == ""
    assert unknown == ["--conf", "0.20", "--model", "x.onnx"]


def test_config_still_parses_explicitly():
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", default="")
    args, unknown = pre.parse_known_args(["--config", "preset.json", "--conf", "0.20"])
    assert args.config == "preset.json"
    assert unknown == ["--conf", "0.20"]
