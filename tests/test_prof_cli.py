import pytest
from profiling.cli import build_parser


def test_parser_requires_dataset_and_defaults():
    parser = build_parser()
    args = parser.parse_args(["/some/dataset", "--out", "/tmp/out"])
    assert args.dataset == "/some/dataset"
    assert args.out == "/tmp/out"
    assert args.region == "manual0"  # default


def test_parser_errors_without_dataset():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
