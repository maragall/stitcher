from profiling.cli import build_parser


def test_parser_has_perpair_flag_default_false():
    args = build_parser().parse_args(["/some/dataset"])
    assert args.perpair is False


def test_parser_perpair_flag_sets_true():
    args = build_parser().parse_args(["/some/dataset", "--perpair"])
    assert args.perpair is True
