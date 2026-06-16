import textwrap
from profiling.attribution import function_for


def test_function_for_maps_line_to_enclosing_function(tmp_path):
    src = textwrap.dedent("""\
        x = 1

        def outer():
            a = 1
            b = 2
            return a + b

        def other():
            return 0
    """)
    f = tmp_path / "mod.py"
    f.write_text(src)

    assert function_for(str(f), 5) == "mod:outer"  # line "b = 2"
    assert function_for(str(f), 9) == "mod:other"  # line "return 0"
    assert function_for(str(f), 1) == "mod:<module>"  # top-level line
