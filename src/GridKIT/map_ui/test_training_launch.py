# map_ui/test_training_launch.py
import ast
import io
from pathlib import Path

SOURCE = Path(__file__).with_name("map_widget_training.py")


def _launch_source() -> str:
    """The body of _launch_training, as text.

    Read rather than called: the function spawns a detached training process,
    which a test must not do. What matters here is the environment it hands
    that process, and that is visible in the source.
    """
    tree = ast.parse(io.open(SOURCE, encoding="utf-8").read())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_launch_training")
    return ast.unparse(node)


def test_the_training_subprocess_gets_utf8_stdio():
    # The child writes train.log through a handle this process opened, but picks
    # its own stdout encoding from the locale: cp1250 on a German Windows. One
    # arrow in a trainer log line raised UnicodeEncodeError, train_run.py's
    # catch-all marked the run FAILED, and the dashboard reported a crash that
    # had nothing to do with the grid. It happened twice, because the arrow came
    # back when the log line was rewritten, which is why the guard lives here
    # rather than only in the string that happened to break.
    assert "PYTHONIOENCODING" in _launch_source()


def test_the_subprocess_still_gets_both_import_roots():
    assert "PYTHONPATH" in _launch_source()


def test_pythonpath_is_joined_with_os_pathsep():
    # ":" works on Linux and silently breaks every Ray worker on Windows
    assert "os.pathsep" in _launch_source()
