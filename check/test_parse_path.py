"""Deterministic tests for ``catagent.tools.other.util_tools.parse_path``.

Run from the repository root:

    python -m check.test_parse_path

No shell command in this file is executed; ``parse_path`` only inspects the
text.  The fake working directory does not need to exist because relative
paths are resolved with ``os.path.normpath``.
"""

from __future__ import annotations

import os
import tempfile

from catagent.tools.other.parse_path import parse_path

CWD = os.path.normpath("/tmp/catagent_parse_path_test/work")


def _abs(path: str) -> str:
    return os.path.normpath(os.path.join(CWD, path))


def paths(command: str) -> list[str]:
    return parse_path(command, cwd=CWD)


def expect(command: str, *expected: str) -> list[str]:
    got = paths(command)
    for path in expected:
        assert path in got, f"{command!r}: missing {path!r} in {got!r}"
    return got


def reject(command: str, *unexpected: str) -> list[str]:
    got = paths(command)
    for path in unexpected:
        assert path not in got, f"{command!r}: unexpected {path!r} in {got!r}"
    return got


def test_relative_and_absolute() -> None:
    assert parse_path("cat file.txt") == ["file.txt"]
    assert paths("cat file.txt") == [_abs("file.txt")]
    assert paths("cat /etc/passwd") == ["/etc/passwd"]
    assert paths("cat ./a/../b.txt") == [_abs("b.txt")]
    assert parse_path("cat file.txt", cwd="/tmp/elsewhere") == [
        "/tmp/elsewhere/file.txt"
    ]


def test_operators_and_redirects() -> None:
    command = (
        "cat file.txt > out.log 2>&1 || echo fail >> err.log; "
        "cp ./a ../b.c && rm -rf /tmp/x/*.log | tee piped.log"
    )
    expect(
        command,
        _abs("file.txt"),
        _abs("out.log"),
        _abs("err.log"),
        _abs("a"),
        os.path.normpath(os.path.join(CWD, "..", "b.c")),
        "/tmp/x/*.log",
        _abs("piped.log"),
    )
    # ``2>&1`` must not be reported as the path ``1``.
    reject(command, _abs("1"), "1")


def test_quoted_paths() -> None:
    expect(
        'cat "dir with space/file name.txt"',
        _abs("dir with space/file name.txt"),
    )
    expect("cat 'single quoted/data.txt'", _abs("single quoted/data.txt"))
    got = expect(
        "bash -c 'cat /etc/passwd > /tmp/passwd.copy'",
        "/etc/passwd",
        "/tmp/passwd.copy",
    )
    assert _abs("cat /etc/passwd > /tmp/passwd.copy") not in got


def test_python_dash_c() -> None:
    expect(
        "python3 -c 'print(open(\"/tmp/a.txt\").read())'",
        "/tmp/a.txt",
    )
    expect(
        "python3 -c 'print(open(\"./data/in.txt\").read())'",
        _abs("data/in.txt"),
    )
    expect(
        "python3 -c \"print(open('/tmp/with space/b.txt').read())\"",
        "/tmp/with space/b.txt",
    )
    # Code without paths must not be mistaken for a path.
    assert paths("python3 -c 'import os; print(1)'") == []


def test_heredocs() -> None:
    python_code = """python3 - <<'EOF'
from pathlib import Path
Path("/tmp/heredoc-out.txt").write_text("x")
print(open("rel/input.txt").read())
EOF"""
    expect(python_code, "/tmp/heredoc-out.txt", _abs("rel/input.txt"))

    shell_code = """bash <<'EOF'
cat /etc/hosts > /tmp/hosts.copy
EOF"""
    expect(shell_code, "/etc/hosts", "/tmp/hosts.copy")


def test_tilde_env_globs() -> None:
    expected_home = os.path.normpath(
        os.path.join(os.path.expanduser("~"), "data", "*.csv")
    )
    assert expected_home in paths("ls ~/data/*.csv")

    old = os.environ.get("CATAGENT_PARSE_PATH_DIR")
    os.environ["CATAGENT_PARSE_PATH_DIR"] = "/tmp/envdir"
    try:
        expect("cat $CATAGENT_PARSE_PATH_DIR/x.txt", "/tmp/envdir/x.txt")
    finally:
        if old is None:
            os.environ.pop("CATAGENT_PARSE_PATH_DIR", None)
        else:
            os.environ["CATAGENT_PARSE_PATH_DIR"] = old


def test_urls_remotes_and_comments() -> None:
    assert paths("curl https://example.com/a/b") == []
    assert paths("git@github.com:org/repo.git") == []
    assert paths("echo hi # /secret/path") == []
    assert paths("echo 'hello world'") == []


def test_bare_existing_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "catagent"))
        with open(os.path.join(tmp, "Makefile"), "w"):
            pass
        assert parse_path("ls catagent", cwd=tmp) == [
            os.path.join(tmp, "catagent")
        ]
        assert parse_path("cat Makefile", cwd=tmp) == [
            os.path.join(tmp, "Makefile")
        ]
        # A bare word that does not exist is not reported.
        assert parse_path("ls does_not_exist_xyz", cwd=tmp) == []
        assert parse_path("echo hello", cwd=tmp) == []


def test_dots_dedup_and_empty() -> None:
    assert paths("cat .") == [CWD]
    assert paths("cat ..") == [os.path.dirname(CWD)]
    assert paths("cat a.txt a.txt /tmp/b.txt a.txt") == [
        _abs("a.txt"),
        "/tmp/b.txt",
    ]
    assert parse_path("", cwd=CWD) == []
    assert parse_path("   \n\t ") == []
    assert parse_path("a || b && c | d; e & f") == []


def main() -> None:
    tests = [
        test_relative_and_absolute,
        test_operators_and_redirects,
        test_quoted_paths,
        test_python_dash_c,
        test_heredocs,
        test_tilde_env_globs,
        test_urls_remotes_and_comments,
        test_bare_existing_names,
        test_dots_dedup_and_empty,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print("parse_path tests: OK")


if __name__ == "__main__":
    main()
