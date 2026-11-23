from pathlib import Path
import re, os
from typing import List, Optional 

# File extensions used to recognise bare filenames such as ``out.log``.
_KNOWN_EXTENSIONS = {
    "a", "awk", "bash", "bz2", "c", "cc", "cfg", "conf", "cpp", "css", "csv",
    "dat", "data", "db", "err", "f", "f77", "f90", "f95", "gz", "h", "hpp",
    "htm", "html", "incar", "ini", "jpeg", "jpg", "js", "json", "kpoints",
    "log", "md", "mod", "npy", "npz", "o", "out", "pdf", "pkl", "png",
    "poscar", "potcar", "py", "pyc", "pyo", "rst", "sed", "sh", "so",
    "sqlite", "sqlite3", "svg", "tar", "toml", "ts", "tsv", "txt", "vasp",
    "xml", "xz", "yaml", "yml", "zip", "zsh",
}
_EXT_ALT = "|".join(sorted(_KNOWN_EXTENSIONS, key=len, reverse=True))

# A path is one of: /abs, ~/x, ./x, ../x, a/relative/path, a bare filename
# with a known extension, a bare existing name (``ls catagent``), or
# ``.``/``..``/``~``.
_PATH_RE = re.compile(
    rf"""
    (?:~|\.{{1,2}})?/[^\s"'`;|&<>():$]+
  | [A-Za-z0-9_.+${{}}-]+/[^\s"'`;|&<>():$]+
  | [A-Za-z0-9_+.-]*\.(?:{_EXT_ALT})\b
  | ~(?=[\s"'`;|&<>()]|$)
  | \.{{1,2}}(?=[\s"'`;|&<>()]|$)
  | [A-Za-z0-9_.+~#-]+(?=[\s"'`;|&<>()/]|$)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Quoted strings may contain spaces: "dir with space/file.txt".  The other
# quote character is excluded so that code such as
# ``'print(open("/tmp/a"))'`` is scanned for its inner string.
_QUOTED_RE = re.compile(
    r'"((?:\\.|[^"\'\\\n])*)"'
    r"|'((?:\\.|[^\"'\\\n])*)'"
)

# URLs and remote ``host:path`` operands are not local filesystem paths; mask
# them before scanning so their slashes are not reported.
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'`;|&<>()]+")
_REMOTE_RE = re.compile(r"(?<![\w/])(?:[\w.+-]+@)?[\w.-]+:[\w./~+-]+")
_COMMENT_RE = re.compile(r"(?m)(?<!\S)#[^\n]*")


def _quoted_body_is_path(body: str) -> bool:
    """Decide whether a quoted string is one path rather than code/text."""
    body = body.strip()
    if body.startswith(("/", "~", "./", "../")):
        return True
    if any(ch in body for ch in ";|&<>()="):
        return False
    return Path(body).suffix.lower().lstrip(".") in _KNOWN_EXTENSIONS


def _normalize_candidate(raw: str, cwd: Optional[str]) -> Optional[str]:
    """Return a normalized path, or ``None`` when ``raw`` is not a path."""
    token = raw.strip().strip("'\"")
    token = token.replace('\\"', '"').replace("\\'", "'")
    if not token or token.startswith("-") or _URL_RE.match(token):
        return None
    if any(ch in token for ch in "();`"):
        return None

    expanded = os.path.expandvars(os.path.expanduser(token))
    if token not in {".", ".."}:
        suffix = Path(expanded).suffix.lower().lstrip(".")
        is_path = (
            expanded.startswith(("/", "./", "../", "~/"))
            or "/" in expanded
            or any(ch in expanded for ch in "*?[")
            or suffix in _KNOWN_EXTENSIONS
        )
        if not is_path and suffix == "" and "/" not in expanded:
            # A bare relative name such as ``ls catagent`` is a path when it
            # exists in the working directory.
            base = cwd if cwd is not None else os.getcwd()
            try:
                is_path = os.path.exists(os.path.join(base, expanded))
            except (OSError, ValueError):
                is_path = False
        if not is_path:
            return None

    if cwd is not None and not os.path.isabs(expanded):
        expanded = os.path.join(cwd, expanded)
    try:
        return os.path.normpath(expanded)
    except (OSError, ValueError):
        return expanded


def parse_path(text: str, cwd: Optional[str] = None) -> List[str]:
    """Extract the filesystem paths that appear in a shell command or code.

    The parser is regex based and deliberately simple; it scans the whole
    string, so paths inside quotes, ``python3 -c "..."``, and heredoc bodies
    are found without special casing.  Shell operators (``>``, ``>>``,
    ``||``, ``&&``, ``|``, ``;``) are just delimiters.  Bare names without an
    extension (``ls catagent``) are reported when they exist under ``cwd``.

    Args:
        text: Shell command or a fragment of Python code.
        cwd: If given, relative paths are resolved against this directory.

    Returns:
        Unique paths in first-seen order.  ``~`` and environment variables are
        expanded; glob patterns such as ``*.log`` are preserved.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")

    cwd_abs = os.path.abspath(os.fspath(cwd)) if cwd is not None else None
    masked = _COMMENT_RE.sub(lambda match: " " * len(match.group(0)), text)
    for pattern in (_URL_RE, _REMOTE_RE):
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)

    candidates: List[tuple[int, str]] = []
    quoted_chars = list(masked)
    for match in _QUOTED_RE.finditer(masked):
        body = match.group(1) if match.group(1) is not None else match.group(2)
        if _quoted_body_is_path(body):
            candidates.append((match.start(), body))
        else:
            # A quoted command or code fragment: scan its contents for paths.
            for inner in _PATH_RE.finditer(body):
                candidates.append((match.start() + inner.start(), inner.group(0)))
        # Do not also report fragments of a quoted path as separate paths.
        for index in range(match.start(), match.end()):
            quoted_chars[index] = " "

    unquoted = "".join(quoted_chars)
    for match in _PATH_RE.finditer(unquoted):
        candidates.append((match.start(), match.group(0)))

    result: List[str] = []
    seen: set[str] = set()
    for _, candidate in sorted(candidates, key=lambda item: item[0]):
        path = _normalize_candidate(candidate, cwd_abs)
        if path is None or path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result

