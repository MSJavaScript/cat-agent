#!/usr/bin/env bash
#
# Install cat-agent and its dependencies.
#
#   ./install.sh                  # into the current Python environment
#   ./install.sh --user           # into ~/.local (no root)
#   ./install.sh --venv .venv     # create a virtualenv and install there
#   ./install.sh --editable       # editable install (development)
#   ./install.sh --no-deps        # dependencies are already installed
#
# After installation the `catagent` command is available:
#
#   catagent cli    --dotenv /path/to/.env --db ./catagent.db
#   catagent server --dotenv /path/to/.env --db ./catagent.db --port 5600
#
# The skills directory ships with the package; --skill-index-json is optional
# and can be overridden with CATAGENT_SKILL_INDEX.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV=""
EDITABLE=0
PIP_ARGS=()

usage() {
    cat <<'EOF'
Install cat-agent and its dependencies.

Usage: ./install.sh [options] [-- PIP_OPTIONS...]

  --venv PATH      create (or reuse) a virtualenv at PATH and install there
  --python PATH    Python interpreter to use (default: $PYTHON or python3)
  --user           pass --user to pip (no root, installs to ~/.local)
  --editable, -e   editable install; the source checkout stays in use
  --no-deps        skip dependency installation
  -h, --help       show this help

Examples:
  ./install.sh
  ./install.sh --venv .venv
  ./install.sh --user --no-deps
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            [[ $# -ge 2 ]] || { echo "error: --venv needs a path" >&2; exit 2; }
            VENV="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "error: --python needs a path" >&2; exit 2; }
            PYTHON="$2"
            shift 2
            ;;
        --user)
            PIP_ARGS+=(--user)
            shift
            ;;
        --editable|-e)
            EDITABLE=1
            shift
            ;;
        --no-deps)
            PIP_ARGS+=(--no-deps)
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PIP_ARGS+=("$@")
            break
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$REPO_ROOT"

# Stale setuptools output would otherwise be copied into the new wheel, which
# can resurrect files that were deleted or moved since the last build.
rm -rf "$REPO_ROOT/build" "$REPO_ROOT"/*.egg-info

if [[ -n "$VENV" ]]; then
    if [[ ! -x "$VENV/bin/python" ]]; then
        echo "Creating virtualenv at $VENV"
        "$PYTHON" -m venv "$VENV"
    fi
    PYTHON="$VENV/bin/python"
fi

echo "Installing cat-agent from $REPO_ROOT"
echo "  interpreter: $("$PYTHON" -c 'import sys; print(sys.executable)')"

TARGET=(.)
if [[ $EDITABLE -eq 1 ]]; then
    TARGET=(-e .)
fi

# ${arr[@]+...} keeps `set -u` happy on bash versions whose empty arrays are
# otherwise reported as unbound.
if ! "$PYTHON" -m pip install ${PIP_ARGS[@]+"${PIP_ARGS[@]}"} "${TARGET[@]}"; then
    cat >&2 <<'EOF'

Installation failed.
If pip reports an "externally managed environment", retry with one of:

    ./install.sh --venv .venv
    ./install.sh --user

EOF
    exit 1
fi

# Verify the console script and the bundled skill index.  The import runs from
# /tmp so a source checkout in the current directory cannot mask the installed
# package.
COMMAND="$("$PYTHON" - <<'PY'
import os
import shutil
import site
import sysconfig

# Prefer this interpreter's own scripts directory: with --venv, an unrelated
# catagent earlier on PATH must not be reported as the installed command.
candidates = (
    os.path.join(sysconfig.get_path("scripts"), "catagent"),
    os.path.join(site.USER_BASE, "bin", "catagent"),
)
found = next((path for path in candidates if os.path.isfile(path)), "")
print(found or shutil.which("catagent") or "")
PY
)"

echo
if [[ -n "$COMMAND" ]]; then
    echo "catagent command: $COMMAND"
    BIN_DIR="$(dirname "$COMMAND")"
else
    BIN_DIR="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
    echo "warning: console script not found (expected in $BIN_DIR)" >&2
fi

SKILL_INDEX="$(cd /tmp && env -u PYTHONPATH -u CATAGENT_SKILL_INDEX "$PYTHON" -c \
    'from catagent._paths import default_skill_index; print(default_skill_index())')"
if [[ -z "$SKILL_INDEX" ]]; then
    echo "error: the installed package does not contain skills/skill_index.json" >&2
    exit 1
fi
echo "skill index:     $SKILL_INDEX"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        echo "warning: $BIN_DIR is not on PATH. Add it with:"
        echo "    export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

cat <<'EOF'

Installed. Next steps:

    catagent --help
    catagent cli    --dotenv /path/to/.env --db ./catagent.db
    catagent server --dotenv /path/to/.env --db ./catagent.db --port 5600

The jobmanager configuration is not installed: provide your own file with
`python -m jobmanager.server --setting /path/to/jobmanager_config.json`.
EOF
