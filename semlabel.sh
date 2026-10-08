#!/usr/bin/env bash
# Start SemLabel (or any `semlabel` subcommand) without activating the virtual environment.
#   ./semlabel.sh                     web app on http://127.0.0.1:8765
#   ./semlabel.sh --host 0.0.0.0      reachable from other machines
#   ./semlabel.sh doctor              check the installation
# Settings come from $SEMLABEL_HOME/semlabel.env (default ~/.semlabel/semlabel.env).
# PYTHONPATH is cleared because ROS 2 setups put their own packages there, which can shadow the venv's.
set -e
cd "$(dirname "$0")"
H="${SEMLABEL_HOME:-$HOME/.semlabel}"
# KEY=value lines; variables already set in the environment win; $HOME and a leading ~ are expanded
if [ -f "$H/semlabel.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[2]}"; val="${BASH_REMATCH[3]}"
    [ -n "${!key+x}" ] && continue
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    val="${val//\$\{HOME\}/$HOME}"; val="${val//\$HOME/$HOME}"
    [[ "$val" == "~"* ]] && val="$HOME${val:1}"
    export "$key=$val"
  done < "$H/semlabel.env"
fi
VENV="${SEMLABEL_VENV:-$PWD/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "No virtual environment at $VENV — run scripts/install.sh first (or set SEMLABEL_VENV)." >&2
  exit 1
fi
exec env -u PYTHONPATH -u LD_LIBRARY_PATH "$VENV/bin/python" -m semlabel "$@"
