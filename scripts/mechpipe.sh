#!/usr/bin/env bash
# Run the mechpipe CLI inside WSL with its own uv venv (kept in the WSL home dir, not on /mnt/c).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
export PATH="$HOME/.local/bin:$PATH"   # uv lives here; non-login shells don't have it on PATH
export UV_PROJECT_ENVIRONMENT="${MECHPIPE_VENV:-$HOME/.venvs/rubicon-client}"
exec uv run --quiet mechpipe "$@"
