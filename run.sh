#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
action=${1:-serve}
if ! docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
  if [[ ${WORLD_RSI_GROUP_APPLIED:-0} != 1 ]] && id -nG "$(id -un)" | tr ' ' '\n' | rg -qx docker; then
    printf -v task_command '%q ' "$PWD/run.sh" "$@"
    exec sg docker -c "WORLD_RSI_GROUP_APPLIED=1 $task_command"
  fi
  echo 'Docker unavailable; no host fallback.' >&2; exit 1
fi
case "$action" in
 serve) exec .venv/bin/python -m web.api ;;
 check) exec .venv/bin/python -c 'from harness.sandbox.docker import Sandbox; import json; print(json.dumps(Sandbox.health(),indent=2))' ;;
 build) exec docker build -t worldrsi:newton . ;;
 *) echo 'Usage: ./run.sh [serve|check|build]' >&2; exit 2 ;;
esac
