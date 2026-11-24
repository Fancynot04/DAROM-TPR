#!/bin/zsh
set -eu
cd -- "${0:A:h}/../.."

dashboard_python="${DAROM_DASHBOARD_PYTHON:-}"
if [[ -z "$dashboard_python" ]]; then
    git_data="$(git rev-parse --path-format=absolute --git-common-dir)"
    for candidate in "$PWD/.venv/bin/python" "${git_data:h}/.venv/bin/python" \
                     "$PWD/tmp/ai/trajectory-dashboard/.venv/bin/python" \
                     "${git_data:h}/tmp/ai/trajectory-dashboard/.venv/bin/python" \
                     "$(command -v python3 || true)"; do
        if [[ -x "$candidate" ]] && "$candidate" -c 'import streamlit, plotly' >/dev/null 2>&1; then
            dashboard_python="$candidate"
            break
        fi
    done
fi
if [[ ! -x "$dashboard_python" ]]; then
    print -u2 '找不到轨迹分析的 Python 环境。请查看同目录 README.md。'
    exit 1
fi

remote_options=(--remote cyl@anhuiuniversity.icu --ssh-port 21501)
if (( $# > 0 )); then
    remote_options=("$@")
fi
exec "$dashboard_python" -m streamlit run scripts/trajectory_dashboard/app.py \
    --server.address 127.0.0.1 --server.port 8501 --server.headless false \
    --server.showEmailPrompt false --browser.gatherUsageStats false \
    -- "${remote_options[@]}"
