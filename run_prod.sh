#!/usr/bin/env bash
# Run EchoMatch with a production WSGI server (gunicorn) instead of
# Flask's development server.
#
# Usage:
#   ./run_prod.sh            # binds 0.0.0.0:8000
#   PORT=9000 ./run_prod.sh  # custom port

set -euo pipefail
PORT="${PORT:-8000}"

exec gunicorn --bind "0.0.0.0:${PORT}" --workers 2 --threads 2 --timeout 60 app:app
