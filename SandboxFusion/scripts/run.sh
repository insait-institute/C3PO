#!/bin/bash
# =============================================================================
# run.sh -- Sandbox server startup script
# =============================================================================
# Determines its own directory, changes to the repository root (one level up),
# and launches the server via `make run-online`. The HOST is set to the empty
# string so the server binds to all interfaces. PORT defaults to 8080 but can
# be overridden by setting the PORT environment variable.
# =============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR"/..

make run-online HOST="''" PORT=${PORT:-8080}
