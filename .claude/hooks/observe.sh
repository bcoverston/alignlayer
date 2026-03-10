#!/bin/bash
# AlignLayer observer hook — forwards Claude Code tool events to the ML server.
# Runs curl in foreground with a short timeout. Hook timeout (3s) > curl timeout (2s).
# Always exits 0 so the agent is never affected.

ALIGNLAYER_URL="${ALIGNLAYER_URL:-http://localhost:8000}"

curl -s -X POST "$ALIGNLAYER_URL/observe" \
  -H "Content-Type: application/json" \
  -d @- \
  --connect-timeout 1 \
  --max-time 2 \
  >/dev/null 2>&1

exit 0
