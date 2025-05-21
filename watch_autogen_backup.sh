#!/bin/bash
PROJECT_DIR="/Users/lorenzo/Sync/Source/AI/autogen"
SCRIPT="/usr/local/bin/restic_backup.sh"

while true; do
  "$SCRIPT" "$PROJECT_DIR"
  sleep 30
done

