#!/usr/bin/env bash
for snap in $(restic -r /Users/lorenzo/Sync/backup/restic/data/ snapshots --json | jq -r '.[].short_id'); do
  echo "🔍 Checking snapshot $snap..."
  if restic -r /Users/lorenzo/Sync/backup/restic/data/ ls "$snap" /Users/lorenzo/Sync/Source/AI/autogen/$1 | grep -q -v '^snapshot '; then
    echo "✅ Files found in snapshot $snap:"
    restic -r /Users/lorenzo/Sync/backup/restic/data/ ls "$snap" /Users/lorenzo/Sync/Source/AI/autogen/$1
    break
  fi
done

