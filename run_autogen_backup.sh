#!/bin/bash
nohup "/Users/lorenzo/Sync/Source/AI/autogen/watch_autogen_backup.sh" > "/Users/lorenzo/Sync/Source/AI/autogen/autogen_backup.log" 2>&1 &
echo "Autogen backup running in background. Log: $HOME/autogen_backup.log"
