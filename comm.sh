#!/bin/bash

# Check if commit message is provided
if [ -z "$1" ]; then
  echo "❌ Error: Please provide a commit message."
  echo "Usage: ./git_commit.sh \"Your commit message here\""
  exit 1
fi

# Add .json and .py files
#git add **/*.py **/*.md **/*.json **/*.txt
source git_add_filtered.sh

# List modified files
git --no-pager diff --cached --name-only

# Commit with the provided message
git commit -a -m "$1"
