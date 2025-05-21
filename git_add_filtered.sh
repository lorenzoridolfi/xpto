#!/bin/bash

# Enable globstar in case you need it later (not used in this script but handy)
shopt -s globstar

# Find and add files while excluding .venv
find . -type d -name ".venv" -prune -o \
  -type f \( \
    -name "*.py" -o \
    -name "*.md" -o \
    -name "*.json" -o \
    -name "*.txt" -o \
    -name "*.yaml" -o \
    -name "*.yml" \
  \) -print -exec git add {} +

