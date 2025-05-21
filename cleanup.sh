#!/bin/bash

# 1. Remove all __pycache__ directories (including subfolders)
find . -type d -name "__pycache__" -exec rm -r {} +

# 2. Remove all *.pyc, *.rsls, *.rslsf files
find . -type f \( -name "*.pyc" -o -name "*.rsls" -o -name "*.rslsf" \) -delete

echo "Cleanup complete!"
