#!/bin/bash

set -e  # Exit immediately on error

rm -fr /Users/lorenzo/Sync/Source/AI/autogen/autogen_extensions/.venv

# 1. Go to base folder
cd /Users/lorenzo/Sync/Source/AI/autogen

# 2. Activate virtual environment
source .venv/bin/activate

uv pip uninstall autogen_extensions

# 3. Enter relative folder
cd autogen_extensions

# 4. Remove 'build' folder
rm -rf build

# 5. Remove all *.egg-info folders
find . -type d -name "*.egg-info" -exec rm -rf {} +

# 6. Install editable package
uv pip install -e .


# 7. Return to base folder
cd ..

#uv sync

# 8. Run pylint for undefined-variable errors only
pylint --ignore=.venv . -E --disable=all --enable=undefined-variable

# 9. Run ruff
ruff check . --ignore E501 --fix

# 10. Run pytest on tests
pytest -vvv ./tests --maxfail=3 --disable-warnings -q
