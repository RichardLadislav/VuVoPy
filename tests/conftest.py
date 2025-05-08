# tests/conftest.py
import sys
import os

# Compute absolute path to the top‐level src/ directory
HERE = os.path.dirname(__file__)
SRC = os.path.normpath(os.path.join(HERE, "..", "src"))

# Prepend it so that `import VuVoPy...` works in all test files
if SRC not in sys.path:
    sys.path.insert(0, SRC)
