"""Pytest configuration: ensure project root is on sys.path.

This allows test files in tests/ to import generate.py and the mage_mlx
package without needing to install the project as a package.
"""
import os
import sys

# Insert project root at the front of sys.path so that `import generate`
# and `from mage_mlx.worker import ...` work from tests/ subdirectories.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
