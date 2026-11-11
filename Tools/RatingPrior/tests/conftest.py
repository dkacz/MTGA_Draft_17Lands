import os
import sys

# The tool is a script folder, not a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
