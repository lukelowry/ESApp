import pytest
import sys
import os

if __name__ == "__main__":
    # Add the project root to sys.path to ensure imports work correctly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run pytest
    sys.exit(pytest.main(["-v", "tests"]))
