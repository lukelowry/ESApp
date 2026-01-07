import pytest
import sys
import os

if __name__ == "__main__":
    # Centralized case path definition. 
    if "SAW_TEST_CASE" not in os.environ:
        os.environ["SAW_TEST_CASE"] = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Cases\Hawaii 37\Hawaii40_20231026.pwb"

    # Add the project root to sys.path to ensure imports work correctly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Run pytest
    sys.exit(pytest.main(["-v", "tests"]))
