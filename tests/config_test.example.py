"""
Test configuration template for ESA++ tests.

Copy this file to 'config_test.py' and update with your local settings.
The config_test.py file is gitignored so you can safely store your paths.

Usage:
    1. Copy this file: cp config_test.example.py config_test.py
    2. Edit config_test.py with your PowerWorld case path
    3. Run tests normally with pytest or VS Code test extension
"""

# Path to PowerWorld case file for integration tests
# Set to None to skip online tests
SAW_TEST_CASE = r"C:\Path\To\Your\Case.pwb"

# Alternative: Use None to always skip online tests
# SAW_TEST_CASE = None
