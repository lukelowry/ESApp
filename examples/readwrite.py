from gridwb import *

CASE_PATH = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Cases\Hawaii 37\Hawaii40_20231026.pwb"


wb = GridWorkBench(CASE_PATH)

# Get Data
DATA = wb[Substation]

# Confirm
print(DATA)