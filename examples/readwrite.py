from gridwb import *

CASE_PATH = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Cases\Hawaii 37\Hawaii40_20231026.pwb"


wb = GridWorkBench(CASE_PATH)

# Get Data
DATA = wb[Bus]

# Change Data
DATA.loc[2, "AreaNum"] = 2

# Write Data
wb.io[Bus] = DATA

DATA = wb[Bus]

# Confirm
print(DATA)