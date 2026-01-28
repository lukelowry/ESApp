"""
Transient Stability Simulation Example
======================================

This example demonstrates how to run a transient stability simulation
using the ESApp dynamics module.

The TS class provides comprehensive intellisense for all available
transient stability result fields, organized by object type.
"""
import ast
from esapp import GridWorkBench, TS
from esapp.grid import Bus, Gen

with open(r'C:\Users\wyattluke.lowery\Documents\GitHub\ESAplus\docs\examples\case.txt', 'r') as f:
    case_path = ast.literal_eval(f.read().strip())

wb = GridWorkBench(case_path)

# Set the simulation runtime (seconds)
wb.dyn.runtime = 10.0

wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

(wb.dyn.contingency("Fault_Bus5")
       .at(1.0).fault_bus("1")        # Apply 3-phase fault at t=1.0s
       .at(1.153).clear_fault("1"))   # Clear fault at t=1.153s

meta, results = wb.dyn.solve("Fault_Bus5")
wb.dyn.plot(meta, results)
