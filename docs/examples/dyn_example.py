"""
Transient Stability Simulation Example
======================================

This example demonstrates how to run a transient stability simulation
using the ESA++ dynamics module.

The TS class provides comprehensive intellisense for all available
transient stability result fields, organized by object type.

Key Features Demonstrated:
- Setting simulation runtime
- Watching specific fields during simulation
- Defining contingencies with the fluent API
- Listing available dynamic models
- Running simulations and plotting results

Usage:
    Update 'case_path' to point to your PowerWorld case file,
    then run this script.
"""
import ast
import os
from esapp import GridWorkBench, TS
from esapp.components import Bus, Gen


# Load case path from file or specify directly
case_txt = os.path.join(os.path.dirname(__file__), 'case.txt')
if os.path.exists(case_txt):
    with open(case_txt, 'r') as f:
        case_path = ast.literal_eval(f.read().strip())
else:
    # Specify your case path here if case.txt doesn't exist
    case_path = "path/to/your/case.pwb"

wb = GridWorkBench(case_path)

# Set the simulation runtime (seconds)
wb.dyn.runtime = 10.0

# Watch generator fields during simulation
# TS provides IDE autocomplete for all transient stability result fields
wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.W, TS.Gen.Delta])

# Define a bus fault contingency using the fluent API
(wb.dyn.contingency("Fault_Bus1")
       .at(1.0).fault_bus("1")        # Apply 3-phase fault at t=1.0s
       .at(1.153).clear_fault("1"))   # Clear fault at t=1.153s (approx 9 cycles)

# List all dynamic models in the case
print("Dynamic Models in Case:")
print(wb.dyn.list_models())

# Run the simulation and plot results
# Uncomment the lines below to execute the simulation
# meta, results = wb.dyn.solve("Fault_Bus1")
# wb.dyn.plot(meta, results)
