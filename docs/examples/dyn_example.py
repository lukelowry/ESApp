import ast
from esapp import GridWorkBench
from esapp.grid import Bus, Gen, TSContingency, TSContingencyElement
from esapp.apps.dynamics import Dynamics

# Setup
with open(r'C:\Users\wyattluke.lowery\Documents\GitHub\ESAplus\docs\examples\case.txt', 'r') as f:
    case_path = ast.literal_eval(f.read().strip())
wb = GridWorkBench(case_path)


# 2. Configure Simulation
# We only need to define what to watch and the total runtime.
wb.dyn.runtime = 5.0
wb.dyn.watch(Bus, ['TSBusVPU', 'TSBusAng'])

# 3. Define Contingency
# The fluent API allows chaining. No manual upload step is required.
(wb.dyn.contingency("Fault_Bus5")
       .at(1.0).fault_bus("5")
       .at(1.0833).clear_fault("5"))

# 4. Run Simulation
# Pass the contingency name directly. The manager handles initialization and retrieval.
meta, results = wb.dyn.solve("Fault_Bus5")

# 5. Visualize
if not results.empty:
    print(f"Simulation complete: {len(results)} time steps retrieved.")
    wb.dyn.plot(meta, results)