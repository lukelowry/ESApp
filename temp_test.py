CASE_PATH = r"C:\Users\wyatt\OneDrive - Texas A&M University\Research\Cases\Hawaii\Hawaii40_20231026.pwb"

from esa import SAW


saw = SAW(FileName=CASE_PATH)

bus_data = saw.get_power_flow_results('bus')

G = saw.get_gmatrix()

saw.SolvePowerFlow()

# NOTE Jacobean returns error until after power flow is solved
J = saw.get_jacobian()

print("Jacobian Matrix:", J.shape, J.nnz)
print("G Matrix:", G.shape, G.nnz)