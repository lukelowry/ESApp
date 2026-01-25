"""
Example: GetSubData - Retrieving nested SubData from PowerWorld objects.

SubData sections store hierarchical data not available through CSV exports:
- BidCurve: Generator/Load cost curves (MW, $/MWh)
- ReactiveCapability: Generator Q limits by MW output (MW, MinMVAR, MaxMVAR)
- CTGElement: Contingency element definitions (actions)
- InterfaceElement: Interface branch membership
- SuperAreaArea: Areas within a super area
- ColorPoint: Contour color breakpoints
- Line: Polyline coordinates for background objects
"""

from esapp.saw import SAW

case_path = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Cases\Hawaii 37\Hawaii40_20231026.pwb"
saw = SAW(case_path)

# -----------------------------------------------------------------------------
# Example 1: Generator Cost Curves (BidCurve) and Reactive Capability
# -----------------------------------------------------------------------------
print("=" * 60)
print("GENERATORS: BidCurve + ReactiveCapability")
print("=" * 60)

df = saw.GetSubData("Gen", ["BusNum", "BusName", "GenID", "GenMW", "GenMWMax"],
                    ["BidCurve", "ReactiveCapability"])

for _, row in df.iterrows():
    print(f"\nGen @ Bus {row['BusNum']} ({row['BusName']}) ID={row['GenID']}")
    print(f"  Output: {row['GenMW']} MW (Max: {row['GenMWMax']})")

    if row["BidCurve"]:
        print(f"  BidCurve: {len(row['BidCurve'])} points")
        for mw, price in row["BidCurve"]:
            print(f"    {mw:>8} MW @ ${price}/MWh")

    if row["ReactiveCapability"]:
        print(f"  ReactiveCapability: {len(row['ReactiveCapability'])} points")
        for mw, qmin, qmax in row["ReactiveCapability"]:
            print(f"    {mw:>8} MW: Q=[{qmin}, {qmax}] MVAR")

# -----------------------------------------------------------------------------
# Example 2: Load Benefit Curves
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("LOADS: BidCurve (Benefit Curves)")
print("=" * 60)

df = saw.GetSubData("Load", ["BusNum", "LoadID", "LoadMW"], ["BidCurve"])
for _, row in df.iterrows():
    if row["BidCurve"]:
        print(f"Load @ Bus {row['BusNum']} ID={row['LoadID']}: {len(row['BidCurve'])} bid points")

# -----------------------------------------------------------------------------
# Example 3: Contingency Elements
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CONTINGENCIES: CTGElement (Actions)")
print("=" * 60)

df = saw.GetSubData("Contingency", ["TSContingency", "CTGSkip"], ["CTGElement"])
for _, row in df.iterrows():
    print(f"\nContingency: {row['TSContingency']} (Skip={row['CTGSkip']})")
    if row["CTGElement"]:
        for elem in row["CTGElement"][:5]:  # Show first 5 elements
            print(f"  {' '.join(elem)}")
        if len(row["CTGElement"]) > 5:
            print(f"  ... and {len(row['CTGElement']) - 5} more elements")

# -----------------------------------------------------------------------------
# Example 4: Interface Elements
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("INTERFACES: InterfaceElement")
print("=" * 60)

df = saw.GetSubData("Interface", ["InterfaceName", "InterfaceMW"], ["InterfaceElement"])
for _, row in df.iterrows():
    if row["InterfaceElement"]:
        print(f"Interface '{row['InterfaceName']}': {len(row['InterfaceElement'])} elements, MW={row['InterfaceMW']}")

# -----------------------------------------------------------------------------
# Example 5: Super Areas
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUPER AREAS: SuperAreaArea")
print("=" * 60)

df = saw.GetSubData("SuperArea", ["SuperAreaName", "SuperAreaNum"], ["SuperAreaArea"])
for _, row in df.iterrows():
    if row["SuperAreaArea"]:
        areas = [a[0] for a in row["SuperAreaArea"]]
        print(f"SuperArea '{row['SuperAreaName']}': Areas {areas}")

# -----------------------------------------------------------------------------
# Example 6: Using Filters
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FILTERED: Generators in Area 1 only")
print("=" * 60)

df = saw.GetSubData("Gen", ["BusNum", "GenID", "GenMW"], ["BidCurve"], filter_name="AreaNum=1")
print(f"Found {len(df)} generators in Area 1")

saw.CloseCase()
print("\nDone.")
