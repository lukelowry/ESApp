"""
GIC Analysis
============

Calculating Geomagnetically Induced Currents (GIC) for a uniform electric field.
"""

from gridwb import GridWorkBench
import os

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # 1. Calculate GIC for a 1 V/km field pointing East (90 degrees)
    print("Calculating GIC (1 V/km, 90 deg)...")
    wb.func.calculate_gic(max_field=1.0, direction=90.0)
    
    # 2. Retrieve transformer GICs
    xfmr_gic = wb.gic.gictool().gicxfmrs[['BusNum', 'BusNum:1', 'GICAmps']]
    print("\nTransformer GIC Currents:")
    print(xfmr_gic.head())

    # 3. Find the transformer with the highest GIC
    max_gic_row = xfmr_gic.loc[xfmr_gic['GICAmps'].idxmax()]
    print(f"\nMax GIC: {max_gic_row['GICAmps']:.2f} A on transformer {max_gic_row['BusNum']}-{max_gic_row['BusNum:1']}")

    # 4. Perform a GIC sweep (varying direction)
    print("\nPerforming GIC direction sweep...")
    max_amps = []
    for angle in range(0, 360, 45):
        wb.func.calculate_gic(max_field=1.0, direction=angle)
        currents = wb.gic.gictool().gicxfmrs['GICAmps']
        max_amps.append({'Angle': angle, 'MaxGIC': currents.max()})
    
    import pandas as pd
    sweep_df = pd.DataFrame(max_amps)
    print(sweep_df)

    # 5. Identify the "worst-case" angle
    worst_angle = sweep_df.loc[sweep_df['MaxGIC'].idxmax()]
    print(f"\nWorst-case field direction: {worst_angle['Angle']} degrees (Max GIC: {worst_angle['MaxGIC']:.2f} A)")