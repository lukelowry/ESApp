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

    wb.io.esa.CalculateGIC(max_field=1.0, direction=90.0)
    
    xfmr_gic = wb.io.esa.GetParametersMultipleElement("GICWinding", ["BusNum", "BusNum:1", "GICAmps"])
    print("\nTransformer GIC Currents:")
    print(xfmr_gic.head())

    max_gic_row = xfmr_gic.loc[xfmr_gic['GICAmps'].idxmax()]
    print(f"\nMax GIC: {max_gic_row['GICAmps']:.2f} A at Bus {max_gic_row['BusNum']}")

    max_amps = []
    for angle in range(0, 360, 45):
        wb.io.esa.CalculateGIC(max_field=1.0, direction=angle)
        currents = wb.io.esa.GetParametersMultipleElement("GICWinding", ["GICAmps"])['GICAmps']
        max_amps.append({'Angle': angle, 'MaxGIC': currents.max()})
    
    import pandas as pd
    sweep_df = pd.DataFrame(max_amps)
    print(sweep_df)

    worst_angle = sweep_df.loc[sweep_df['MaxGIC'].idxmax()]
    print(f"\nWorst-case field direction: {worst_angle['Angle']} degrees (Max GIC: {worst_angle['MaxGIC']:.2f} A)")