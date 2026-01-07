"""
Advanced Reporting and Excel Integration
========================================

This example demonstrates how to use the Adapter to generate custom reports
and export data directly to Microsoft Excel, leveraging PowerWorld's 
built-in reporting actions.
"""

from gridwb import GridWorkBench
from gridwb.grid.components import Bus, Branch
import os

# Use the test case from environment or a dummy path
case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    # 1. Create a custom report using SaveDataWithExtra
    # This allows adding custom headers and values (like timestamps or analyst names)
    print("Generating custom CSV report with extra headers...")
    report_path = os.path.abspath("system_health_report.csv")
    
    # We use the underlying SAW object for specialized AUX actions
    wb.esa.SaveDataWithExtra(
        report_path, 
        "CSVCOLHEADER", 
        "Bus", 
        ["BusNum", "BusName", "BusPUVolt", "BusAngle"], 
        subdatalist=[],
        filter_name="", 
        sort_field_list=[],
        header_list=["Report_Generated_By", "Project_ID"],
        header_value_list=["ESA++_Automator", "TAMU_RESEARCH_2026"]
    )
    print(f"Report saved to: {report_path}")

    # 2. Export directly to Excel
    # This uses the SimAuto SendToExcel command to populate a spreadsheet
    print("\nExporting branch loading data to Excel...")
    try:
        # Note: This requires Excel to be installed on the machine
        wb.esa.SendToExcel(
            "Branch", 
            ["BusNum", "BusNum:1", "LineCircuit", "LineMVA", "LineLimit", "LinePercent"],
            filter_name="",
            use_column_headers=True,
            workbookname="GridAnalysis.xlsx",
            worksheetname="BranchLoading"
        )
        print("Excel export successful. File 'GridAnalysis.xlsx' created.")
    except Exception as e:
        print(f"Excel export failed (this is expected if Excel is not installed): {e}")

else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")