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

case_path = os.environ.get("SAW_TEST_CASE", "case.pwb")

if os.path.exists(case_path):
    wb = GridWorkBench(case_path)

    report_path = os.path.abspath("system_health_report.csv")
    wb.io.esa.SaveDataWithExtra(
        report_path, 
        "CSVCOLHEADER", 
        "Bus", 
        ["BusNum", "BusName", "BusPUVolt", "BusAngle"], 
        subdatalist=[],
        filter_name="", 
        sortfieldlist=[],
        header_list=["Report_Generated_By", "Project_ID"],
        header_value_list=["ESA++_Automator", "TAMU_RESEARCH_2026"]
    )
    print(f"Report saved to: {report_path}")

    try:
        wb.io.esa.SendToExcel(
            "Branch", 
            ["BusNum", "BusNum:1", "LineCircuit", "LineMVA", "LineLimit", "LinePercent"],
            filter_name="",
            use_column_headers=True,
            workbook="GridAnalysis.xlsx",
            worksheet="BranchLoading"
        )
        print("Excel export successful. File 'GridAnalysis.xlsx' created.")
    except Exception as e:
        print(f"Excel export failed: {e}")

else:
    print(f"Case file not found at {case_path}. Please set SAW_TEST_CASE environment variable.")