"""Time Step Simulation specific functions."""
from typing import List


class TimeStepMixin:
    """Mixin for Time Step Simulation functions."""

    def TimeStepDoRun(self, start_time: str = "", end_time: str = ""):
        """Solves the Time Step Simulation.

        :param start_time: ISO8601 start time (e.g. '2025-06-01T00:00:00-05:00').
        :param end_time: ISO8601 end time.
        """
        args = ""
        if start_time and end_time:
            args = f"{start_time}, {end_time}"
        return self.RunScriptCommand(f"TimeStepDoRun({args});")

    def TimeStepDoSinglePoint(self, time_point: str):
        """Solves the Time Step Simulation for a single point.

        :param time_point: ISO8601 date time.
        """
        return self.RunScriptCommand(f"TimeStepDoSinglePoint({time_point});")

    def TimeStepClearResults(self, start_time: str = "", end_time: str = ""):
        """Clears Time Step Simulation results."""
        args = ""
        if start_time and end_time:
            args = f"{start_time}, {end_time}"
        return self.RunScriptCommand(f"TimeStepClearResults({args});")

    def TimeStepDeleteAll(self):
        """Deletes all time points."""
        return self.RunScriptCommand("TimeStepDeleteAll;")

    def TimeStepResetRun(self):
        """Resets the run to the beginning."""
        return self.RunScriptCommand("TimeStepResetRun;")

    def TimeStepAppendPWW(self, filename: str, solution_type: str = "Single Solution"):
        """Appends a PWW file to the Time Step Simulation."""
        return self.RunScriptCommand(f'TimeStepAppendPWW("{filename}", "{solution_type}");')

    def TimeStepAppendPWWRange(self, filename: str, start_time: str, end_time: str, solution_type: str = "Single Solution"):
        """Appends a range of timepoints from a PWW file."""
        return self.RunScriptCommand(f'TimeStepAppendPWWRange("{filename}", {start_time}, {end_time}, "{solution_type}");')

    def TimeStepAppendPWWRangeLatLon(self, filename: str, start_time: str, end_time: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, solution_type: str = "Single Solution"):
        """Appends a range of timepoints with geographic filtering."""
        return self.RunScriptCommand(f'TimeStepAppendPWWRangeLatLon("{filename}", {start_time}, {end_time}, {min_lat}, {max_lat}, {min_lon}, {max_lon}, "{solution_type}");')

    def TimeStepLoadB3D(self, filename: str, solution_type: str = "GIC Only (No Power Flow)"):
        """Loads a B3D file."""
        return self.RunScriptCommand(f'TimeStepLoadB3D("{filename}", "{solution_type}");')

    def TimeStepLoadPWW(self, filename: str, solution_type: str = "Single Solution"):
        """Loads a PWW file into the Time Step Simulation.

        :param filename: Name of the PWW file.
        :param solution_type: Solution type string (e.g. 'OPF', 'SCOPF').
        """
        return self.RunScriptCommand(f'TimeStepLoadPWW("{filename}", "{solution_type}");')

    def TimeStepLoadPWWRange(
        self, filename: str, start_time: str, end_time: str, solution_type: str = "Single Solution"
    ):
        """Loads a range of timepoints from a PWW file."""
        return self.RunScriptCommand(
            f'TimeStepLoadPWWRange("{filename}", {start_time}, {end_time}, "{solution_type}");'
        )

    def TimeStepLoadPWWRangeLatLon(self, filename: str, start_time: str, end_time: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, solution_type: str = "Single Solution"):
        """Loads a range of timepoints with geographic filtering."""
        return self.RunScriptCommand(f'TimeStepLoadPWWRangeLatLon("{filename}", {start_time}, {end_time}, {min_lat}, {max_lat}, {min_lon}, {max_lon}, "{solution_type}");')

    def TimeStepSavePWW(self, filename: str):
        """Saves existing weather data to a PWW file."""
        return self.RunScriptCommand(f'TimeStepSavePWW("{filename}");')

    def TimeStepSaveResultsByTypeCSV(
        self, object_type: str, filename: str, start_time: str = "", end_time: str = ""
    ):
        """Saves results for a specific object type to CSV.

        :param object_type: Object type (e.g. 'GEN').
        :param filename: Output CSV filename.
        :param start_time: Optional start time.
        :param end_time: Optional end time.
        """
        args = f'{object_type}, "{filename}"'
        if start_time and end_time:
            args += f", {start_time}, {end_time}"
        return self.RunScriptCommand(f"TimeStepSaveResultsByTypeCSV({args});")

    def TimeStepSavePWWRange(self, filename: str, start_time: str, end_time: str):
        """Saves a range of weather data to a PWW file."""
        return self.RunScriptCommand(f'TimeStepSavePWWRange("{filename}", {start_time}, {end_time});')

    def TIMESTEPSaveSelectedModifyStart(self):
        """Starts modification of selected objects for saving."""
        return self.RunScriptCommand("TIMESTEPSaveSelectedModifyStart;")

    def TIMESTEPSaveSelectedModifyFinish(self):
        """Finishes modification of selected objects for saving."""
        return self.RunScriptCommand("TIMESTEPSaveSelectedModifyFinish;")

    def TIMESTEPSaveInputCSV(self, filename: str, field_list: List[str], start_time: str = "", end_time: str = ""):
        """Saves input fields to CSV."""
        fields = "[" + ", ".join(field_list) + "]"
        args = f'"{filename}", {fields}, {start_time}, {end_time}'
        return self.RunScriptCommand(f"TIMESTEPSaveInputCSV({args});")

    def TimeStepSaveFieldsSet(self, object_type: str, field_list: List[str], filter_name: str = "ALL"):
        """Sets fields to save during simulation.

        :param object_type: Object type.
        :param field_list: List of fields.
        :param filter_name: Filter to apply.
        """
        fields = "[" + ", ".join(field_list) + "]"
        filt = f'"{filter_name}"' if filter_name != "ALL" and filter_name != "SELECTED" else filter_name
        return self.RunScriptCommand(f"TimeStepSaveFieldsSet({object_type}, {fields}, {filt});")

    def TimeStepSaveFieldsClear(self, object_types: List[str] = None):
        """Clears save fields for object types.

        :param object_types: List of object types. If None, clears all.
        """
        objs = ""
        if object_types:
            objs = "[" + ", ".join(object_types) + "]"
        return self.RunScriptCommand(f"TimeStepSaveFieldsClear({objs});")

    def TimeStepLoadTSB(self, filename: str):
        """Loads a TSB file."""
        return self.RunScriptCommand(f'TimeStepLoadTSB("{filename}");')

    def TimeStepSaveTSB(self, filename: str):
        """Saves a TSB file."""
        return self.RunScriptCommand(f'TimeStepSaveTSB("{filename}");')