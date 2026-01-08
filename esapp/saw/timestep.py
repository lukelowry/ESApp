"""Time Step Simulation specific functions."""
from typing import List


class TimeStepMixin:
    """Mixin for Time Step Simulation functions."""

    def TimeStepDoRun(self, start_time: str = "", end_time: str = ""):
        """
        Solves the Time Step Simulation.

        Parameters
        ----------
        start_time : str, optional
            ISO8601 start time (e.g. '2025-06-01T00:00:00-05:00').
        end_time : str, optional
            ISO8601 end time.

        Returns
        -------
        str
            The result of the script command.
        """
        args = ""
        if start_time and end_time:
            args = f"{start_time}, {end_time}"
        return self.RunScriptCommand(f"TimeStepDoRun({args});")

    def TimeStepDoSinglePoint(self, time_point: str):
        """
        Solves the Time Step Simulation for a single point.

        Parameters
        ----------
        time_point : str
            ISO8601 date time.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f"TimeStepDoSinglePoint({time_point});")

    def TimeStepClearResults(self, start_time: str = "", end_time: str = ""):
        """
        Clears Time Step Simulation results.

        Parameters
        ----------
        start_time : str, optional
            Start time of the range to clear.
        end_time : str, optional
            End time of the range to clear.

        Returns
        -------
        str
            The result of the script command.
        """
        args = ""
        if start_time and end_time:
            args = f"{start_time}, {end_time}"
        return self.RunScriptCommand(f"TimeStepClearResults({args});")

    def TimeStepDeleteAll(self):
        """
        Deletes all time points.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand("TimeStepDeleteAll;")

    def TimeStepResetRun(self):
        """
        Resets the run to the beginning.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand("TimeStepResetRun;")

    def TimeStepAppendPWW(self, filename: str, solution_type: str = "Single Solution"):
        """
        Appends a PWW file to the Time Step Simulation.

        Parameters
        ----------
        filename : str
            Path to the PWW file.
        solution_type : str, optional
            The solution type. Defaults to "Single Solution".

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepAppendPWW("{filename}", "{solution_type}");')

    def TimeStepAppendPWWRange(self, filename: str, start_time: str, end_time: str, solution_type: str = "Single Solution"):
        """
        Appends a range of timepoints from a PWW file.

        Parameters
        ----------
        filename : str
            Path to the PWW file.
        start_time : str
            Start time.
        end_time : str
            End time.
        solution_type : str, optional
            The solution type.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepAppendPWWRange("{filename}", {start_time}, {end_time}, "{solution_type}");')

    def TimeStepAppendPWWRangeLatLon(self, filename: str, start_time: str, end_time: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, solution_type: str = "Single Solution"):
        """
        Appends a range of timepoints with geographic filtering.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepAppendPWWRangeLatLon("{filename}", {start_time}, {end_time}, {min_lat}, {max_lat}, {min_lon}, {max_lon}, "{solution_type}");')

    def TimeStepLoadB3D(self, filename: str, solution_type: str = "GIC Only (No Power Flow)"):
        """
        Loads a B3D file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepLoadB3D("{filename}", "{solution_type}");')

    def TimeStepLoadPWW(self, filename: str, solution_type: str = "Single Solution"):
        """
        Loads a PWW file into the Time Step Simulation.

        Parameters
        ----------
        filename : str
            Name of the PWW file.
        solution_type : str, optional
            Solution type string (e.g. 'OPF', 'SCOPF').

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepLoadPWW("{filename}", "{solution_type}");')

    def TimeStepLoadPWWRange(
        self, filename: str, start_time: str, end_time: str, solution_type: str = "Single Solution"
    ):
        """
        Loads a range of timepoints from a PWW file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(
            f'TimeStepLoadPWWRange("{filename}", {start_time}, {end_time}, "{solution_type}");'
        )

    def TimeStepLoadPWWRangeLatLon(self, filename: str, start_time: str, end_time: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, solution_type: str = "Single Solution"):
        """
        Loads a range of timepoints with geographic filtering.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepLoadPWWRangeLatLon("{filename}", {start_time}, {end_time}, {min_lat}, {max_lat}, {min_lon}, {max_lon}, "{solution_type}");')

    def TimeStepSavePWW(self, filename: str):
        """
        Saves existing weather data to a PWW file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepSavePWW("{filename}");')

    def TimeStepSaveResultsByTypeCSV(
        self, object_type: str, filename: str, start_time: str = "", end_time: str = ""
    ):
        """
        Saves results for a specific object type to CSV.

        Parameters
        ----------
        object_type : str
            Object type (e.g. 'GEN').
        filename : str
            Output CSV filename.
        start_time : str, optional
            Optional start time.
        end_time : str, optional
            Optional end time.

        Returns
        -------
        str
            The result of the script command.
        """
        args = f'{object_type}, "{filename}"'
        if start_time and end_time:
            args += f", {start_time}, {end_time}"
        return self.RunScriptCommand(f"TimeStepSaveResultsByTypeCSV({args});")

    def TimeStepSavePWWRange(self, filename: str, start_time: str, end_time: str):
        """
        Saves a range of weather data to a PWW file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepSavePWWRange("{filename}", {start_time}, {end_time});')

    def TIMESTEPSaveSelectedModifyStart(self):
        """
        Starts modification of selected objects for saving.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand("TIMESTEPSaveSelectedModifyStart;")

    def TIMESTEPSaveSelectedModifyFinish(self):
        """
        Finishes modification of selected objects for saving.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand("TIMESTEPSaveSelectedModifyFinish;")

    def TIMESTEPSaveInputCSV(self, filename: str, field_list: List[str], start_time: str = "", end_time: str = ""):
        """
        Saves input fields to CSV.

        Returns
        -------
        str
            The result of the script command.
        """
        fields = "[" + ", ".join(field_list) + "]"
        args = f'"{filename}", {fields}, {start_time}, {end_time}'
        return self.RunScriptCommand(f"TIMESTEPSaveInputCSV({args});")

    def TimeStepSaveFieldsSet(self, object_type: str, field_list: List[str], filter_name: str = "ALL"):
        """
        Sets fields to save during simulation.

        Parameters
        ----------
        object_type : str
            Object type.
        field_list : List[str]
            List of fields.
        filter_name : str, optional
            Filter to apply. Defaults to "ALL".

        Returns
        -------
        str
            The result of the script command.
        """
        fields = "[" + ", ".join(field_list) + "]"
        filt = f'"{filter_name}"' if filter_name != "ALL" and filter_name != "SELECTED" else filter_name
        return self.RunScriptCommand(f"TimeStepSaveFieldsSet({object_type}, {fields}, {filt});")

    def TimeStepSaveFieldsClear(self, object_types: List[str] = None):
        """
        Clears save fields for object types.

        Parameters
        ----------
        object_types : List[str], optional
            List of object types. If None, clears all.

        Returns
        -------
        str
            The result of the script command.
        """
        objs = ""
        if object_types:
            objs = "[" + ", ".join(object_types) + "]"
        return self.RunScriptCommand(f"TimeStepSaveFieldsClear({objs});")

    def TimeStepLoadTSB(self, filename: str):
        """
        Loads a TSB file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepLoadTSB("{filename}");')

    def TimeStepSaveTSB(self, filename: str):
        """
        Saves a TSB file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand(f'TimeStepSaveTSB("{filename}");')