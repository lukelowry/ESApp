"""Time Step Simulation specific functions."""
from typing import List, Union
from ._helpers import format_list
from ._enums import FilterKeyword, format_filter


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
        return self._run_script("TimeStepDoRun", start_time or None, end_time or None)

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
        return self._run_script("TimeStepDoSinglePoint", time_point)

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
        return self._run_script("TimeStepClearResults", start_time or None, end_time or None)

    def TimeStepDeleteAll(self):
        """
        Deletes all time points.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepDeleteAll")

    def TimeStepResetRun(self):
        """
        Resets the run to the beginning.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepResetRun")

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
        return self._run_script("TimeStepAppendPWW", f'"{filename}"', f'"{solution_type}"')

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
        return self._run_script("TimeStepAppendPWWRange", f'"{filename}"', start_time, end_time, f'"{solution_type}"')

    def TimeStepAppendPWWRangeLatLon(self, filename: str, start_time: str, end_time: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, solution_type: str = "Single Solution"):
        """
        Appends a range of timepoints with geographic filtering.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepAppendPWWRangeLatLon", f'"{filename}"', start_time, end_time, min_lat, max_lat, min_lon, max_lon, f'"{solution_type}"')

    def TimeStepLoadB3D(self, filename: str, solution_type: str = "GIC Only (No Power Flow)"):
        """
        Loads a B3D file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepLoadB3D", f'"{filename}"', f'"{solution_type}"')

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
        return self._run_script("TimeStepLoadPWW", f'"{filename}"', f'"{solution_type}"')

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
        return self._run_script("TimeStepLoadPWWRange", f'"{filename}"', start_time, end_time, f'"{solution_type}"')

    def TimeStepLoadPWWRangeLatLon(self, filename: str, start_time: str, end_time: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, solution_type: str = "Single Solution"):
        """
        Loads a range of timepoints with geographic filtering.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepLoadPWWRangeLatLon", f'"{filename}"', start_time, end_time, min_lat, max_lat, min_lon, max_lon, f'"{solution_type}"')

    def TimeStepSavePWW(self, filename: str):
        """
        Saves existing weather data to a PWW file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepSavePWW", f'"{filename}"')

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
        return self._run_script("TimeStepSaveResultsByTypeCSV", object_type, f'"{filename}"', start_time or None, end_time or None)

    def TimeStepSavePWWRange(self, filename: str, start_time: str, end_time: str):
        """
        Saves a range of weather data to a PWW file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepSavePWWRange", f'"{filename}"', start_time, end_time)

    def TIMESTEPSaveSelectedModifyStart(self):
        """
        Starts modification of selected objects for saving.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TIMESTEPSaveSelectedModifyStart")

    def TIMESTEPSaveSelectedModifyFinish(self):
        """
        Finishes modification of selected objects for saving.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TIMESTEPSaveSelectedModifyFinish")

    def TIMESTEPSaveInputCSV(self, filename: str, field_list: List[str], start_time: str = "", end_time: str = ""):
        """
        Saves input fields to CSV.

        Returns
        -------
        str
            The result of the script command.
        """
        fields = format_list(field_list)
        return self._run_script("TIMESTEPSaveInputCSV", f'"{filename}"', fields, start_time, end_time)

    def TimeStepSaveFieldsSet(self, object_type: str, field_list: List[str], filter_name: Union[FilterKeyword, str] = FilterKeyword.ALL):
        """
        Sets fields to save during simulation.

        Parameters
        ----------
        object_type : str
            Object type.
        field_list : List[str]
            List of fields.
        filter_name : Union[FilterKeyword, str], optional
            Filter to apply. Defaults to FilterKeyword.ALL.

        Returns
        -------
        str
            The result of the script command.
        """
        fields = format_list(field_list)
        filt = format_filter(filter_name)
        return self._run_script("TimeStepSaveFieldsSet", object_type, fields, filt)

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
        objs = format_list(object_types) if object_types else ""
        return self._run_script("TimeStepSaveFieldsClear", objs)

    def TimeStepLoadTSB(self, filename: str):
        """
        Loads a TSB file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepLoadTSB", f'"{filename}"')

    def TimeStepSaveTSB(self, filename: str):
        """
        Saves a TSB file.

        Returns
        -------
        str
            The result of the script command.
        """
        return self._run_script("TimeStepSaveTSB", f'"{filename}"')
