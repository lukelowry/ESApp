"""Weather related functions."""
from typing import List

class WeatherMixin:
    """Mixin for Weather functions."""

    def WeatherLimitsGenUpdate(self, update_max: bool = True, update_min: bool = True):
        """
        Updates generator MW limits based on weather data.

        Parameters
        ----------
        update_max : bool, optional
            If True, updates the maximum MW limit. Defaults to True.
        update_min : bool, optional
            If True, updates the minimum MW limit. Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        umax = "YES" if update_max else "NO"
        umin = "YES" if update_min else "NO"
        return self.RunScriptCommand(f"WeatherLimitsGenUpdate({umax}, {umin});")

    def TemperatureLimitsBranchUpdate(
        self, rating_set_precedence: str = "NORMAL", normal_rating_set: str = "DEFAULT", ctg_rating_set: str = "DEFAULT"
    ):
        """
        Updates branch limits based on temperature.

        Parameters
        ----------
        rating_set_precedence : str, optional
            Determines which rating set takes precedence. Defaults to "NORMAL".
        normal_rating_set : str, optional
            The rating set to use for normal operation. Defaults to "DEFAULT".
        ctg_rating_set : str, optional
            The rating set to use for contingency operation. Defaults to "DEFAULT".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(
            f"TemperatureLimitsBranchUpdate({rating_set_precedence}, {normal_rating_set}, {ctg_rating_set});"
        )

    def WeatherPFWModelsSetInputs(self):
        """
        Sets inputs for PFWModels.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("WeatherPFWModelsSetInputs;")

    def WeatherPFWModelsSetInputsAndApply(self, solve_pf: bool = True):
        """
        Sets inputs for PFWModels and applies them to the case.

        Parameters
        ----------
        solve_pf : bool, optional
            If True, solves the power flow after applying inputs. Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        spf = "YES" if solve_pf else "NO"
        return self.RunScriptCommand(f"WeatherPFWModelsSetInputsAndApply({spf});")

    def WeatherPWWFileAllMeasValid(self, filename: str, field_list: List[str], start_time: str = "", end_time: str = ""):
        """
        Checks if PWW file has valid measurements.

        Parameters
        ----------
        filename : str
            The path to the PWW file.
        field_list : List[str]
            List of fields to check.
        start_time : str, optional
            Start time for the validity check. Defaults to "".
        end_time : str, optional
            End time for the validity check. Defaults to "".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        fields = "[" + ", ".join(field_list) + "]"
        return self.RunScriptCommand(f'WeatherPWWFileAllMeasValid("{filename}", {fields}, {start_time}, {end_time});')

    def WeatherPFWModelsRestoreDesignValues(self):
        """
        Restores case values changed by WeatherPFWModels.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("WeatherPFWModelsRestoreDesignValues;")

    def WeatherPWWLoadForDateTimeUTC(self, iso_datetime: str):
        """
        Loads weather for a specific date and time.

        Parameters
        ----------
        iso_datetime : str
            The date and time in ISO format (UTC).

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(f'WeatherPWWLoadForDateTimeUTC("{iso_datetime}");')

    def WeatherPWWSetDirectory(self, directory: str, include_subdirs: bool = True):
        """
        Sets the directory to search for PWW files.

        Parameters
        ----------
        directory : str
            The directory path.
        include_subdirs : bool, optional
            If True, includes subdirectories in the search. Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        sub = "YES" if include_subdirs else "NO"
        return self.RunScriptCommand(f'WeatherPWWSetDirectory("{directory}", {sub});')

    def WeatherPWWFileCombine2(self, source1: str, source2: str, dest: str):
        """
        Combines two PWW files.

        Parameters
        ----------
        source1 : str
            Path to the first source file.
        source2 : str
            Path to the second source file.
        dest : str
            Path to the destination file.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(f'WeatherPWWFileCombine2("{source1}", "{source2}", "{dest}");')

    def WeatherPWWFileGeoReduce(
        self, source: str, dest: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ):
        """
        Reduces the geographic scope of a PWW file.

        Parameters
        ----------
        source : str
            Path to the source PWW file.
        dest : str
            Path to the destination PWW file.
        min_lat : float
            Minimum latitude.
        max_lat : float
            Maximum latitude.
        min_lon : float
            Minimum longitude.
        max_lon : float
            Maximum longitude.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(
            f'WeatherPWWFileGeoReduce("{source}", "{dest}", {min_lat}, {max_lat}, {min_lon}, {max_lon});'
        )