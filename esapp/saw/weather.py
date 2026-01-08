"""Weather related functions."""
from typing import List

class WeatherMixin:
    """Mixin for Weather functions."""

    def WeatherLimitsGenUpdate(self, update_max: bool = True, update_min: bool = True):
        """Updates generator MW limits based on weather data."""
        umax = "YES" if update_max else "NO"
        umin = "YES" if update_min else "NO"
        return self.RunScriptCommand(f"WeatherLimitsGenUpdate({umax}, {umin});")

    def TemperatureLimitsBranchUpdate(
        self, rating_set_precedence: str = "NORMAL", normal_rating_set: str = "DEFAULT", ctg_rating_set: str = "DEFAULT"
    ):
        """Updates branch limits based on temperature."""
        return self.RunScriptCommand(
            f"TemperatureLimitsBranchUpdate({rating_set_precedence}, {normal_rating_set}, {ctg_rating_set});"
        )

    def WeatherPFWModelsSetInputs(self):
        """Sets inputs for PFWModels."""
        return self.RunScriptCommand("WeatherPFWModelsSetInputs;")

    def WeatherPFWModelsSetInputsAndApply(self, solve_pf: bool = True):
        """Sets inputs for PFWModels and applies them to the case."""
        spf = "YES" if solve_pf else "NO"
        return self.RunScriptCommand(f"WeatherPFWModelsSetInputsAndApply({spf});")

    def WeatherPWWFileAllMeasValid(self, filename: str, field_list: List[str], start_time: str = "", end_time: str = ""):
        """Checks if PWW file has valid measurements."""
        fields = "[" + ", ".join(field_list) + "]"
        return self.RunScriptCommand(f'WeatherPWWFileAllMeasValid("{filename}", {fields}, {start_time}, {end_time});')

    def WeatherPFWModelsRestoreDesignValues(self):
        """Restores case values changed by WeatherPFWModels."""
        return self.RunScriptCommand("WeatherPFWModelsRestoreDesignValues;")

    def WeatherPWWLoadForDateTimeUTC(self, iso_datetime: str):
        """Loads weather for a specific date and time."""
        return self.RunScriptCommand(f'WeatherPWWLoadForDateTimeUTC("{iso_datetime}");')

    def WeatherPWWSetDirectory(self, directory: str, include_subdirs: bool = True):
        """Sets the directory to search for PWW files."""
        sub = "YES" if include_subdirs else "NO"
        return self.RunScriptCommand(f'WeatherPWWSetDirectory("{directory}", {sub});')

    def WeatherPWWFileCombine2(self, source1: str, source2: str, dest: str):
        """Combines two PWW files."""
        return self.RunScriptCommand(f'WeatherPWWFileCombine2("{source1}", "{source2}", "{dest}");')

    def WeatherPWWFileGeoReduce(
        self, source: str, dest: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ):
        """Reduces the geographic scope of a PWW file."""
        return self.RunScriptCommand(
            f'WeatherPWWFileGeoReduce("{source}", "{dest}", {min_lat}, {max_lat}, {min_lon}, {max_lon});'
        )