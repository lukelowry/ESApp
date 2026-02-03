"""Weather related functions."""
from typing import List

from ._enums import YesNo
from ._helpers import format_list


class WeatherMixin:
    """Mixin for Weather functions."""

    def WeatherLimitsGenUpdate(self, update_max: bool = True, update_min: bool = True):
        """
        Update generator MW limits based on weather data.

        Updates generator MW limits using weather limit curves and weather
        station temperature data. This allows for temperature-dependent
        generator capacity modeling.

        This is a wrapper for the ``WeatherLimitsGenUpdate`` script command.

        Parameters
        ----------
        update_max : bool, optional
            If True, updates the Max MW limit based on the calculated
            weather-dependent MWMax limit. Defaults to True.
        update_min : bool, optional
            If True, updates the Min MW limit based on the calculated
            weather-dependent MWMin limit. Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        umax = YesNo.from_bool(update_max)
        umin = YesNo.from_bool(update_min)
        return self._run_script("WeatherLimitsGenUpdate", umax, umin)

    def TemperatureLimitsBranchUpdate(
        self, rating_set_precedence: str = "NORMAL", normal_rating_set: str = "DEFAULT", ctg_rating_set: str = "DEFAULT"
    ):
        """
        Update branch limits based on temperature limit curves.

        Updates branch thermal limits using temperature limit curves and
        weather station temperature data. This allows for dynamic line
        rating based on ambient conditions.

        This is a wrapper for the ``TemperatureLimitsBranchUpdate`` script command.

        Parameters
        ----------
        rating_set_precedence : str, optional
            Determines which rating set takes precedence when the same rating
            set is specified for both normal and CTG curves. Valid values are
            "NORMAL", "CTG", or blank (same as "NORMAL"). Defaults to "NORMAL".
        normal_rating_set : str, optional
            Which limit to update with the normal temperature-dependent limit.
            Valid values are "DEFAULT" (uses Limit Monitoring Settings),
            "NO" (don't update), or "A" through "O". Defaults to "DEFAULT".
        ctg_rating_set : str, optional
            Which limit to update with the contingency temperature-dependent
            limit. Valid values are "DEFAULT", "NO", or "A" through "O".
            Defaults to "DEFAULT".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self._run_script("TemperatureLimitsBranchUpdate", rating_set_precedence, normal_rating_set, ctg_rating_set)

    def WeatherPFWModelsSetInputs(self):
        """
        Set inputs for all case PFWModels without applying them.

        Sets the inputs for all Power Flow Weather (PFW) models in the case,
        but does not apply them to the power flow case. Usually these inputs
        require the availability of weather measurements.

        This is a wrapper for the ``WeatherPFWModelsSetInputs`` script command.

        Returns
        -------
        str
            The response from the PowerWorld script command.

        See Also
        --------
        WeatherPFWModelsSetInputsAndApply : Sets inputs and applies to case.
        """
        return self._run_script("WeatherPFWModelsSetInputs")

    def WeatherPFWModelsSetInputsAndApply(self, solve_pf: bool = True):
        """
        Set inputs for PFWModels and apply them to the case.

        Sets the inputs for all Power Flow Weather (PFW) models and applies
        them to the power flow case. Usually these inputs require the
        availability of weather measurements, which can be loaded using
        ``WeatherPWWLoadForDateTimeUTC``.

        When PFWModels are applied, some case values may be changed (e.g.,
        generator MaxMW fields). Use ``WeatherPFWModelsRestoreDesignValues``
        to restore these values to the design values.

        This is a wrapper for the ``WeatherPFWModelsSetInputsAndApply`` script command.

        Parameters
        ----------
        solve_pf : bool, optional
            If True, solves the power flow using the default method after
            applying inputs. If False, you can call another solution command
            (e.g., SolvePowerFlow or SolvePrimalLP). Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.

        See Also
        --------
        WeatherPFWModelsRestoreDesignValues : Restores values changed by this method.
        WeatherPWWLoadForDateTimeUTC : Loads weather data for a specific time.
        """
        spf = YesNo.from_bool(solve_pf)
        return self._run_script("WeatherPFWModelsSetInputsAndApply", spf)

    def WeatherPWWFileAllMeasValid(self, filename: str, field_list: List[str], start_time: str = "", end_time: str = ""):
        """
        Check if a PWW file has valid measurements for specified fields.

        Returns true if the specified PWW file: 1) has all the specified
        fields, and 2) all the measurements for those fields are valid.
        This command only works with version 2 or greater PWW files.

        This is a wrapper for the ``WeatherPWWFileAllMeasValid`` script command.

        Parameters
        ----------
        filename : str
            The path to the PWW file to check.
        field_list : List[str]
            List of fields to check. At least one field must be provided.
            Valid fields include: TEMP, DEWPOINT, WINDSPEED, WINDSPEED100,
            GLOBALHORZIRRAD, DIRECTHORZIRRAD, WINDGUST, SMOKEVERTINT,
            PRECIPRATE, PRECIPPERCFROZEN.
        start_time : str, optional
            Start datetime in ISO8601 format. If provided, only returns
            true if the PWW file's starting datetime is at or before this.
            Defaults to "" (no start time check).
        end_time : str, optional
            End datetime in ISO8601 format. If provided, only returns true
            if the PWW file's ending datetime is at or after this.
            Defaults to "" (no end time check).

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        fields = format_list(field_list)
        return self._run_script("WeatherPWWFileAllMeasValid", f'"{filename}"', fields, start_time or None, end_time or None)

    def WeatherPFWModelsRestoreDesignValues(self):
        """
        Restore case values changed by PFWModels to their design values.

        Restores the case values (such as generator MaxMW fields) that were
        changed by ``WeatherPFWModelsSetInputsAndApply`` back to the design
        values specified with each PFWModel.

        This is a wrapper for the ``WeatherPFWModelsRestoreDesignValues`` script command.

        Returns
        -------
        str
            The response from the PowerWorld script command.

        See Also
        --------
        WeatherPFWModelsSetInputsAndApply : The method whose changes this restores.
        """
        return self._run_script("WeatherPFWModelsRestoreDesignValues")

    def WeatherPWWLoadForDateTimeUTC(self, iso_datetime: str):
        """
        Load weather data for a specific date and time.

        Loads weather data by searching the directory (and optionally
        subdirectories) set with ``WeatherPWWSetDirectory``.

        This is a wrapper for the ``WeatherPWWLoadForDateTimeUTC`` script command.

        Parameters
        ----------
        iso_datetime : str
            The desired date and time in ISO8601 format. This should be
            either a UTC value (e.g., "2024-03-06T18:00Z") or local time
            with time zone offset (e.g., "2024-03-06T12:00-06").

        Returns
        -------
        str
            The response from the PowerWorld script command.

        See Also
        --------
        WeatherPWWSetDirectory : Sets the directory to search for PWW files.
        """
        return self._run_script("WeatherPWWLoadForDateTimeUTC", f'"{iso_datetime}"')

    def WeatherPWWSetDirectory(self, directory: str, include_subdirs: bool = True):
        """
        Set the directory to search for PWW weather files.

        Specifies the directory (and optionally its subdirectories) to search
        when loading weather information from PWW files.

        This is a wrapper for the ``WeatherPWWSetDirectory`` script command.

        Parameters
        ----------
        directory : str
            Directory path that contains the PWW files.
        include_subdirs : bool, optional
            If True, includes subdirectories in the search path.
            Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.

        See Also
        --------
        WeatherPWWLoadForDateTimeUTC : Loads weather data using this directory.
        """
        sub = YesNo.from_bool(include_subdirs)
        return self._run_script("WeatherPWWSetDirectory", f'"{directory}"', sub)

    def WeatherPWWFileCombine2(self, source1: str, source2: str, dest: str):
        """
        Combine two PWW weather files into one.

        Merges two PWW files, provided they have the same weather stations
        and non-overlapping datetime ranges. The source2 file should be
        the second file chronologically.

        This is a wrapper for the ``WeatherPWWFileCombine2`` script command.

        Parameters
        ----------
        source1 : str
            Path to the first source file (first chronologically). Must exist.
            Should have ".pww" extension or no extension.
        source2 : str
            Path to the second source file (second chronologically). Must exist.
            Should have ".pww" extension or no extension.
        dest : str
            Path to the destination file. Does not need to exist and can be
            the same as either source file.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self._run_script("WeatherPWWFileCombine2", f'"{source1}"', f'"{source2}"', f'"{dest}"')

    def WeatherPWWFileGeoReduce(
        self, source: str, dest: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float
    ):
        """
        Reduce the geographic scope of a PWW file.

        Extracts weather data only for the geographic region bounded by the
        specified latitude and longitude coordinates. Rectangles spanning
        the international date line are not allowed.

        This is a wrapper for the ``WeatherPWWFileGeoReduce`` script command.

        Parameters
        ----------
        source : str
            Path to the source PWW file. Must exist. Should have ".pww"
            extension or no extension.
        dest : str
            Path to the destination PWW file. Does not need to exist and
            can be the same as the source file.
        min_lat : float
            Minimum latitude for the bounding rectangle. Must be >= -90
            and less than max_lat.
        max_lat : float
            Maximum latitude for the bounding rectangle. Must be <= 90
            and greater than min_lat.
        min_lon : float
            Minimum longitude for the bounding rectangle. Must be >= -180
            and less than max_lon.
        max_lon : float
            Maximum longitude for the bounding rectangle. Must be <= 180
            and greater than min_lon.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self._run_script("WeatherPWWFileGeoReduce", f'"{source}"', f'"{dest}"', min_lat, max_lat, min_lon, max_lon)
