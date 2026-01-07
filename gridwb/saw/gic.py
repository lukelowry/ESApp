"""Geomagnetically Induced Current (GIC) specific functions."""


class GICMixin:
    """Mixin for GIC analysis functions."""

    def CalculateGIC(self, max_field: float, direction: float, solve_pf: bool = True):
        """Calculates the 'Single Snapshot' GIC solution.

        :param max_field: Maximum Electric Field in Volts/km.
        :param direction: Storm Direction, Degrees from 0 to 360.
        :param solve_pf: Include GIC in the Power Flow.
        """
        spf = "YES" if solve_pf else "NO"
        return self.RunScriptCommand(f"GICCalculate({max_field}, {direction}, {spf});")

    def ClearGIC(self):
        """Clear GIC Values."""
        return self.RunScriptCommand("GICClear;")

    def GICLoad3DEfield(self, file_type: str, filename: str, setup_on_load: bool = True):
        """Loads GIC data including time varying fields.

        :param file_type: Type of file to be loaded (CSV, B3D, JSON, DAT).
        :param filename: Name of the file to be loaded.
        :param setup_on_load: Run procedure to setup time varying series after loading file.
        """
        sol = "YES" if setup_on_load else "NO"
        return self.RunScriptCommand(f'GICLoad3DEfield({file_type}, "{filename}", {sol});')

    def GICReadFilePSLF(self, filename: str):
        """Reads GIC supplemental data from a GMD text file format.

        :param filename: Name of the file to be loaded, with extension GMD.
        """
        return self.RunScriptCommand(f'GICReadFilePSLF("{filename}");')

    def GICReadFilePTI(self, filename: str):
        """Reads GIC supplemental data from a GIC text file format.

        :param filename: Name of the file to be loaded, with extension GIC.
        """
        return self.RunScriptCommand(f'GICReadFilePTI("{filename}");')

    def GICSaveGMatrix(self, gmatrix_filename: str, gmatrix_id_filename: str):
        """Save the GMatrix used with the GIC calculations.

        :param gmatrix_filename: File in which to save the G Matrix.
        :param gmatrix_id_filename: File to save a description of what each row and column represents.
        """
        return self.RunScriptCommand(f'GICSaveGMatrix("{gmatrix_filename}", "{gmatrix_id_filename}");')

    def GICSetupTimeVaryingSeries(self, start: float = 0.0, end: float = 0.0, delta: float = 0.0):
        """Creates a set of Branch series DC input voltages.

        :param start: Start Time Offset (seconds).
        :param end: End Time Offset (seconds).
        :param delta: Sampling Rate (seconds).
        """
        return self.RunScriptCommand(f"GICSetupTimeVaryingSeries({start}, {end}, {delta});")

    def GICShiftOrStretchInputPoints(
        self,
        lat_shift: float = 0.0,
        lon_shift: float = 0.0,
        mag_scalar: float = 1.0,
        stretch_scalar: float = 1.0,
        update_time_varying_series: bool = False,
    ):
        """Scales, shifts, or stretches the active set of Time Varying Electric Field Inputs.

        :param lat_shift: Latitude Shift in degrees.
        :param lon_shift: Longitude Shift in degrees.
        :param mag_scalar: E-Field Magnitude scalar.
        :param stretch_scalar: Geographic Stretch scalar.
        :param update_time_varying_series: Update the time varying voltage input values.
        """
        update = "YES" if update_time_varying_series else "NO"
        return self.RunScriptCommand(
            f"GICShiftOrStretchInputPoints({lat_shift}, {lon_shift}, {mag_scalar}, {stretch_scalar}, {update});"
        )

    def GICTimeVaryingCalculate(self, the_time: float, solve_pf: bool = True):
        """Calculate GIC Values using the 'Time-Varying Series Voltage Inputs' Calculation Mode.

        :param the_time: Current Time Offset from Reference (seconds).
        :param solve_pf: Include GIC in the Power Flow and Transient Stability.
        """
        spf = "YES" if solve_pf else "NO"
        return self.RunScriptCommand(f"GICTimeVaryingCalculate({the_time}, {spf});")

    def GICTimeVaryingAddTime(self, new_time: float):
        """Adds a new input values at specified time.

        :param new_time: New Time for new input values.
        """
        return self.RunScriptCommand(f"GICTimeVaryingAddTime({new_time});")

    def GICTimeVaryingDeleteAllTimes(self):
        """Delete All Input time varying voltage input values."""
        return self.RunScriptCommand("GICTimeVaryingDeleteAllTimes;")

    def GICTimeVaryingEFieldCalculate(self, the_time: float, solve_pf: bool = True):
        """Calculate GIC Values using the 'Time-Varying Electric Field Inputs' Calculation Mode.

        :param the_time: Current Time Offset from Reference (seconds).
        :param solve_pf: Include GIC in the Power Flow and Transient Stability.
        """
        spf = "YES" if solve_pf else "NO"
        return self.RunScriptCommand(f"GICTimeVaryingEFieldCalculate({the_time}, {spf});")

    def GICTimeVaryingElectricFieldsDeleteAllTimes(self):
        """Clear all the time varying electric field input values."""
        return self.RunScriptCommand("GICTimeVaryingElectricFieldsDeleteAllTimes;")

    def GICWriteFilePSLF(self, filename: str, use_filters: bool = False):
        """Writes GIC supplemental data from a GMD text file format.

        :param filename: Name of the file to be loaded, with extension GMD.
        :param use_filters: Use Area/Zone Filters.
        """
        uf = "YES" if use_filters else "NO"
        return self.RunScriptCommand(f'GICWriteFilePSLF("{filename}", {uf});')

    def GICWriteFilePTI(self, filename: str, use_filters: bool = False, version: int = 4):
        """Writes GIC supplemental data from a GIC text file format.

        :param filename: Name of the file to be loaded, with extension GIC.
        :param use_filters: Use Area/Zone Filters.
        :param version: The version number of the GIC file.
        """
        uf = "YES" if use_filters else "NO"
        return self.RunScriptCommand(f'GICWriteFilePTI("{filename}", {uf}, {version});')

    def GICWriteOptions(self, filename: str, key_field: str = "PRIMARY"):
        """Writes the current GIC solution options to an auxiliary file.

        :param filename: Name of Aux file name to write out the options.
        :param key_field: Identifier to use for the data (PRIMARY, SECONDARY, LABEL).
        """
        return self.RunScriptCommand(f'GICWriteOptions("{filename}", {key_field});')
