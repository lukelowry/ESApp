"""General script commands and data interaction functions."""
from typing import List


class GeneralMixin:
    """Mixin for General Program Actions and Data Interaction."""

    def CopyFile(self, old_filename: str, new_filename: str):
        """Copies a file from `old_filename` to `new_filename`.

        Parameters
        ----------
        old_filename : str
            The path to the source file.
        new_filename : str
            The path to the destination file.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, permission issues).
        """
        return self.RunScriptCommand(f'CopyFile("{old_filename}", "{new_filename}");')

    def DeleteFile(self, filename: str):
        """Deletes a specified file.

        Parameters
        ----------
        filename : str
            The path to the file to delete.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, permission issues).
        """
        return self.RunScriptCommand(f'DeleteFile("{filename}");')

    def RenameFile(self, old_filename: str, new_filename: str):
        """Renames a file from `old_filename` to `new_filename`.

        Parameters
        ----------
        old_filename : str
            The current path of the file.
        new_filename : str
            The new path/name for the file.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, new name already exists).
        """
        return self.RunScriptCommand(f'RenameFile("{old_filename}", "{new_filename}");')

    def WriteTextToFile(self, filename: str, text: str):
        """Writes a given text string to a file.

        Parameters
        ----------
        filename : str
            The path to the file where the text will be written.
        text : str
            The text string to write.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., permission issues).
        """
        escaped_text = text.replace('"', '""')
        return self.RunScriptCommand(f'WriteTextToFile("{filename}", "{escaped_text}");')

    def LogAdd(self, text: str) -> None:
        """Adds a message to the PowerWorld Simulator Message Log.

        Parameters
        ----------
        text : str
            The message string to add to the log.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'LogAdd("{text}");')

    def LogClear(self) -> None:
        """Clears all messages from the PowerWorld Simulator Message Log.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("LogClear;")

    def LogShow(self, show: bool = True):
        """Shows or hides the PowerWorld Simulator Message Log window.

        Parameters
        ----------
        show : bool, optional
            If True, shows the Message Log window. If False, hides it. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        yn = "YES" if show else "NO"
        return self.RunScriptCommand(f"LogShow({yn});")

    def LogSave(self, filename: str, append: bool = False):
        """Saves the contents of the PowerWorld Simulator Message Log to a file.

        Parameters
        ----------
        filename : str
            The path to the file where the log will be saved.
        append : bool, optional
            If True, appends to the file if it exists. If False, overwrites the file.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., permission issues).
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'LogSave("{filename}", {app});')

    def SetCurrentDirectory(self, directory: str, create_if_not_found: bool = False):
        """Sets the current working directory for PowerWorld Simulator.

        This directory is used for resolving relative file paths in subsequent commands.

        Parameters
        ----------
        directory : str
            The path to the directory to set as current.
        create_if_not_found : bool, optional
            If True, creates the directory if it does not exist. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid path, permission issues).
        """
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'SetCurrentDirectory("{directory}", {c});')

    def EnterMode(self, mode: str) -> None:
        """Enters PowerWorld Simulator into a specific operating mode.

        Parameters
        ----------
        mode : str
            The mode to enter. Must be either "RUN" or "EDIT".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `mode` is not "RUN" or "EDIT".
        PowerWorldError
            If the SimAuto call fails.
        """
        if mode.upper() not in ["RUN", "EDIT"]:
            raise ValueError("Mode must be either 'RUN' or 'EDIT'.")
        return self.RunScriptCommand(f"EnterMode({mode.upper()});")

    def StoreState(self, statename: str) -> None:
        """Stores the current state of the PowerWorld case under a given name.

        This creates a named snapshot of the case that can be restored later.

        Parameters
        ----------
        statename : str
            The name to assign to the stored state.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'StoreState("{statename}");')

    def RestoreState(self, statename: str) -> None:
        """Restores a previously saved user state by its name.

        Parameters
        ----------
        statename : str
            The name of the state to restore.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., state not found).
        """
        return self.RunScriptCommand(f'RestoreState(USER, "{statename}");')

    def DeleteState(self, statename: str) -> None:
        """Deletes a previously saved user state by its name.

        Parameters
        ----------
        statename : str
            The name of the state to delete.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., state not found).
        """
        return self.RunScriptCommand(f'DeleteState(USER, "{statename}");')

    def LoadAux(self, filename: str, create_if_not_found: bool = False):
        """Loads an auxiliary (.aux) file into PowerWorld Simulator.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file.
        create_if_not_found : bool, optional
            If True, attempts to create objects defined in the aux file if they don't exist.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, syntax error in aux file).
        """
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadAux("{filename}", {c});')

    def ImportData(self, filename: str, filetype: str, header_line: int = 1, create_if_not_found: bool = False):
        """Imports data from a file in various formats into PowerWorld Simulator.

        Parameters
        ----------
        filename : str
            The path to the data file.
        filetype : str
            The format of the file (e.g., "CSV", "TXT", "PTI").
        header_line : int, optional
            The line number where the header (column names) is located. Defaults to 1.
        create_if_not_found : bool, optional
            If True, attempts to create objects if they don't exist. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'ImportData("{filename}", {filetype}, {header_line}, {c});')

    def LoadCSV(self, filename: str, create_if_not_found: bool = False):
        """Loads a CSV file, typically one formatted similarly to output from `SendToExcel`.

        Parameters
        ----------
        filename : str
            The path to the CSV file.
        create_if_not_found : bool, optional
            If True, attempts to create objects if they don't exist. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadCSV("{filename}", {c});')

    def LoadScript(self, filename: str, script_name: str = ""):
        """Loads and runs a script from an auxiliary file.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file containing the script.
        script_name : str, optional
            The name of the script within the file to execute. If empty, the first
            script in the file is run. Defaults to "".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'LoadScript("{filename}", "{script_name}");')

    def SaveData(
        self,
        filename: str,
        filetype: str,
        objecttype: str,
        fieldlist: List[str],
        subdatalist: List[str] = None,
        filter_name: str = "",
        sortfieldlist: List[str] = None,
        transpose: bool = False,
        append: bool = True,
    ):
        """Saves data for specified objects and fields to a file using the `SaveData` script command.

        Parameters
        ----------
        filename : str
            The path to the output file.
        filetype : str
            The format of the output file (e.g., "CSV", "AUX", "TXT").
        objecttype : str
            The PowerWorld object type (e.g., "Bus", "Gen").
        fieldlist : List[str]
            A list of internal field names to save.
        subdatalist : List[str], optional
            A list of sub-data fields to save (e.g., for time series data). Defaults to None.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).
        sortfieldlist : List[str], optional
            A list of fields to sort the output by. Defaults to None.
        transpose : bool, optional
            If True, transposes the output data (rows become columns, columns become rows).
            Defaults to False.
        append : bool, optional
            If True, appends data to the file if it exists. If False, overwrites.
            Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fields = "[" + ", ".join(fieldlist) + "]"
        subs = "[" + ", ".join(subdatalist) if subdatalist else "[]"
        if subdatalist:
            subs += "]"

        sorts = "[" + ", ".join(sortfieldlist) if sortfieldlist else "[]"
        if sortfieldlist:
            sorts += "]"

        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name

        trans = "YES" if transpose else "NO"
        app = "YES" if append else "NO"

        cmd = (
            f'SaveData("{filename}", {filetype}, {objecttype}, {fields}, {subs}, '
            f'{filt}, {sorts}, {trans}, {app});'
        )
        return self.RunScriptCommand(cmd)

    def SaveDataWithExtra(self, filename: str, filetype: str, objecttype: str, fieldlist: List[str], subdatalist: List[str] = None, filter_name: str = "", sortfieldlist: List[str] = None, header_list: List[str] = None, header_value_list: List[str] = None, transpose: bool = False, append: bool = True):
        """Saves data with extra user-specified header fields and values.

        This method extends `SaveData` by allowing custom header information
        to be added to the output file, useful for metadata or tracking.

        Parameters
        ----------
        filename : str
            The path to the output file.
        filetype : str
            The format of the output file (e.g., "CSV", "AUX", "TXT").
        objecttype : str
            The PowerWorld object type (e.g., "Bus", "Gen").
        fieldlist : List[str]
            A list of internal field names to save.
        subdatalist : List[str], optional
            A list of sub-data fields to save. Defaults to None.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string.
        sortfieldlist : List[str], optional
            A list of fields to sort the output by. Defaults to None.
        header_list : List[str], optional
            A list of custom header names to add to the file. Defaults to None.
        header_value_list : List[str], optional
            A list of values corresponding to `header_list`. Defaults to None.
        transpose : bool, optional
            If True, transposes the output data. Defaults to False.
        append : bool, optional
            If True, appends data to the file. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fields = "[" + ", ".join(fieldlist) + "]"
        subs = "[" + ", ".join(subdatalist) if subdatalist else "[]"
        if subdatalist: subs += "]"
        sorts = "[" + ", ".join(sortfieldlist) if sortfieldlist else "[]"
        if sortfieldlist: sorts += "]"
        headers = "[" + ", ".join([f'"{h}"' for h in header_list]) if header_list else "[]"
        if header_list: headers += "]"
        values = "[" + ", ".join([f'"{v}"' for v in header_value_list]) if header_value_list else "[]"
        if header_value_list: values += "]"
        
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        trans = "YES" if transpose else "NO"
        app = "YES" if append else "NO"
        
        cmd = f'SaveDataWithExtra("{filename}", {filetype}, {objecttype}, {fields}, {subs}, {filt}, {sorts}, {headers}, {values}, {trans}, {app});'
        return self.RunScriptCommand(cmd)

    def SetData(self, objecttype: str, fieldlist: List[str], valuelist: List[str], filter_name: str = ""):
        """Sets data for specified objects and fields.

        This is a generic method for modifying object parameters.

        Parameters
        ----------
        objecttype : str
            The PowerWorld object type (e.g., "Bus", "Gen").
        fieldlist : List[str]
            A list of internal field names to set.
        valuelist : List[str]
            A list of values corresponding to `fieldlist`.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fields = "[" + ", ".join(fieldlist) + "]"
        values = "[" + ", ".join([str(v) for v in valuelist]) + "]"
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f"SetData({objecttype}, {fields}, {values}, {filt});")

    def CreateData(self, objecttype: str, fieldlist: List[str], valuelist: List[str]):
        """Creates a new object of a specified type with initial field values.

        Parameters
        ----------
        objecttype : str
            The PowerWorld object type to create (e.g., "Bus", "Gen").
        fieldlist : List[str]
            A list of internal field names for the new object. This must include
            all primary key fields.
        valuelist : List[str]
            A list of values corresponding to `fieldlist` for the new object.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., object already exists, invalid parameters).
        """
        fields = "[" + ", ".join(fieldlist) + "]"
        values = "[" + ", ".join([str(v) for v in valuelist]) + "]"
        return self.RunScriptCommand(f"CreateData({objecttype}, {fields}, {values});")

    def SaveObjectFields(self, filename: str, objecttype: str, fieldlist: List[str]):
        """Saves a list of fields available for the specified objecttype to a file.

        Parameters
        ----------
        filename : str
            The path to the output file.
        objecttype : str
            The PowerWorld object type.
        fieldlist : List[str]
            A list of internal field names to save.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fields = "[" + ", ".join(fieldlist) + "]"
        return self.RunScriptCommand(f'SaveObjectFields("{filename}", {objecttype}, {fields});')

    def Delete(self, objecttype: str, filter_name: str = ""):
        """Deletes objects of a specified type, optionally filtered.

        Parameters
        ----------
        objecttype : str
            The PowerWorld object type to delete.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"Delete({objecttype}, {filt});")

    def SelectAll(self, objecttype: str, filter_name: str = ""):
        """Sets the 'Selected' field to YES for objects of a specified type, optionally filtered.

        Parameters
        ----------
        objecttype : str
            The PowerWorld object type.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"SelectAll({objecttype}, {filt});")

    def UnSelectAll(self, objecttype: str, filter_name: str = ""):
        """Sets the 'Selected' field to NO for objects of a specified type, optionally filtered.

        Parameters
        ----------
        objecttype : str
            The PowerWorld object type.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"UnSelectAll({objecttype}, {filt});")

    def SendToExcelAdvanced(self, objecttype: str, fieldlist: List[str], filter_name: str = "", use_column_headers: bool = True, workbook: str = "", worksheet: str = "", sortfieldlist: List[str] = None, header_list: List[str] = None, header_value_list: List[str] = None, clear_existing: bool = True, row_shift: int = 0, col_shift: int = 0):
        """Sends data for specified objects and fields directly to Microsoft Excel with advanced options.

        This is an extended version of SendToExcel that provides additional control over
        Excel output including workbook/worksheet names, sorting, custom headers, and positioning.
        This method requires Microsoft Excel to be installed on the system.

        Parameters
        ----------
        objecttype : str
            The PowerWorld object type (e.g., "Bus", "Gen").
        fieldlist : List[str]
            A list of internal field names to export.
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).
        use_column_headers : bool, optional
            If True, includes column headers in the Excel output. Defaults to True.
        workbook : str, optional
            The name of the Excel workbook to write to. If empty, a new workbook is created.
            Defaults to "".
        worksheet : str, optional
            The name of the worksheet within the workbook. If empty, a new worksheet is created.
            Defaults to "".
        sortfieldlist : List[str], optional
            A list of fields to sort the output by. Defaults to None.
        header_list : List[str], optional
            A list of custom header names to add to the Excel output. Defaults to None.
        header_value_list : List[str], optional
            A list of values corresponding to `header_list`. Defaults to None.
        clear_existing : bool, optional
            If True, clears existing data in the target worksheet before writing.
            Defaults to True.
        row_shift : int, optional
            Number of rows to shift the output down from the top-left corner. Defaults to 0.
        col_shift : int, optional
            Number of columns to shift the output right from the top-left corner. Defaults to 0.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., Excel not installed, invalid parameters).
        
        See Also
        --------
        SendToExcel : Basic version with fewer parameters for simple exports.
        """
        fields = "[" + ", ".join(fieldlist) + "]"
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        uch = "YES" if use_column_headers else "NO"
        sorts = "[" + ", ".join(sortfieldlist) if sortfieldlist else "[]"
        if sortfieldlist: sorts += "]"
        headers = "[" + ", ".join([f'"{h}"' for h in header_list]) if header_list else "[]"
        if header_list: headers += "]"
        values = "[" + ", ".join([f'"{v}"' for v in header_value_list]) if header_value_list else "[]"
        if header_value_list: values += "]"
        ce = "YES" if clear_existing else "NO"
        
        cmd = f'SendtoExcel({objecttype}, {fields}, {filt}, {uch}, "{workbook}", "{worksheet}", {sorts}, {headers}, {values}, {ce}, {row_shift}, {col_shift});'
        return self.RunScriptCommand(cmd)

    def LogAddDateTime(
        self,
        label: str,
        include_date: bool = True,
        include_time: bool = True,
        include_milliseconds: bool = False
    ):
        """Adds the current date and time to the PowerWorld Simulator Message Log.

        Use this action to add a timestamped entry to the message log for tracking
        when certain operations occur during script execution.

        Parameters
        ----------
        label : str
            A string which will appear at the start of the line containing the date/time.
        include_date : bool, optional
            If True, includes the date in the log entry. Defaults to True.
        include_time : bool, optional
            If True, includes the time in the log entry. Defaults to True.
        include_milliseconds : bool, optional
            If True, includes milliseconds in the time. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.

        Examples
        --------
        >>> saw.LogAddDateTime("DateTime", True, True, True)
        # Adds a log entry labeled "DateTime" with current date, time, and milliseconds.
        """
        id = "YES" if include_date else "NO"
        it = "YES" if include_time else "NO"
        im = "YES" if include_milliseconds else "NO"
        return self.RunScriptCommand(f'LogAddDateTime("{label}", {id}, {it}, {im});')

    def LoadAuxDirectory(
        self,
        file_directory: str,
        filter_string: str = "",
        create_if_not_found: bool = False
    ):
        """Loads multiple auxiliary files from a specified directory.

        The auxiliary files will be loaded in alphabetical order by name. This is
        useful for batch loading configuration files or data updates.

        Parameters
        ----------
        file_directory : str
            The directory where the auxiliary files are located.
        filter_string : str, optional
            A filter string using Windows wildcard patterns (e.g., "*.aux").
            If not specified, all files in the directory are loaded. Defaults to "".
        create_if_not_found : bool, optional
            If True, objects that cannot be found will be created while reading
            DATA sections from the files. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., directory not found).

        Examples
        --------
        >>> saw.LoadAuxDirectory("C:/SimCases/AuxFiles", "*.aux", True)
        # Loads all .aux files from the directory in alphabetical order.
        """
        c = "YES" if create_if_not_found else "NO"
        if filter_string:
            return self.RunScriptCommand(f'LoadAuxDirectory("{file_directory}", "{filter_string}", {c});')
        else:
            return self.RunScriptCommand(f'LoadAuxDirectory("{file_directory}", , {c});')

    def LoadData(self, filename: str, data_name: str, create_if_not_found: bool = False):
        """Loads a named DATA section from another auxiliary file.

        This opens the auxiliary file denoted by filename but only reads the
        specific data section specified, ignoring other sections.

        Parameters
        ----------
        filename : str
            The filename of the auxiliary file being loaded.
        data_name : str
            The specific data section name from the auxiliary file that should be loaded.
        create_if_not_found : bool, optional
            If True, objects that cannot be found will be created. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file or data section not found).
        """
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadData("{filename}", {data_name}, {c});')

    def StopAuxFile(self):
        """Treats the remainder of the file after this command as a comment.

        This includes any script commands inside the present SCRIPT block,
        as well as all remaining SCRIPT or DATA blocks. Useful for temporarily
        disabling portions of an auxiliary file.

        Note: This command is primarily useful when executing auxiliary files
        directly, not when using SimAuto programmatically.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("StopAuxFile;")