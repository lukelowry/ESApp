"""General script commands and data interaction functions."""
from typing import List


class GeneralMixin:
    """Mixin for General Program Actions and Data Interaction."""

    def CopyFile(self, old_filename: str, new_filename: str):
        """Copies a file."""
        return self.RunScriptCommand(f'CopyFile("{old_filename}", "{new_filename}");')

    def DeleteFile(self, filename: str):
        """Deletes a file."""
        return self.RunScriptCommand(f'DeleteFile("{filename}");')

    def RenameFile(self, old_filename: str, new_filename: str):
        """Renames a file."""
        return self.RunScriptCommand(f'RenameFile("{old_filename}", "{new_filename}");')

    def WriteTextToFile(self, filename: str, text: str):
        """Writes text to a file."""
        escaped_text = text.replace('"', '""')
        return self.RunScriptCommand(f'WriteTextToFile("{filename}", "{escaped_text}");')

    def LogAdd(self, text: str) -> None:
        """Adds a message to the PowerWorld Message Log."""
        return self.RunScriptCommand(f'LogAdd("{text}");')

    def LogClear(self) -> None:
        """Clears the PowerWorld Message Log."""
        return self.RunScriptCommand("LogClear;")

    def LogShow(self, show: bool = True):
        """Shows or hides the Message Log."""
        yn = "YES" if show else "NO"
        return self.RunScriptCommand(f"LogShow({yn});")

    def LogSave(self, filename: str, append: bool = False):
        """Saves the message log to a file."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'LogSave("{filename}", {app});')

    def SetCurrentDirectory(self, directory: str, create_if_not_found: bool = False):
        """Sets the current working directory."""
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'SetCurrentDirectory("{directory}", {c});')

    def EnterMode(self, mode: str) -> None:
        """Enters PowerWorld into a specific mode."""
        if mode.upper() not in ["RUN", "EDIT"]:
            raise ValueError("Mode must be either 'RUN' or 'EDIT'.")
        return self.RunScriptCommand(f"EnterMode({mode.upper()});")

    def StoreState(self, statename: str) -> None:
        """Stores the current state under a given name."""
        return self.RunScriptCommand(f'StoreState("{statename}");')

    def RestoreState(self, statename: str) -> None:
        """Restores a previously saved user state."""
        return self.RunScriptCommand(f'RestoreState(USER, "{statename}");')

    def DeleteState(self, statename: str) -> None:
        """Deletes a previously saved user state."""
        return self.RunScriptCommand(f'DeleteState(USER, "{statename}");')

    def LoadAux(self, filename: str, create_if_not_found: bool = False):
        """Loads an auxiliary file."""
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadAux("{filename}", {c});')

    def ImportData(self, filename: str, filetype: str, header_line: int = 1, create_if_not_found: bool = False):
        """Imports data in various file formats."""
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'ImportData("{filename}", {filetype}, {header_line}, {c});')

    def LoadCSV(self, filename: str, create_if_not_found: bool = False):
        """Loads a CSV file formatted like Send To Excel."""
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadCSV("{filename}", {c});')

    def LoadScript(self, filename: str, script_name: str = ""):
        """Loads and runs a script from an auxiliary file."""
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
        """Saves data to a file using the SaveData script command."""
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
        """Saves data with extra user-specified fields."""
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
        """Sets data for objects."""
        fields = "[" + ", ".join(fieldlist) + "]"
        values = "[" + ", ".join([str(v) for v in valuelist]) + "]"

        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name

        return self.RunScriptCommand(f"SetData({objecttype}, {fields}, {values}, {filt});")

    def CreateData(self, objecttype: str, fieldlist: List[str], valuelist: List[str]):
        """Creates a new object."""
        fields = "[" + ", ".join(fieldlist) + "]"
        values = "[" + ", ".join([str(v) for v in valuelist]) + "]"
        return self.RunScriptCommand(f"CreateData({objecttype}, {fields}, {values});")

    def SaveObjectFields(self, filename: str, objecttype: str, fieldlist: List[str]):
        """Saves a list of fields available for the specified objecttype."""
        fields = "[" + ", ".join(fieldlist) + "]"
        return self.RunScriptCommand(f'SaveObjectFields("{filename}", {objecttype}, {fields});')

    def Delete(self, objecttype: str, filter_name: str = ""):
        """Deletes objects."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"Delete({objecttype}, {filt});")

    def SelectAll(self, objecttype: str, filter_name: str = ""):
        """Sets the Selected field to YES for objects."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"SelectAll({objecttype}, {filt});")

    def UnSelectAll(self, objecttype: str, filter_name: str = ""):
        """Sets the Selected field to NO for objects."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"UnSelectAll({objecttype}, {filt});")

    def SendToExcel(self, objecttype: str, fieldlist: List[str], filter_name: str = "", use_column_headers: bool = True, workbook: str = "", worksheet: str = "", sortfieldlist: List[str] = None, header_list: List[str] = None, header_value_list: List[str] = None, clear_existing: bool = True, row_shift: int = 0, col_shift: int = 0):
        """Sends data to Excel."""
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