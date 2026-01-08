"""Regions specific functions."""
from typing import List


class RegionsMixin:
    """Mixin for Regions functions."""

    def RegionLoadShapefile(
        self,
        filename: str,
        class_name: str,
        attribute_names: List[str],
        add_to_open_onelines: bool = False,
        display_style_name: str = "",
        delete_existing: bool = False,
    ):
        """Loads shapes from a shapefile."""
        attrs = "[" + ", ".join(attribute_names) + "]"
        add = "YES" if add_to_open_onelines else "NO"
        delete = "YES" if delete_existing else "NO"
        return self.RunScriptCommand(
            f'RegionLoadShapefile("{filename}", "{class_name}", {attrs}, {add}, "{display_style_name}", {delete});'
        )

    def RegionRename(self, old_name: str, new_name: str, update_onelines: bool = True):
        """Renames an existing region."""
        uo = "YES" if update_onelines else "NO"
        return self.RunScriptCommand(f'RegionRename("{old_name}", "{new_name}", {uo});')

    def RegionRenameClass(self, old_class: str, new_class: str, update_onelines: bool = True, filter_name: str = ""):
        """Changes the class name of regions."""
        uo = "YES" if update_onelines else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RegionRenameClass("{old_class}", "{new_class}", {uo}, {filt});')

    def RegionRenameProper1(self, old_prop: str, new_prop: str, update_onelines: bool = True, filter_name: str = ""):
        """Changes the proper1 name of regions."""
        uo = "YES" if update_onelines else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RegionRenameProper1("{old_prop}", "{new_prop}", {uo}, {filt});')

    def RegionRenameProper2(self, old_prop: str, new_prop: str, update_onelines: bool = True, filter_name: str = ""):
        """Changes the proper2 name of regions."""
        uo = "YES" if update_onelines else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RegionRenameProper2("{old_prop}", "{new_prop}", {uo}, {filt});')

    def RegionRenameProper3(self, old_prop: str, new_prop: str, update_onelines: bool = True, filter_name: str = ""):
        """Changes the proper3 name of regions."""
        uo = "YES" if update_onelines else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RegionRenameProper3("{old_prop}", "{new_prop}", {uo}, {filt});')

    def RegionRenameProper12Flip(self, update_onelines: bool = True, filter_name: str = ""):
        """Flips proper1 and proper2 names."""
        uo = "YES" if update_onelines else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f"RegionRenameProper12Flip({uo}, {filt});")

    def RegionUpdateBuses(self):
        """Updates the buses in all the regions."""
        return self.RunScriptCommand("RegionUpdateBuses;")