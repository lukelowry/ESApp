"""Regions specific functions."""
from typing import List

from ._enums import YesNo
from ._helpers import format_list


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
        """
        Loads shapes from a shapefile.

        Parameters
        ----------
        filename : str
            Path to the shapefile.
        class_name : str
            The object class to associate with the shapes.
        attribute_names : List[str]
            List of attribute names to map.
        add_to_open_onelines : bool, optional
            If True, adds shapes to open onelines. Defaults to False.
        display_style_name : str, optional
            Name of the display style to use. Defaults to "".
        delete_existing : bool, optional
            If True, deletes existing objects of this class. Defaults to False.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        attrs = format_list(attribute_names)
        add = YesNo.from_bool(add_to_open_onelines)
        delete = YesNo.from_bool(delete_existing)
        return self._run_script("RegionLoadShapefile", f'"{filename}"', f'"{class_name}"', attrs, add, f'"{display_style_name}"', delete)

    def RegionRename(self, old_name: str, new_name: str, update_onelines: bool = True):
        """
        Renames an existing region.

        Parameters
        ----------
        old_name : str
            The current name of the region.
        new_name : str
            The new name for the region.
        update_onelines : bool, optional
            If True, updates onelines to reflect the change. Defaults to True.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        uo = YesNo.from_bool(update_onelines)
        return self._run_script("RegionRename", f'"{old_name}"', f'"{new_name}"', uo)

    def RegionRenameClass(self, old_class: str, new_class: str, update_onelines: bool = True, filter_name: str = ""):
        """
        Changes the class name of regions.

        Parameters
        ----------
        old_class : str
            The current class name.
        new_class : str
            The new class name.
        update_onelines : bool, optional
            If True, updates onelines. Defaults to True.
        filter_name : str, optional
            Filter to apply. Defaults to "".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        uo = YesNo.from_bool(update_onelines)
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("RegionRenameClass", f'"{old_class}"', f'"{new_class}"', uo, filt)

    def RegionRenameProper1(self, old_prop: str, new_prop: str, update_onelines: bool = True, filter_name: str = ""):
        """
        Changes the proper1 name of regions.

        Parameters
        ----------
        old_prop : str
            The current proper1 name.
        new_prop : str
            The new proper1 name.
        update_onelines : bool, optional
            If True, updates onelines. Defaults to True.
        filter_name : str, optional
            Filter to apply. Defaults to "".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        uo = YesNo.from_bool(update_onelines)
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("RegionRenameProper1", f'"{old_prop}"', f'"{new_prop}"', uo, filt)

    def RegionRenameProper2(self, old_prop: str, new_prop: str, update_onelines: bool = True, filter_name: str = ""):
        """
        Changes the proper2 name of regions.

        Parameters
        ----------
        old_prop : str
            The current proper2 name.
        new_prop : str
            The new proper2 name.
        update_onelines : bool, optional
            If True, updates onelines. Defaults to True.
        filter_name : str, optional
            Filter to apply. Defaults to "".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        uo = YesNo.from_bool(update_onelines)
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("RegionRenameProper2", f'"{old_prop}"', f'"{new_prop}"', uo, filt)

    def RegionRenameProper3(self, old_prop: str, new_prop: str, update_onelines: bool = True, filter_name: str = ""):
        """
        Changes the proper3 name of regions.

        Parameters
        ----------
        old_prop : str
            The current proper3 name.
        new_prop : str
            The new proper3 name.
        update_onelines : bool, optional
            If True, updates onelines. Defaults to True.
        filter_name : str, optional
            Filter to apply. Defaults to "".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        uo = YesNo.from_bool(update_onelines)
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("RegionRenameProper3", f'"{old_prop}"', f'"{new_prop}"', uo, filt)

    def RegionRenameProper12Flip(self, update_onelines: bool = True, filter_name: str = ""):
        """
        Flips proper1 and proper2 names.

        Parameters
        ----------
        update_onelines : bool, optional
            If True, updates onelines. Defaults to True.
        filter_name : str, optional
            Filter to apply. Defaults to "".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        uo = YesNo.from_bool(update_onelines)
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("RegionRenameProper12Flip", uo, filt)

    def RegionUpdateBuses(self):
        """
        Updates the buses in all the regions.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self._run_script("RegionUpdateBuses")
