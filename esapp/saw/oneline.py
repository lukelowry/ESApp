"""Oneline diagram specific functions."""


class OnelineMixin:
    """Mixin for oneline diagram functions."""

    def OpenOneLine(
        self,
        filename: str,
        view: str = "",
        full_screen: str = "NO",
        show_full: str = "NO",
        link_method: str = "LABELS",
        left: float = 0.0,
        top: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
    ) -> None:
        """
        Open a oneline diagram.
        Note: view needs to be quoted if not empty.

        Parameters
        ----------
        filename : str
            The path to the oneline diagram file (.axd).
        view : str, optional
            The name of a specific view within the oneline diagram to open. Defaults to "".
        full_screen : str, optional
            "YES" or "NO" to open in full screen. Defaults to "NO".
        show_full : str, optional
            "YES" or "NO" to show the full diagram. Defaults to "NO".
        link_method : str, optional
            Method for linking objects ("LABELS", "NUMBERS"). Defaults to "LABELS".
        left : float, optional
            Left coordinate for window placement. Defaults to 0.0.
        top : float, optional
            Top coordinate for window placement. Defaults to 0.0.
        width : float, optional
            Width of the window. Defaults to 0.0.
        height : float, optional
            Height of the window. Defaults to 0.0.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found).
        """
        view_str = f'"{view}"' if view else '""'
        script = (
            f'OpenOneline("{filename}", {view_str}, {full_screen}, {show_full}, '
            f"{link_method}, {left}, {top}, {width}, {height})"
        )
        return self.RunScriptCommand(script)

    def CloseOneline(self, OnelineName: str = "") -> None:
        """Closes an open oneline diagram.

        Parameters
        ----------
        OnelineName : str, optional
            The name of the oneline diagram to close. If empty, closes the active oneline.
            Defaults to "".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        script = f'CloseOneline("{OnelineName}")'
        return self.RunScriptCommand(script)

    def SaveOneline(self, filename: str, oneline_name: str, save_file_type: str = "PWB"):
        """Saves an open oneline diagram to a file.

        Parameters
        ----------
        filename : str
            The path to the file where the oneline diagram will be saved.
        oneline_name : str
            The name of the oneline diagram to save.
        save_file_type : str, optional
            The file type to save as (e.g., "PWB", "AXD"). Defaults to "PWB".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'SaveOneline("{filename}", "{oneline_name}", {save_file_type});')

    def ExportOneline(self, filename: str, oneline_name: str, image_type: str, view: str = "", full_screen: str = "NO", show_full: str = "NO"):
        """Exports an image of the open oneline diagram.

        Parameters
        ----------
        filename : str
            The path to the output image file.
        oneline_name : str
            The name of the oneline diagram to export.
        image_type : str
            The image file type (e.g., "JPG", "PNG", "BMP").
        view : str, optional
            The name of a specific view within the oneline diagram to export. Defaults to "".
        full_screen : str, optional
            "YES" or "NO" to export in full screen. Defaults to "NO".
        show_full : str, optional
            "YES" or "NO" to show the full diagram in the export. Defaults to "NO".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'ExportOneline("{filename}", "{oneline_name}", {image_type}, "{view}", {full_screen}, {show_full});')

    def ExportBusView(self, filename: str, bus_key: str, image_type: str, width: int, height: int, export_options: list = None):
        """Exports an image of a bus view oneline diagram.

        Parameters
        ----------
        filename : str
            The path to the output image file.
        bus_key : str
            The key of the bus for which to export the view (e.g., '[BUS 1]').
        image_type : str
            The image file type (e.g., "JPG", "PNG", "BMP").
        width : int
            The width of the exported image in pixels.
        height : int
            The height of the exported image in pixels.
        export_options : List[Any], optional
            A list of additional export options. Defaults to None.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        opts = ""
        if export_options:
            opts = ", [" + ", ".join([str(o) for o in export_options]) + "]"
        return self.RunScriptCommand(f'ExportBusView("{filename}", "{bus_key}", {image_type}, {width}, {height}{opts});')

    def ExportOnelineAsShapeFile(self, filename: str, oneline_name: str, description_name: str, use_lon_lat: bool = True, point_location: str = "center"):
        """Saves an open oneline diagram to a shapefile.

        Parameters
        ----------
        filename : str
            The path to the output shapefile.
        oneline_name : str
            The name of the oneline diagram to export.
        description_name : str
            A description name for the shapefile.
        use_lon_lat : bool, optional
            If True, uses longitude and latitude for point locations. Defaults to True.
        point_location : str, optional
            Specifies the point location ("center", "bus", "gen", etc.). Defaults to "center".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        ull = "YES" if use_lon_lat else "NO"
        return self.RunScriptCommand(f'ExportOnelineAsShapeFile("{filename}", "{oneline_name}", "{description_name}", {ull}, {point_location});')

    def PanAndZoomToObject(self, object_id: str, display_object_type: str = "", do_zoom: bool = True):
        """Pans to and optionally zooms in on a display object on the active oneline diagram.

        Parameters
        ----------
        object_id : str
            The ID of the object to pan/zoom to (e.g., '[BUS 1]', '[GEN 2]').
        display_object_type : str, optional
            The type of display object (e.g., "Bus", "Gen"). Defaults to "".
        do_zoom : bool, optional
            If True, also zooms in on the object. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        dz = "YES" if do_zoom else "NO"
        return self.RunScriptCommand(f'PanAndZoomToObject("{object_id}", "{display_object_type}", {dz});')

    def OpenBusView(self, bus_key: str, force_new_window: bool = False):
        """Opens the Bus View to a particular bus.

        Parameters
        ----------
        bus_key : str
            The key of the bus for which to open the view (e.g., '[BUS 1]').
        force_new_window : bool, optional
            If True, forces the view to open in a new window. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fnw = "YES" if force_new_window else "NO"
        return self.RunScriptCommand(f'OpenBusView("{bus_key}", {fnw});')

    def OpenSubView(self, substation_key: str, force_new_window: bool = False):
        """Opens the Substation View to a particular substation.

        Parameters
        ----------
        substation_key : str
            The key of the substation for which to open the view (e.g., '[SUB 1]').
        force_new_window : bool, optional
            If True, forces the view to open in a new window. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fnw = "YES" if force_new_window else "NO"
        return self.RunScriptCommand(f'OpenSubView("{substation_key}", {fnw});')

    def LoadAXD(self, filename: str, oneline_name: str, create_if_not_found: bool = False):
        """Applies a display auxiliary file (.axd) to an open oneline diagram.

        This can be used to load graphical elements or display settings.

        Parameters
        ----------
        filename : str
            The path to the display auxiliary file.
        oneline_name : str
            The name of the target oneline diagram.
        create_if_not_found : bool, optional
            If True, creates the oneline diagram if it does not exist. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadAXD("{filename}", "{oneline_name}", {c});')

    def RelinkAllOpenOnelines(self):
        """Attempts to relink all objects on all open oneline diagrams.

        This is useful if objects have been renumbered or modified in the case
        and their graphical representations need to be updated.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("RelinkAllOpenOnelines;")