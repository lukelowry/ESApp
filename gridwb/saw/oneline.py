"""Oneline diagram specific functions."""


class OnelineMixin:
    """Mixin for oneline diagram functions."""

    def OpenOneLine(
        self,
        filename: str,
        view: str = "",
        FullScreen: str = "NO",
        ShowFull: str = "NO",
        LinkMethod: str = "LABELS",
        Left: float = 0.0,
        Top: float = 0.0,
        Width: float = 0.0,
        Height: float = 0.0,
    ) -> None:
        """
        Open a oneline diagram.
        Note: view needs to be quoted if not empty.
        """
        view_str = f'"{view}"' if view else '""'
        script = (
            f'OpenOneline("{filename}", {view_str}, {FullScreen}, {ShowFull}, '
            f"{LinkMethod}, {Left}, {Top}, {Width}, {Height})"
        )
        return self.RunScriptCommand(script)

    def CloseOneline(self, OnelineName: str = "") -> None:
        """Close a oneline diagram."""
        script = f'CloseOneline("{OnelineName}")'
        return self.RunScriptCommand(script)

    def SaveOneline(self, filename: str, oneline_name: str, save_file_type: str = "PWB"):
        """Save an open oneline diagram to file."""
        return self.RunScriptCommand(f'SaveOneline("{filename}", "{oneline_name}", {save_file_type});')

    def ExportOneline(self, filename: str, oneline_name: str, image_type: str, view: str = "", full_screen: str = "NO", show_full: str = "NO"):
        """Export an image of the open oneline diagram."""
        return self.RunScriptCommand(f'ExportOneline("{filename}", "{oneline_name}", {image_type}, "{view}", {full_screen}, {show_full});')

    def ExportBusView(self, filename: str, bus_key: str, image_type: str, width: int, height: int, export_options: list = None):
        """Export an image of a bus view oneline diagram."""
        opts = ""
        if export_options:
            opts = ", [" + ", ".join([str(o) for o in export_options]) + "]"
        return self.RunScriptCommand(f'ExportBusView("{filename}", "{bus_key}", {image_type}, {width}, {height}{opts});')

    def ExportOnelineAsShapeFile(self, filename: str, oneline_name: str, description_name: str, use_lon_lat: bool = True, point_location: str = "center"):
        """Save an open oneline diagram to a shapefile."""
        ull = "YES" if use_lon_lat else "NO"
        return self.RunScriptCommand(f'ExportOnelineAsShapeFile("{filename}", "{oneline_name}", "{description_name}", {ull}, {point_location});')

    def PanAndZoomToObject(self, object_id: str, display_object_type: str = "", do_zoom: bool = True):
        """Pan to and optionally zoom in on a display object."""
        dz = "YES" if do_zoom else "NO"
        return self.RunScriptCommand(f'PanAndZoomToObject("{object_id}", "{display_object_type}", {dz});')

    def OpenBusView(self, bus_key: str, force_new_window: bool = False):
        """Opens the Bus View to a particular bus."""
        fnw = "YES" if force_new_window else "NO"
        return self.RunScriptCommand(f'OpenBusView("{bus_key}", {fnw});')

    def OpenSubView(self, substation_key: str, force_new_window: bool = False):
        """Opens the Substation View to a particular substation."""
        fnw = "YES" if force_new_window else "NO"
        return self.RunScriptCommand(f'OpenSubView("{substation_key}", {fnw});')

    def LoadAXD(self, filename: str, oneline_name: str, create_if_not_found: bool = False):
        """Apply a display auxiliary file to an open oneline diagram."""
        c = "YES" if create_if_not_found else "NO"
        return self.RunScriptCommand(f'LoadAXD("{filename}", "{oneline_name}", {c});')

    def RelinkAllOpenOnelines(self):
        """Attempt to relink all objects on all open onelines."""
        return self.RunScriptCommand("RelinkAllOpenOnelines;")