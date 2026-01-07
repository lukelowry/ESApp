"""Scheduled Actions specific functions."""


class ScheduledActionsMixin:
    """Mixin for Scheduled Actions functions."""

    def ApplyScheduledActionsAt(self, start_time: str, end_time: str = "", filter_name: str = "", revert: bool = False):
        """
        Applies scheduled actions active during the specified window.

        Parameters
        ----------
        start_time : str
            The start time of the window.
        end_time : str, optional
            The end time of the window.
        filter_name : str, optional
            Filter to apply to the actions.
        revert : bool, optional
            Whether to revert the actions.

        Returns
        -------
        str
            The result of the script command.
        """
        rev = "YES" if revert else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'ApplyScheduledActionsAt("{start_time}", "{end_time}", {filt}, {rev});')

    def IdentifyBreakersForScheduledActions(self, identify_from_normal: bool = True):
        """
        Identifies breakers for scheduled actions.

        Parameters
        ----------
        identify_from_normal : bool, optional
            Whether to identify from normal status.

        Returns
        -------
        str
            The result of the script command.
        """
        ifn = "YES" if identify_from_normal else "NO"
        return self.RunScriptCommand(f"IdentifyBreakersForScheduledActions({ifn});")

    def RevertScheduledActionsAt(self, start_time: str, end_time: str = "", filter_name: str = ""):
        """
        Reverts scheduled actions.

        Parameters
        ----------
        start_time : str
            The start time of the window.
        end_time : str, optional
            The end time of the window.
        filter_name : str, optional
            Filter to apply to the actions.

        Returns
        -------
        str
            The result of the script command.
        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'RevertScheduledActionsAt("{start_time}", "{end_time}", {filt});')

    def ScheduledActionsSetReference(self):
        """
        Sets the reference state restored prior to applying a time stamp.

        Returns
        -------
        str
            The result of the script command.
        """
        return self.RunScriptCommand("ScheduledActionsSetReference;")

    def SetScheduleView(
        self, view_time: str, apply_actions: bool = None, use_normal_status: bool = None, apply_window: bool = None
    ):
        """
        Sets the View Time for Scheduled Actions.

        Parameters
        ----------
        view_time : str
            The time to view.
        apply_actions : bool, optional
            Whether to apply actions.
        use_normal_status : bool, optional
            Whether to use normal status.
        apply_window : bool, optional
            Whether to apply the window.

        Returns
        -------
        str
            The result of the script command.
        """
        aa = "YES" if apply_actions else "NO" if apply_actions is not None else ""
        uns = "YES" if use_normal_status else "NO" if use_normal_status is not None else ""
        aw = "YES" if apply_window else "NO" if apply_window is not None else ""
        return self.RunScriptCommand(f'SetScheduleView("{view_time}", {aa}, {uns}, {aw});')

    def SetScheduleWindow(
        self, start_time: str, end_time: str, resolution: float = None, resolution_units: str = None
    ):
        """
        Defines the window of interest for Scheduled Actions.

        Parameters
        ----------
        start_time : str
            The start time of the window.
        end_time : str
            The end time of the window.
        resolution : float, optional
            The resolution of the window.
        resolution_units : str, optional
            The units of the resolution.

        Returns
        -------
        str
            The result of the script command.
        """
        res = str(resolution) if resolution is not None else ""
        units = resolution_units if resolution_units else ""
        return self.RunScriptCommand(f'SetScheduleWindow("{start_time}", "{end_time}", {res}, {units});')