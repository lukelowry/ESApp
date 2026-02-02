"""Scheduled Actions specific functions."""


from esapp.saw._enums import YesNo


class ScheduledActionsMixin:
    """Mixin for Scheduled Actions functions."""

    def ApplyScheduledActionsAt(self, start_time: str, end_time: str = "", filter_name: str = "", revert: bool = False):
        """Applies scheduled actions active during the specified time window.

        Scheduled actions are predefined changes (e.g., opening/closing branches,
        changing generation) that occur at specific times.

        Parameters
        ----------
        start_time : str
            The start time of the window (e.g., "01/01/2025 10:00").
        end_time : str, optional
            The end time of the window. If empty, only actions at `start_time` are applied.
            Defaults to "".
        filter_name : str, optional
            A PowerWorld filter name to apply to scheduled actions. Defaults to an empty string (all).
        revert : bool, optional
            If True, reverts the actions instead of applying them. Defaults to False.

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        rev = YesNo.from_bool(revert)
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("ApplyScheduledActionsAt", f'"{start_time}"', f'"{end_time}"', filt, rev)

    def IdentifyBreakersForScheduledActions(self, identify_from_normal: bool = True):
        """Identifies breakers for scheduled actions.

        This action helps in setting up scheduled actions that involve breaker operations.

        Parameters
        ----------
        identify_from_normal : bool, optional
            If True, identifies breakers based on their normal status. Defaults to True.

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        ifn = YesNo.from_bool(identify_from_normal)
        return self._run_script("IdentifyBreakersForScheduledActions", ifn)

    def RevertScheduledActionsAt(self, start_time: str, end_time: str = "", filter_name: str = ""):
        """Reverts scheduled actions that were active during the specified time window.

        This undoes the changes made by `ApplyScheduledActionsAt`.

        Parameters
        ----------
        start_time : str
            The start time of the window (e.g., "01/01/2025 10:00").
        end_time : str, optional
            The end time of the window. If empty, only actions at `start_time` are reverted.
            Defaults to "".
        filter_name : str, optional
            A PowerWorld filter name to apply to scheduled actions. Defaults to an empty string (all).

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("RevertScheduledActionsAt", f'"{start_time}"', f'"{end_time}"', filt)

    def ScheduledActionsSetReference(self):
        """Sets the current system state as the reference for scheduled actions.

        This reference state is used when applying or reverting scheduled actions
        to ensure a consistent baseline.

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("ScheduledActionsSetReference")

    def SetScheduleView(
        self, view_time: str, apply_actions: bool = None, use_normal_status: bool = None, apply_window: bool = None
    ):
        """Sets the View Time for Scheduled Actions.

        This allows viewing the system state at a specific point in time,
        considering all scheduled actions up to that point.

        Parameters
        ----------
        view_time : str
            The specific time to view (e.g., "01/01/2025 10:00").
        apply_actions : bool, optional
            If True, applies scheduled actions up to `view_time`. Defaults to None (uses current setting).
        use_normal_status : bool, optional
            If True, uses normal status for elements. Defaults to None (uses current setting).
        apply_window : bool, optional
            If True, applies the scheduled window. Defaults to None (uses current setting).

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        aa = YesNo.from_bool(apply_actions) if apply_actions is not None else None
        uns = YesNo.from_bool(use_normal_status) if use_normal_status is not None else None
        aw = YesNo.from_bool(apply_window) if apply_window is not None else None
        return self._run_script("SetScheduleView", f'"{view_time}"', aa, uns, aw)

    def SetScheduleWindow(
        self, start_time: str, end_time: str, resolution: float = None, resolution_units: str = None
    ):
        """Defines the window of interest for Scheduled Actions.

        This sets the time range and resolution for displaying or processing
        scheduled actions.

        Parameters
        ----------
        start_time : str
            The start time of the window (e.g., "01/01/2025 00:00").
        end_time : str
            The end time of the window (e.g., "02/01/2025 00:00").
        resolution : float, optional
            The time step resolution for the window (e.g., 1.0). Defaults to None.
        resolution_units : str, optional
            The units for the resolution ("HOURS", "MINUTES", "SECONDS"). Defaults to None.

        Returns
        -------
        str
            The result of the script command.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("SetScheduleWindow", f'"{start_time}"', f'"{end_time}"', resolution, resolution_units)
