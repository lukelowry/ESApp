"""
saw is short for SimAuto Wrapper. This package provides a class, SAW, for
interfacing with PowerWorld's Simulator Automation Server (SimAuto).
"""
from .base import SAWBase
from .atc import ATCMixin
from .case_actions import CaseActionsMixin
from .contingency import ContingencyMixin
from .general import GeneralMixin
from .fault import FaultMixin
from .gic import GICMixin
from .matrices import MatrixMixin
from .modify import ModifyMixin
from .oneline import OnelineMixin
from .opf import OPFMixin
from .powerflow import PowerflowMixin
from .pv import PVMixin
from .qv import QVMixin
from .regions import RegionsMixin
from .sensitivity import SensitivityMixin
from .scheduled import ScheduledActionsMixin
from .topology import TopologyMixin
from .transient import TransientMixin
from .timestep import TimeStepMixin
from .weather import WeatherMixin


class SAW(
    SAWBase,
    CaseActionsMixin,
    ContingencyMixin,
    GeneralMixin,
    MatrixMixin,
    ModifyMixin,
    OnelineMixin,
    PowerflowMixin,
    RegionsMixin,
    SensitivityMixin,
    ScheduledActionsMixin,
    TopologyMixin,
    TransientMixin,
    FaultMixin,
    ATCMixin,
    GICMixin,
    OPFMixin,
    PVMixin,
    QVMixin,
    TimeStepMixin,
    WeatherMixin,
):
    """A SimAuto Wrapper in Python, composed of a base class and several
    functionality-specific mixin classes.
    """

    pass
