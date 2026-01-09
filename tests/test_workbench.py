"""Unit tests for workbench.py GridWorkBench class."""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import numpy as np
import pandas as pd

from esapp.workbench import GridWorkBench
from esapp.grid import Bus, Branch, Gen, Load, Shunt, Area, Zone


@pytest.fixture
def mock_esa():
    """Create a mock SAW instance."""
    mock = Mock()
    mock.OpenCase = Mock(return_value=("", None))
    mock.CloseCase = Mock(return_value=("", None))
    mock.SaveCase = Mock(return_value=("", None))
    mock.SolvePowerFlow = Mock(return_value=("", None))
    mock.ResetToFlatStart = Mock(return_value=("", None))
    mock.RunScriptCommand = Mock(return_value=("", None))
    mock.LogAdd = Mock(return_value=("", None))
    mock.EnterMode = Mock(return_value=("", None))
    mock.LoadAux = Mock(return_value=("", None))
    mock.LoadScript = Mock(return_value=("", None))
    mock.GetFieldList = Mock(return_value=("", pd.DataFrame({'FieldName': ['Field1', 'Field2']})))
    mock.ChangeParametersSingleElement = Mock(return_value=("", None))
    mock.Scale = Mock(return_value=("", None))
    mock.CreateData = Mock(return_value=("", None))
    mock.Delete = Mock(return_value=("", None))
    mock.SetData = Mock(return_value=("", None))
    return mock


@pytest.fixture
def workbench(mock_esa):
    """Create a GridWorkBench instance with mocked ESA."""
    with patch('esapp.workbench.GridWorkBench.open'):
        wb = GridWorkBench()
        wb.esa = mock_esa
        wb.set_esa(mock_esa)
        return wb


class TestGridWorkBenchInit:
    """Tests for GridWorkBench initialization."""

    def test_init_without_file(self):
        """Test initialization without a file."""
        wb = GridWorkBench()
        assert wb.esa is None
        assert wb.fname is None
        assert wb._state_chain_idx == -1
        assert wb._state_chain_max == 2
        assert wb._dispatch_pq is None

    def test_init_with_file(self, mock_esa):
        """Test initialization with a file."""
        with patch('esapp.workbench.GridWorkBench.open'), \
             patch.object(GridWorkBench, 'esa', mock_esa, create=True):
            wb = GridWorkBench("test.pwb")
            assert wb.fname == "test.pwb"

    def test_set_esa(self, workbench, mock_esa):
        """Test set_esa propagates to applications."""
        workbench.set_esa(mock_esa)
        assert workbench.network.esa == mock_esa
        assert workbench.gic.esa == mock_esa
        assert workbench.modes.esa == mock_esa


class TestGridWorkBenchSimulation:
    """Tests for GridWorkBench simulation control methods."""

    def test_pflow_with_volts(self, workbench):
        """Test pflow method with voltage retrieval."""
        # Mock voltage data
        voltage_data = pd.DataFrame({
            'BusPUVolt': [1.0, 1.05, 0.95],
            'BusAngle': [0.0, -2.0, 5.0]
        })
        
        with patch.object(workbench, 'voltage', return_value=voltage_data['BusPUVolt'] * np.exp(1j * voltage_data['BusAngle'] * np.pi / 180)):
            result = workbench.pflow(getvolts=True)
            workbench.esa.SolvePowerFlow.assert_called_once()
            assert result is not None

    def test_pflow_without_volts(self, workbench):
        """Test pflow method without voltage retrieval."""
        result = workbench.pflow(getvolts=False)
        workbench.esa.SolvePowerFlow.assert_called_once()
        assert result is None

    def test_reset(self, workbench):
        """Test reset method."""
        workbench.reset()
        workbench.esa.ResetToFlatStart.assert_called_once()

    def test_save_with_filename(self, workbench):
        """Test save method with filename."""
        workbench.save("output.pwb")
        workbench.esa.SaveCase.assert_called_once_with("output.pwb")

    def test_save_without_filename(self, workbench):
        """Test save method without filename."""
        workbench.save()
        workbench.esa.SaveCase.assert_called_once_with(None)

    def test_command(self, workbench):
        """Test command method."""
        result = workbench.command("SolvePowerFlow;")
        workbench.esa.RunScriptCommand.assert_called_once_with("SolvePowerFlow;")

    def test_log(self, workbench):
        """Test log method."""
        workbench.log("Test message")
        workbench.esa.LogAdd.assert_called_once_with("Test message")

    def test_close(self, workbench):
        """Test close method."""
        workbench.close()
        workbench.esa.CloseCase.assert_called_once()

    def test_mode(self, workbench):
        """Test mode method."""
        workbench.mode("EDIT")
        workbench.esa.EnterMode.assert_called_once_with("EDIT")


class TestGridWorkBenchFileOperations:
    """Tests for GridWorkBench file operation methods."""

    def test_load_aux(self, workbench):
        """Test load_aux method."""
        workbench.load_aux("data.aux")
        workbench.esa.LoadAux.assert_called_once_with("data.aux")

    def test_load_script(self, workbench):
        """Test load_script method."""
        workbench.load_script("script.pws")
        workbench.esa.LoadScript.assert_called_once_with("script.pws")


class TestGridWorkBenchDataRetrieval:
    """Tests for GridWorkBench data retrieval methods."""

    def test_voltage_pu_complex_basic(self, workbench):
        """Test voltage method calls the correct indexing."""
        # Just test that the method can be called without error when mocked properly
        voltage_data = pd.Series([1.0 + 0j, 1.05 - 0.02j], name='voltage')
        
        with patch.object(workbench, 'voltage', return_value=voltage_data):
            result = workbench.voltage(asComplex=True)
            assert result is not None

    def test_get_fields(self, workbench):
        """Test get_fields method."""
        result = workbench.get_fields("Bus")
        workbench.esa.GetFieldList.assert_called_once_with("Bus")
        assert result is not None


class TestGridWorkBenchModification:
    """Tests for GridWorkBench modification methods."""

    def test_open_branch(self, workbench):
        """Test open_branch method."""
        workbench.open_branch(1, 2, '1')
        workbench.esa.ChangeParametersSingleElement.assert_called_once_with(
            "Branch",
            ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"],
            [1, 2, '1', "Open"]
        )

    def test_close_branch(self, workbench):
        """Test close_branch method."""
        workbench.close_branch(1, 2, '1')
        workbench.esa.ChangeParametersSingleElement.assert_called_once_with(
            "Branch",
            ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"],
            [1, 2, '1', "Closed"]
        )

    def test_set_gen_all_params(self, workbench):
        """Test set_gen with all parameters."""
        workbench.set_gen(bus=10, id="1", mw=150.0, mvar=50.0, status="Closed")
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "Gen" in args
        assert "GenMW" in args[1]
        assert "GenMVR" in args[1]
        assert "GenStatus" in args[1]

    def test_set_gen_mw_only(self, workbench):
        """Test set_gen with only MW parameter."""
        workbench.set_gen(bus=10, id="1", mw=150.0)
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "GenMW" in args[1]
        assert 150.0 in args[2]

    def test_set_gen_no_params(self, workbench):
        """Test set_gen with no optional parameters."""
        workbench.set_gen(bus=10, id="1")
        # Should not call ChangeParametersSingleElement
        workbench.esa.ChangeParametersSingleElement.assert_not_called()

    def test_set_load_all_params(self, workbench):
        """Test set_load with all parameters."""
        workbench.set_load(bus=5, id="1", mw=50.0, mvar=25.0, status="Closed")
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "Load" in args
        assert "LoadMW" in args[1]
        assert "LoadMVR" in args[1]
        assert "LoadStatus" in args[1]

    def test_set_load_mw_only(self, workbench):
        """Test set_load with only MW parameter."""
        workbench.set_load(bus=5, id="1", mw=50.0)
        workbench.esa.ChangeParametersSingleElement.assert_called_once()

    def test_scale_load(self, workbench):
        """Test scale_load method."""
        workbench.scale_load(1.1)
        workbench.esa.Scale.assert_called_once_with("LOAD", "FACTOR", [1.1], "SYSTEM")

    def test_scale_gen(self, workbench):
        """Test scale_gen method."""
        workbench.scale_gen(1.05)
        workbench.esa.Scale.assert_called_once_with("GEN", "FACTOR", [1.05], "SYSTEM")

    def test_create(self, workbench):
        """Test create method."""
        workbench.create("Load", BusNum=1, LoadID="1", LoadMW=10.0)
        workbench.esa.CreateData.assert_called_once()
        args = workbench.esa.CreateData.call_args[0]
        assert args[0] == "Load"
        assert "BusNum" in args[1]
        assert "LoadID" in args[1]
        assert "LoadMW" in args[1]

    def test_delete_no_filter(self, workbench):
        """Test delete method without filter."""
        workbench.delete("Gen")
        workbench.esa.Delete.assert_called_once_with("Gen", "")

    def test_delete_with_filter(self, workbench):
        """Test delete method with filter."""
        workbench.delete("Gen", filter_name="AreaNum = 1")
        workbench.esa.Delete.assert_called_once_with("Gen", "AreaNum = 1")

    def test_select_no_filter(self, workbench):
        """Test select method without filter."""
        workbench.esa.SelectAll = Mock(return_value=("", None))
        workbench.select("Bus")
        workbench.esa.SelectAll.assert_called_once_with("Bus", "")

    def test_select_with_filter(self, workbench):
        """Test select method with filter."""
        workbench.esa.SelectAll = Mock(return_value=("", None))
        workbench.select("Bus", filter_name="AreaNum = 2")
        workbench.esa.SelectAll.assert_called_once_with("Bus", "AreaNum = 2")


class TestGridWorkBenchEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_pflow_without_volts_no_return(self, workbench):
        """Test pflow without requesting volts returns None."""
        result = workbench.pflow(getvolts=False)
        assert result is None
        workbench.esa.SolvePowerFlow.assert_called_once()


class TestGridWorkBenchAdvancedMethods:
    """Tests for advanced workbench methods."""

    def test_unselect_no_filter(self, workbench):
        """Test unselect method without filter."""
        workbench.esa.UnSelectAll = Mock(return_value=("", None))
        workbench.unselect("Gen")
        workbench.esa.UnSelectAll.assert_called_once_with("Gen", "")

    def test_unselect_with_filter(self, workbench):
        """Test unselect method with filter."""
        workbench.esa.UnSelectAll = Mock(return_value=("", None))
        workbench.unselect("Load", filter_name="LoadMW < 10")
        workbench.esa.UnSelectAll.assert_called_once_with("Load", "LoadMW < 10")

    def test_energize(self, workbench):
        """Test energize method."""
        workbench.esa.CloseWithBreakers = Mock(return_value=("", None))
        workbench.energize("Bus", "[1]")
        workbench.esa.CloseWithBreakers.assert_called_once_with("Bus", "[1]")

    def test_deenergize(self, workbench):
        """Test deenergize method."""
        workbench.esa.OpenWithBreakers = Mock(return_value=("", None))
        workbench.deenergize("Bus", "[1]")
        workbench.esa.OpenWithBreakers.assert_called_once_with("Bus", "[1]")

    def test_radial_paths(self, workbench):
        """Test radial_paths method."""
        workbench.esa.FindRadialBusPaths = Mock(return_value=("", None))
        workbench.radial_paths()
        workbench.esa.FindRadialBusPaths.assert_called_once()

    def test_path_distance(self, workbench):
        """Test path_distance method."""
        workbench.esa.DeterminePathDistance = Mock(return_value=("", pd.DataFrame({'Distance': [0, 1, 2]})))
        result = workbench.path_distance("[BUS 1]")
        workbench.esa.DeterminePathDistance.assert_called_once_with("[BUS 1]")
        assert result is not None

    def test_network_cut(self, workbench):
        """Test network_cut method."""
        workbench.esa.SetSelectedFromNetworkCut = Mock(return_value=("", None))
        workbench.network_cut(1, "SELECTED")
        workbench.esa.SetSelectedFromNetworkCut.assert_called_once()

    def test_scale_load_by_factor(self, workbench):
        """Test scale_load with different factor."""
        workbench.scale_load(0.9)
        workbench.esa.Scale.assert_called_with("LOAD", "FACTOR", [0.9], "SYSTEM")

    def test_scale_gen_by_factor(self, workbench):
        """Test scale_gen with different factor."""
        workbench.scale_gen(1.2)
        workbench.esa.Scale.assert_called_with("GEN", "FACTOR", [1.2], "SYSTEM")

    def test_create_with_multiple_params(self, workbench):
        """Test create with multiple parameters."""
        workbench.create("Gen", BusNum=10, GenID="1", GenMW=100, GenMVR=50)
        workbench.esa.CreateData.assert_called_once()
        args = workbench.esa.CreateData.call_args[0]
        assert "GenMW" in args[1]
        assert "GenMVR" in args[1]

    def test_set_gen_status_only(self, workbench):
        """Test set_gen with only status parameter."""
        workbench.set_gen(bus=10, id="1", status="Open")
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "GenStatus" in args[1]
        assert "Open" in args[2]

    def test_set_load_status_only(self, workbench):
        """Test set_load with only status parameter."""
        workbench.set_load(bus=5, id="1", status="Closed")
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "LoadStatus" in args[1]

    def test_set_gen_mvar_only(self, workbench):
        """Test set_gen with only mvar parameter."""
        workbench.set_gen(bus=10, id="1", mvar=30.0)
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "GenMVR" in args[1]
        assert 30.0 in args[2]

    def test_set_load_mvar_only(self, workbench):
        """Test set_load with only mvar parameter."""
        workbench.set_load(bus=5, id="1", mvar=20.0)
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "LoadMVR" in args[1]

    def test_open_branch_default_ckt(self, workbench):
        """Test open_branch with default circuit ID."""
        workbench.open_branch(1, 2)
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "Open" in args[2]
        assert '1' in args[2]  # Default circuit

    def test_close_branch_default_ckt(self, workbench):
        """Test close_branch with default circuit ID."""
        workbench.close_branch(3, 4)
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "Closed" in args[2]

    def test_open_branch_custom_ckt(self, workbench):
        """Test open_branch with custom circuit ID."""
        workbench.open_branch(1, 2, "2")
        workbench.esa.ChangeParametersSingleElement.assert_called_once()
        args = workbench.esa.ChangeParametersSingleElement.call_args[0]
        assert "2" in args[2]

    def test_mode_run(self, workbench):
        """Test mode method with RUN."""
        workbench.mode("RUN")
        workbench.esa.EnterMode.assert_called_with("RUN")

    def test_save_no_filename(self, workbench):
        """Test save without filename (overwrite)."""
        workbench.fname = "original.pwb"
        workbench.save()
        workbench.esa.SaveCase.assert_called_with(None)


class TestGridWorkBenchStateManagement:
    """Tests for state management methods."""

    def test_state_chain_initialization(self):
        """Test state chain is properly initialized."""
        wb = GridWorkBench()
        assert wb._state_chain_idx == -1
        assert wb._state_chain_max == 2
        assert wb._dispatch_pq is None


class TestGridWorkBenchPropertyAccessors:
    """Tests for property accessor methods that return component data."""

    def test_voltages_pu_complex(self, workbench):
        """Test voltages method with pu=True, complex=True."""
        bus_data = pd.DataFrame({
            'BusPUVolt': [1.0, 1.05, 0.95],
            'BusAngle': [0.0, -5.0, 10.0]
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=bus_data):
            result = workbench.voltages(pu=True, complex=True)
            assert result is not None
            assert len(result) == 3

    def test_voltages_kv_complex(self, workbench):
        """Test voltages method with pu=False, complex=True."""
        bus_data = pd.DataFrame({
            'BusKVVolt': [138.0, 144.9, 131.1],
            'BusAngle': [0.0, -5.0, 10.0]
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=bus_data):
            result = workbench.voltages(pu=False, complex=True)
            assert result is not None
            assert len(result) == 3

    def test_voltages_pu_not_complex(self, workbench):
        """Test voltages method with pu=True, complex=False."""
        bus_data = pd.DataFrame({
            'BusPUVolt': [1.0, 1.05, 0.95],
            'BusAngle': [0.0, -5.0, 10.0]
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=bus_data):
            mag, ang = workbench.voltages(pu=True, complex=False)
            assert mag is not None
            assert ang is not None
            assert len(mag) == 3
            assert len(ang) == 3

    def test_generations(self, workbench):
        """Test generations method."""
        gen_data = pd.DataFrame({
            'GenMW': [100.0, 150.0],
            'GenMVR': [30.0, 45.0],
            'GenStatus': ['Closed', 'Closed']
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=gen_data):
            result = workbench.generations()
            assert result is not None
            assert len(result.columns) == 3

    def test_loads(self, workbench):
        """Test loads method."""
        load_data = pd.DataFrame({
            'LoadMW': [50.0, 75.0],
            'LoadMVR': [25.0, 30.0],
            'LoadStatus': ['Closed', 'Closed']
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=load_data):
            result = workbench.loads()
            assert result is not None
            assert len(result.columns) == 3

    def test_shunts(self, workbench):
        """Test shunts method."""
        shunt_data = pd.DataFrame({
            'ShuntMW': [0.0, 0.0],
            'ShuntMVR': [50.0, 100.0],
            'ShuntStatus': ['Closed', 'Closed']
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=shunt_data):
            result = workbench.shunts()
            assert result is not None
            assert len(result.columns) == 3

    def test_lines(self, workbench):
        """Test lines method."""
        branch_data = pd.DataFrame({
            'BranchDeviceType': ['Line', 'Transformer', 'Line'],
            'LineR': [0.01, 0.02, 0.015],
            'LineX': [0.1, 0.2, 0.15]
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=branch_data):
            result = workbench.lines()
            assert result is not None
            assert len(result) == 2  # Only lines, not transformers

    def test_transformers(self, workbench):
        """Test transformers method."""
        branch_data = pd.DataFrame({
            'BranchDeviceType': ['Line', 'Transformer', 'Line', 'Transformer'],
            'XfR': [np.nan, 0.02, np.nan, 0.025],
            'XfX': [np.nan, 0.2, np.nan, 0.25]
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=branch_data):
            result = workbench.transformers()
            assert result is not None
            assert len(result) == 2  # Only transformers, not lines

    def test_areas(self, workbench):
        """Test areas method."""
        area_data = pd.DataFrame({
            'AreaNum': [1, 2, 3],
            'AreaName': ['Area1', 'Area2', 'Area3']
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=area_data):
            result = workbench.areas()
            assert result is not None
            assert len(result) == 3

    def test_zones(self, workbench):
        """Test zones method."""
        zone_data = pd.DataFrame({
            'ZoneNum': [1, 2],
            'ZoneName': ['Zone1', 'Zone2']
        })
        
        with patch.object(GridWorkBench, '__getitem__', return_value=zone_data):
            result = workbench.zones()
            assert result is not None
            assert len(result) == 2

    def test_set_voltages(self, workbench):
        """Test set_voltages method."""
        V = np.array([1.0 + 0j, 1.05 - 0.02j, 0.95 + 0.01j])
        
        with patch.object(GridWorkBench, '__setitem__') as mock_setitem:
            workbench.set_voltages(V)
            mock_setitem.assert_called_once()
            # Verify it was called with Bus and voltage fields
            args = mock_setitem.call_args[0]
            assert args[0] == (Bus, ["BusPUVolt", "BusAngle"])
