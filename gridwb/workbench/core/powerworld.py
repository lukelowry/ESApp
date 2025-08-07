from typing import Type
from pandas import DataFrame
from os import path
from numpy import unique

from ..grid.components import *
from ..utils.decorators import timing
from ..io.model import IModelIO
from ...saw import SAW, CommandNotRespectedError # NOTE Should be the only file importing SAW


# Helper Function to parse Python Syntax/Field Syntax outliers
# Example: fexcept('ThreeWindingTransformer') -> '3WindingTransformer
fexcept = lambda t: "3" + t[5:] if t[:5] == "Three" else t

# Power World Read/Write
class PowerWorldIO(IModelIO):
    esa: SAW

    @timing
    def open(self):
        # Validate Path Name
        if not path.isabs(self.fname):
            self.fname = path.abspath(self.fname)

        # ESA Object & Transient Sim
        self.esa = SAW(self.fname, CreateIfNotFound=True, early_bind=True)

        # Attempt and Initialize TS so we get initial values
        self.TSInit()
    
    def __getitem__(self, index) -> DataFrame | None:
        '''Retrieve Data frome Power world with Indexor Notation
        
        
        Examples:
        wb.pw[Bus] # Get Primary Keys of PW Buses
        wb.pw[Bus, 'BusPUVolt'] # Get Voltage Magnitudes
        wb.pw[Bus, ['SubNum', 'BusPUVolt']] # Get Two Fields
        wb.pw[Bus, :] # Get all fields
        '''
        
        # Type checking is an anti-pattern but this is accepted within community as a necessary part of the magic function
        # >1 Argument - Objecet Type & Fields(s)
        if isinstance(index, tuple): 
            gtype, fields = index
            if isinstance(fields, str): fields = fields,
            elif isinstance(fields, slice): fields = gtype.fields
        # 1 Argument - Object Type: retrieve only key fields
        else: 
            gtype, fields = index, ()

        # Keys and then Fields
        key_fields = gtype.keys
        data_fields = [f for f in fields if f not in key_fields]
        unique_fields = [*key_fields, *data_fields]

        # If no fields (I.e. there were no keys and no data field passed)
        if len(unique_fields) < 1:
            return None

        # Retrieve data from unique list of fields
        df = self.esa.GetParametersMultipleElement(gtype.TYPE, unique_fields)

        # Set Index of DF if key field exists and DF valid
        #if df is not None and len(key_fields)>0:
            #df.set_index(key_fields, inplace=True) # NOTE I might want to rethink indexing these. It is useful to me but might be confusing.
        
        return df
    
    # TODO remove the conditional index, too confusing.
    def __setitem__(self, args, value) -> None:
        '''Set grid data using indexors directly to Power World
        Must be atleast 2 args: Type & Field

        Examples:
        wb.pw[Bus, 'BusPUVolt'] = 1
        wb.pw[Bus, [xxx,xxx,xxx]] = [arr1, arr2, arr3]
        '''

        # Type checking is an anti-pattern but this is accepted within community as a necessary part of the magic function
        # Extract Arguments depending on Index Method

        # Ensure Edit Mode
        self.edit_mode()

        # PARSE ARGUMENT FORMAT OPTIONS

        # Limited Data Passed NOTE useful when target data is not a properly named DF
        if isinstance(args, tuple) or isinstance(args, list):

            # [Type, [Fields]] -> Target data listed by field, expects array/dataframe with respective ordering
            if len(args)==2:   
                gtype, fields = args[0], args[1]

            # Format fields passed as list
            if isinstance(fields, str): 
                fields = fields,
            
            # Retrieve active power world records with keys only
            base = self[gtype,:]

            # Assign Values based on index
            base.loc[:,fields] = value

        # [Type] -> Try and Create New (Requires properly formatted df)
        else: 
            gtype, base = args, value
            
        # Send to Power World
        self.esa.change_parameters_multiple_element_df(gtype.TYPE, base)

        # Enter back into run mode
        self.run_mode()

    def save(self):
        '''
        Description:
            Save all Open Changes to PWB File.
        Note: 
            only data/settings written back to PowerWorld will be saved.
        '''
        return self.esa.SaveCase()
   
    ''' Power Flow Commands '''

    def flatstart(self):
        '''
        Description:
            Resets voltage vector to a flat start.
        '''
        self.esa.RunScriptCommand("ResetToFlatStart()")


    def pflow(self, retry=True):
        '''
        Description
            Executes Power Flow in PowerWorld. 
        Parameters
            retry: if True (default) this function will reset 
                    PF if it fails and try one additional time.
        Other
            Use do_one_iteration() to pause after each iteration.
        '''
        try:
            self.esa.SolvePowerFlow()
        except:
            if retry:
                self.flatstart()
                self.esa.SolvePowerFlow()


    ''' Playin Signal Section'''

    def clearsignals(self):
        '''
        Description
            Clears all playin signals
        '''
        self.esa.RunScriptCommand('DELETE(PLAYINSIGNAL);')

    def setsignals(self, name, times, signals):
        '''
        Sets Playin signals
        Parameters:
        name: Name of PlayIn Configuration
        times: 1D Array of times of Length N
        signals: N x M where M is number of Signals
        Power World blocks signal data from being written for some reason so we must set through AUX command.'''

        # Format Data Header
        cmd = 'DATA (PLAYINSIGNAL, [TSName, TSTime'

        for idx in range(signals.shape[1]):

            cmd += ', TSSignal'

            if idx > 0:
                cmd += ':'+ str(idx)
        
        cmd += ']){\n' 

        # Format each time record
        for t, row in zip(times, signals):
            cmd += f'"{name}"\t{t:.6f}'
            for d in row:
                cmd += f'\t{d:.6f}'
            cmd += '\n'

        cmd += '}\n'

        # Execute
        self.esa.exec_aux(cmd)

    ''' Power World Settings '''

    def set_mva_tol(self, tol=0.1):
        '''
        Description
            Sets the MVA Tolerance for NR Convergence
        '''
        self[Sim_Solution_Options,'ConvergenceTol:2'] = tol

    def do_one_iteration(self, enable=True):
        '''
        Description
            Sets the "DoOneIteration" setting.
        Parameters:
            enable: True (default) forces one iteration.
        '''
        self[Sim_Solution_Options,'DoOneIteration'] = 'YES' if enable else 'NO'

    def inner_loop_check_mvar_immediatly(self, enable=True):
        ''' 
        Description
            Set inner loop of power flow to check mvar limits before proceeding to outer loop.
        '''
        self[Sim_Solution_Options,'ChkVars'] = 'YES' if enable else 'NO'

    def get_min_volt(self):
        '''
        Description:
            The minimum p.u. voltage magnitude in the case.
        '''
        return self[PWCaseInformation,'BusPUVolt:1'].iloc[0,0]
    
    def get_mismatch(self):
        '''
        Description:
            The complex power bus mismatch vector
        Returns:
            (P, Q) Tuple of pandas series for P and Q mismatch in MVA
        '''
        mm = self[Bus,['BusMismatchP', 'BusMismatchQ']]
        return mm['BusMismatchP'], mm['BusMismatchQ']

    def save_state(self, statename="GWB"):
        '''
        Description
            Store a state under an alias and restore it later.
        '''
        self.run_mode()
        self.esa.RunScriptCommand(f'StoreState({statename});')

    def restore_state(self, statename="GWB"):
        '''
        Description:
            Restore to a saved state.
        Parameters:
            statename: The state name
        '''
        self.run_mode()
        self.esa.RunScriptCommand(f'RestoreState(USER,{statename});')

    def delete_state(self, statename="GWB"):
        '''
        Description:
            Deletes a saved state.
        Parameters:
            statename: The state name
        '''
        self.esa.RunScriptCommand('EnterMode(RUN);')
        self.esa.RunScriptCommand(f'DeleteState(USER,{statename});')

    def edit_mode(self):
        '''
        Description:
            Enters PowerWorld into EDIT mode.
        '''
        self.esa.RunScriptCommand("EnterMode(EDIT);")

    def run_mode(self):
        '''
        Description:
            Enters PowerWorld into RUN mode.
        '''
        self.esa.RunScriptCommand("EnterMode(RUN);")
        
    '''
    TODO Implement with setitem function
    def __set_sol_opts(self, name, value):
        settings = self[Sim_Solution_Options]
        settings['name'] = value
        self.upload({
            Sim_Solution_Options: settings
        })

    def max_iterations(self, n: int):
        self.__set_sol_opts('MaxItr', n)

    def zbr_threshold(self, v: float):
        self.__set_sol_opts('ZBRThreshold', v)

    '''

    ''' Power World Contingency Analysis '''

    
    '''
    Depricated until .upload removed
    def skipallbut(self, ctgs):
        ctgset = self.get(TSContingency)

        # Set Skip if not in ctg list
        ctgset["CTGSkip"] = "YES"
        ctgset.loc[ctgset["TSCTGName"].isin(ctgs), "CTGSkip"] = "NO"

        self.upload({TSContingency: ctgset})
    '''

    def TSInit(self):
        '''
        Description
            Initialize Transient Stability Parameters
        '''
        try:
            self.esa.RunScriptCommand("TSInitialize()")
        except:
            print("Failed to Initialize TS Values")

        
    # Execute Dynamic Simulation for Non-Skipped Contingencies
    def TSSolveAll(self):
        self.esa.RunScriptCommand("TSSolveAll()")

    def clearram(self):
        # Disable RAM storage & Delete Existing Data in RAM
        self.esa.RunScriptCommand("TSResultStorageSetAll(ALL, NO)")
        self.esa.RunScriptCommand("TSClearResultsFromRAM(ALL,YES,YES,YES,YES,YES)")


    savemap = {
        'TSGenDelta': 'TSSaveGenDelta',
        'TSGenMachineState:1': 'TSSaveGenMachine', # Delta
        'TSGenMachineState:2': 'TSSaveGenMachine', # Speed Deviation
        'TSGenMachineState:3': 'TSSaveGenMachine', # Eqp
        'TSGenMachineState:4': 'TSSaveGenMachine', # Psidp
        'TSGenMachineState:5': 'TSSaveGenMachine', # Psippq
        'TSGenExciterState:1': 'TSSaveGenExciter', # Efd
        'TSGenExciterState:2': 'TSSaveGenExciter', # ?
        'TSGenExciterState:3': 'TSSaveGenExciter', # Vr
        'TSGenExciterState:4': 'TSSaveGenExciter', # Vf
        'TSBusRad': 'TSSaveBusDeg',
    }

    def saveinram(self, objdf, datafields):
        '''
        Save Specified Fields for TS
        '''

        # Get Respective Data
        savefields = []
        for field in datafields:
            if field in self.savemap: 
                savefield = self.savemap[field]
            else:
                savefield = 'TSSave' + field[2:]
            
            objdf[savefield] = 'YES'
            savefields.append(savefield)

        # First three 
        keys = list(objdf.columns[:2])

        # Unique Save Fields
        savefields = np.unique(savefields)

        # Write to PW
        self.esa.change_and_confirm_params_multiple_element(
            ObjectType=objdf.Name,
            command_df=objdf[np.concatenate([keys,savefields])].copy(),
        )
