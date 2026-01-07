
from ..indextool import IndexTool

# TODO App Features
# - TSGetVCurveData("FileName", filter);This field comapres to QV curve!
# - TSRunResultAnalyzer PW Post-transient analysis
# - Tolerance MVA
# - Need to auto create DSTimeSChedule & LoadCharacteristic for Ramp

# TODO Remove this 'Conditions' Feature I Decided I do not like it

# Application Base Class
class PWApp:
    def __init__(self, io: IndexTool) -> None:
        
        # Application Interface
        self.io = io

