# Do not import this with module. For meta development# Problematic Objects

import pandas as pd
import numpy as np

'''
IMPORT DATA
'''


data = pd.read_excel(r'C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\PW Vars\PW Raw.xlsx', index_col=0) 
data = data[["Variable Name","Type of Variable","Key/Required Fields","Description","Enterable", "Default"]]
data.index = data.reset_index()["Type"].ffill()
data["Key/Required Fields"].fillna("", inplace=True)
data["Default"].fillna("", inplace=True)
recs = data.dropna().reset_index().to_dict('records')

'''
PRE-SPECIFICATION PARAMETERS
'''

# Problematic Objects
excludeObjects = [
    'AlarmOptions',
    'GenMWMaxMin_GenMWMaxMinXYCurve',
    'GenMWMaxMin_GenMWMaxMinXYCurve',
    'GenMWMax_SolarPVBasic1',
    'GenMWMax_SolarPVBasic2',
    'GenMWMax_TemperatureBasic1',
    'GenMWMax_WindBasic',
    'GenMWMax_WindClass1',
    'GenMWMax_WindClass2',
    'GenMWMax_WindClass3',
    'GenMWMax_WindClass4',
    'GICGeographicRegionSet',
    'GIC_Options',
    'LPOPFMarginalControls',
    'MvarMarginalCostValues',
    'MWMarginalCostValues',
    'NEMGroupBranch',
    'NEMGroupGroup',
    'NEMGroupNode',
    'PieSizeColorOptions',
    'PWBranchDataObject',
    'RT_Study_Options',
    'SchedSubscription',
    'TSFreqSummaryObject',
    'TSModalAnalysisObject',
    'TSSchedule',
    'Exciter_Generic',
    'Governor_Generic',
    'InjectionGroupModel_GenericInjectionGroup',
    'LoadCharacteristic_Generic'
]

# Problematic Fields
excludeFields = [
    'BusMarginalControl',
    'BusMCMVARValue',
    'BusMCMWValue',
    'LoadGrounded',
    'GEDateIn', 
    'GEDateOut'
]

# Repetative Defaults
massDefault = {
    "TSDeviceStatus": 'Active',   
}

# Um
creationkeyDefaults = {
    "AreaNum": 1,
    "ZoneNum": 1,
    "BusName_NomVolt": ""
}

# Data Type Mapping
dtypemap = {
    "String": "str",
    "Real": "float",
    "Integer": "int"
}


'''

INITIAL PARSING FILTER OF DOCUMENT

'''

# Clean and structure data from CSV

otypes = {r['Type']: [] for r in recs if len(r['Type'])>1 and r['Type'] not in excludeObjects}

# Extract
for r in recs:

    otype = r['Type']
    vname = r['Variable Name']
    p = r["Key/Required Fields"]
    e = r["Enterable"]
    defVal = r['Default']

    # Filter Objects
    if otype in excludeObjects:
        continue

    # Filter Out one Specific Fields
    if vname not in excludeFields and '/' not in vname:

        # Priority
        if '1' in p:
            p = 1
        elif '2' in p:
            p = 2
        elif '3' in p:
            p = 3
        # Secondaries    
        elif 'A' in p:
            p = 4
        elif 'B' in p:
            p = 5
        elif 'C' in p:
            p = 6
        # Required
        elif '**' in p: 
            p = 7
        # Optional    
        else:
            p = 10

        # Variable Name
        vname = str(vname).replace(":", "__")
        if vname[0]=='3':
            vname = 'Three'+vname[1:]

        #Default Value
        if defVal=="":
            if vname in massDefault:
                defVal = massDefault[vname]
            elif p==7 and vname in creationkeyDefaults:
                defVal = creationkeyDefaults[vname]
            else:
                defVal = np.NaN
        
        otypes[r['Type']].append({
            "VName": vname,
            "Priority": p,
            "DType": dtypemap[r['Type of Variable']],
            "Desc": str(r["Description"])[1:-1],
            "Edit": 'Yes' in e or 'Depends' in e,
            "Default": defVal
        })

# Sort By Priority
for o in otypes:
    otypes[o].sort(key=lambda x: x['Priority'])


'''

 WRITE TO FILE

'''


# Helper Function to parse Python Syntax/Field Syntax outliers
# Example: fexcept('ThreeWindingTransformer') -> '3WindingTransformer
#FIX_STR = lambda t: "3" + t[5:] if t[:5] == "Three" else t

def FIX_STR(name):

    newName = "3" + name[5:] if name[:5] == "Three" else name 

    newName = str.replace(newName, '__', ':')

    return newName

dclass = """
from enum import Enum, Flag, auto

class FieldPriority(Flag):
    PRIMARY   = auto()
    SECONDARY = auto()
    REQUIRED  = auto()
    OPTIONAL  = auto()
    EDITABLE  = auto()


class GObject(Enum):

    # Called when each field of subclass is parsed by python
    def __new__(cls, *args):
        
        # The object type string name 
        if len(args) == 1:

            cls._TYPE = args[0]
            
             # Set intgeer and name as member value
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = value

            return obj
        
        # Everything else is field
        else:
            # Look at raw field name and priority
            field_name_str, field_dtype, field_priority = args 

            # Check if fields class function has been initialized
            if not hasattr(cls, '_FIELDS'):
                cls._FIELDS = []
            if not hasattr(cls, '_KEYS'):
                cls._KEYS = []

            # Add to appropriate Lists
            cls._FIELDS.append(field_name_str)
            if field_priority & FieldPriority.PRIMARY == FieldPriority.PRIMARY:
                cls._KEYS.append(field_name_str)


            # Set intgeer and name as member value
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = (value,  field_name_str, field_dtype, field_priority)

            return obj
    
    def __repr__(self) -> str:
        return str(self._value_)
    
    def __str__(self) -> str:
        return f'Field String: {self._value_[1]}'
    
    @classmethod
    @property
    def keys(cls):
        '''
        Get the properly formatted string names of all fields
        '''
        if not hasattr(cls, '_KEYS'):
            return []
        return cls._KEYS
    
    @classmethod
    @property
    def fields(cls):
        '''
        Get the properly formatted string names of all fields
        '''
        if not hasattr(cls, '_FIELDS'):
            return []
        return cls._FIELDS 
    
    @classmethod
    @property
    def TYPE(cls):
        if not hasattr(cls, '_TYPE'):
            return 'NO_OBJECT_NAME'
        return cls._TYPE

\n\n
"""

# Write all Obj/Fields Formatted as Python File

OUT_FNAME = 'classes2.txt'
with open(OUT_FNAME, "w") as out:
    out.write(dclass)

    for obj in otypes:

        # Class
        cls = obj.split(" ")[0]

        # One of the only non-valid name corrections for ThreeWindingXFMR
        if cls[0]=='3':
            cls = 'Three'+cls[1:]

        # NOTE changing to Enum approach. dataclass is not even needed
        out.write(f'class {cls}(GObject):')

        # Variables
        for var in otypes[obj]:

            out.write('\n\t')
            out.write(var["VName"] + r' = ("' + FIX_STR(var["VName"]) + r'", ' + var['DType'] + ", ")

            p = var["Priority"]
            dv = var['Default']

            # Write the PW String ID 
            out.write(r'FieldPriority.')

            # TODO add Priority indication somewhere
            if p <= 3:
                out.write(r'PRIMARY')
            elif p > 3:
                # Secondary Fields
                if p<=7:
                    if dv is np.NaN:
                        out.write(r'SECONDARY')
                    else:
                        out.write(r'SECONDARY')
                # Other        
                else:
                    if dv is np.NaN:
                        out.write(r'OPTIONAL')
                    else:
                        out.write(r'OPTIONAL')
                # Required Field Flag
                if p==7:
                    out.write(r' | FieldPriority.REQUIRED')
                    
            out.write(")\n\t")
            out.write(r'"""')
            out.write(str(var["Desc"]).replace("\\", "/"))
            out.write(r' """')

        # Hidden Type
        out.write("\n\t")
        out.write("ObjectString = '" + obj + "'")
        out.write('\n')

    out.close()