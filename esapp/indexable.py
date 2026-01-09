from .saw import SAW
from .gobject import GObject
from .utils import timing
from typing import Type, Optional
from pandas import DataFrame
from os import path


# Helper Function to parse Python Syntax/Field Syntax outliers
# Example: fexcept('ThreeWindingTransformer') -> '3WindingTransformer
fexcept = lambda t: "3" + t[5:] if t[:5] == "Three" else t

# Power World Read/Write
class Indexable:
    """
    PowerWorld Read/Write tool providing indexer-based access to grid components.

    This class enables DataFrame-like access to PowerWorld Simulator data,
    allowing users to retrieve and modify component parameters using familiar
    indexing syntax.
    """
    esa: SAW
    fname: str

    def set_esa(self, esa: SAW):
        """
        Set the SAW (SimAuto Wrapper) instance for this object.

        Parameters
        ----------
        esa : SAW
            An initialized SAW instance.
        """
        self.esa = esa

    @timing
    def open(self):
        """
        Open the PowerWorld case and initialize transient stability.

        This method validates the case path, initializes the SimAuto COM object,
        and attempts to initialize transient stability to ensure initial values
        are available for dynamic models.
        """
        # Validate Path Name
        if not path.isabs(self.fname):
            self.fname = path.abspath(self.fname)

        # ESA Object & Transient Sim
        self.esa = SAW(self.fname, CreateIfNotFound=True, early_bind=True)

        # Attempt and Initialize TS so we get initial values
        self.esa.TSInitialize()
    
    def __getitem__(self, index) -> Optional[DataFrame]:
        """Retrieve data from PowerWorld using indexer notation.

        This method allows for flexible querying of grid component data directly
        from the PowerWorld simulation instance.

        Parameters
        ----------
        index : Union[Type[GObject], Tuple[Type[GObject], Any]]
            Can be a `GObject` type to get key fields, or a tuple of
            (GObject type, fields) to specify fields. `fields` can be a
            single field name (str), a list of names, or `slice(None)` (:)
            to retrieve all available fields.

        Returns
        -------
        Optional[pandas.DataFrame]
            A DataFrame containing the requested data, or ``None`` if no
            data could be retrieved.

        Raises
        ------
        ValueError
            If an unsupported slice is used for field selection.
        """
        # 1. Parse index to get gtype and what fields are requested.
        if isinstance(index, tuple):
            gtype, requested_fields = index
        else:
            gtype, requested_fields = index, None

        # 2. Determine the complete set of fields to retrieve.
        # Always start with the object's key fields.
        fields_to_get = set(gtype.keys)

        # 3. Add any additional fields based on the request.
        if requested_fields is None:
            # Case: wb.pw[Bus] -> only key fields are needed.
            pass
        elif requested_fields == slice(None):
            # Case: wb.pw[Bus, :] -> add all defined fields.
            fields_to_get.update(gtype.fields)
        else:
            # Case: wb.pw[Bus, 'field'] or wb.pw[Bus, ['f1', 'f2']]
            # Normalize to an iterable to handle single or multiple fields.
            if isinstance(requested_fields, (str, GObject)):
                requested_fields = [requested_fields]
            
            for field in requested_fields:
                if isinstance(field, GObject):
                    fields_to_get.add(field.value[1])
                elif isinstance(field, str):
                    fields_to_get.add(field)
                elif isinstance(field, slice):
                    raise ValueError("Only the full slice [:] is supported for selecting fields.")

        # 4. Handle edge case where no fields are identified.
        if not fields_to_get:
            return None

        # 5. Retrieve data from PowerWorld
        return self.esa.GetParamsRectTyped(gtype.TYPE, sorted(list(fields_to_get)))
    
    def __setitem__(self, args, value) -> None:
        """
        Set grid data in PowerWorld using indexer notation.

        Parameters
        ----------
        args : Union[Type[GObject], Tuple[Type[GObject], Union[str, List[str]]]]
            The target object type and optional fields.
        value : Union[pandas.DataFrame, Any]
            The data to write. If `args` is just a GObject type, `value`
            must be a DataFrame containing primary keys. If `args` includes
            fields, `value` can be a scalar (which is broadcast) or a
            list/array matching the number of objects.

        Raises
        ------
        TypeError
            If the index or value types are mismatched or unsupported.
        """
        # Case 1: Bulk update from a DataFrame. e.g., wb.pw[Bus] = df
        if isinstance(args, type) and issubclass(args, GObject):
            self._bulk_update_from_df(args, value)
            return

        # Case 2: Broadcast update to specific fields. e.g., wb.pw[Bus, 'BusPUVolt'] = 1.05
        if isinstance(args, tuple) and len(args) == 2:
            gtype, fields = args

            if not (isinstance(gtype, type) and issubclass(gtype, GObject)):
                raise TypeError(f"First element of index must be a GObject subclass, not {type(gtype)}")

            # Normalize fields to be a list of strings
            if isinstance(fields, str):
                fields = [fields]
            elif not isinstance(fields, (list, tuple)):
                raise TypeError("Fields must be a string or a list/tuple of strings.")

            self._broadcast_update_to_fields(gtype, fields, value)
            return

        raise TypeError(f"Unsupported index for __setitem__: {args}")

    def _bulk_update_from_df(self, gtype: Type[GObject], df: DataFrame):
        """Handles creating or overwriting objects from a complete DataFrame.

        This corresponds to the use case: `wb.pw[ObjectType] = dataframe`.

        Parameters
        ----------
        gtype : Type[GObject]
            The GObject subclass representing the type of objects to update.
        df : pandas.DataFrame
            The DataFrame containing object data. Columns must match PowerWorld
            field names, including primary keys.
        """
        if not isinstance(df, DataFrame):
            raise TypeError("A DataFrame is required for bulk updates.")
        self.esa.ChangeParametersMultipleElementRect(gtype.TYPE, df.columns.tolist(), df)

    def _broadcast_update_to_fields(self, gtype: Type[GObject], fields: list[str], value):
        """Modifies specific fields for existing objects by broadcasting a value.

        This corresponds to the use case: `wb.pw[ObjectType, 'FieldName'] = value`.

        Parameters
        ----------
        gtype : Type[GObject]
            The GObject subclass representing the type of objects to update.
        fields : List[str]
            A list of field names to update.
        value : Any
            The value to broadcast to the specified fields. Can be a scalar or
            a list/array if updating multiple fields on a keyless object.

        Raises
        ------
        ValueError
            If value length doesn't match field length for keyless objects.
        """
        # For objects without keys (e.g., Sim_Solution_Options), we construct
        # the change DataFrame directly without reading from PowerWorld first.
        if not gtype.keys:
            data_dict = {}
            if len(fields) == 1:
                data_dict[fields[0]] = [value]
            elif isinstance(value, (list, tuple)) and len(value) == len(fields):
                for i, field in enumerate(fields):
                    data_dict[field] = [value[i]]
            else:
                raise ValueError(
                    "For multiple fields on a keyless object, 'value' must be a list/tuple of the same length as the fields."
                )
            change_df = DataFrame(data_dict)
        
        # For objects with keys, we first get the keys of all existing objects
        # to ensure we only modify what's already there.
        else:
            keys = gtype.keys
            change_df = self[gtype, keys]
            
            if change_df is None or change_df.empty:
                # No objects of this type exist, so there's nothing to modify.
                return

            # Add the new values to the DataFrame of keys.
            # Pandas will broadcast a scalar `value` or align a list/array `value`.
            # When fields has a single element, use the field name directly to avoid pandas treating it as multiple columns
            if len(fields) == 1:
                change_df[fields[0]] = value
            else:
                change_df[fields] = value
        
        # Send the minimal DataFrame to PowerWorld.
        self.esa.ChangeParametersMultipleElementRect(gtype.TYPE, change_df.columns.tolist(), change_df)