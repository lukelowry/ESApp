Development
===========

This section covers the internal maintenance and extension of the ESA++ toolkit, specifically focusing on how the library maintains its structured representation of PowerWorld objects to facilitate ObjectField access.

Updating Component Definitions
------------------------------

ESA++ uses a code generation script to build the component types that enable intuitive access to PowerWorld data. These classes (found in ``esapp.grid.components``) are auto-generated to ensure compatibility with different versions of PowerWorld Simulator.

When a new version of PowerWorld is released, or if you need to access newly added fields, follow this procedure to update the component definitions:

 **Export the Field List from PowerWorld**:
    *   Open PowerWorld Simulator.
    *   Navigate to the **Window** ribbon.
    *   Click on **Export Case Object Fields**.
    *   Save the resulting tab-delimited text file.

**Prepare the Raw Data**:
    *   Rename the exported file to ``PWRaw``.
    *   Place this file in the ``esapp/grid/`` folder, overwriting the existing one.

**Run the Generation Script**:
    *   Open a terminal in the project root.
    *   Execute the generation script:
        .. code-block:: bash

            python esapp/grid/generate_components.py

**Verify the Changes**:
    *   The script will parse ``PWRaw`` and update ``esapp/grid/components.py``.
    *   Check the console output for any errors or excluded objects/fields.

The ``generate_components.py`` script handles the sanitization of field names (e.g., converting ``Bus:Num`` to ``Bus__Num``) and assigns priorities to fields based on whether they are primary keys, required, or optional. This automation allows ESA++ to support the vast array of objects and fields available in PowerWorld without manual coding for each component.

Generally, this is the only step required to keep ESA++ compatible with new PowerWorld releases regarding data access and modification.