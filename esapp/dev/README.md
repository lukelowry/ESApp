# PWRaw File Schema Description

**File Format:** Tab-Separated Values (TSV)
**Purpose:** Defines the data structure, variable types, and identifier logic for power system objects.

## 1. Column Definitions
The file is organized hierarchically: **Object Type** headers define the category (e.g., `Gen`), followed by rows of variables for that object.

| Col | Field Name | Content | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Object Type** | String | The object class (e.g., `Gen`, `Load`). *Only appears on the header row.* |
| **2** | **SUBDATA** | Flag | `Yes` if nested sub-data sections are allowed; otherwise blank. |
| **3** | **Key/Required** | Symbol | **The Priority Code.** Defines if a field is a Key (ID), a Base Value, or standard data. (See Section 2). |
| **4** | **Variable Name** | String | The internal variable ID used in scripting. |
| **5** | **Concise Name** | String | A shorter alias for the variable. |
| **6** | **Type** | Type | Data type: `Integer`, `Real` (float), or `String`. |
| **7** | **Description** | String | Human-readable explanation of the field. |
| **8** | **Available List** | String | The GUI menu path where this field is found. |
| **9** | **Enterable** | Flag | `Yes` = Editable by user. Blank = Read-only/Calculated. |

---

## 2. Key & Priority Legend (Column 3)
These symbols define exactly how the software identifies objects and prioritizes data.

| Symbol | Priority Type | Description |
| :--- | :--- | :--- |
| **`*`** | **Primary Key** | The main unique identifier (e.g., `BusNum` for a Bus). |
| **`*A*`** | **Alternate Key** | A unique Name/String that can replace the Primary Key (e.g., `BusName`). |
| **`*1*`** | **Composite Key 1** | The first part of a multi-part ID (usually "From Bus" or Location). |
| **`*2*`** | **Composite Key 2** | The second part of a multi-part ID (usually "To Bus"). |
| **`*3*`** | **Composite Key 3** | The third part of a multi-part ID (e.g., Tertiary winding bus). |
| **`*2B*`** | **Secondary ID** | A string ID used to distinguish devices at the same location (e.g., `GenID`). |
| **`*4B*`** | **Circuit ID** | A string ID used to distinguish parallel branches between the same buses. |
| **`**`** | **Base Value** | A fundamental physical parameter required for the model (e.g., `NomVolt`). |
| **`<`** | **Standard Field** | A regular property, setting, or status flag. |

---

## 3. Object Identifier Examples
How the symbols in **Column 3** combine to form unique keys for common objects:

| Object Type | Key Structure | Explanation |
| :--- | :--- | :--- |
| **Simple Object** | `*1*` | Identified by a single Number (e.g., **Bus**). |
| **Generator / Load** | `*1*` + `*2B*` | Identified by **Location** (BusNum) + **ID** (GenID/LoadID). |
| **Line / Branch** | `*1*` + `*2*` + `*4B*` | Identified by **From Bus** + **To Bus** + **Circuit ID**. |
| **3-Winding XF** | `*1*` + `*2*` + `*3*` | Identified by **Primary** + **Secondary** + **Tertiary** Bus Numbers. |

---

## 4. Primitive Data Types (Column 6)
* **Integer:** Whole numbers (e.g., Bus Numbers, Status flags).
* **Real:** Floating-point numbers (e.g., Voltage, MW, Resistance).
* **String:** Text (e.g., Names, "Yes/No", Labels).