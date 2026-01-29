"""Sphinx extension that renders compact lists of grid components and TS fields.

Parses ``esapp/components/grid.py`` and ``esapp/components/ts_fields.py`` with the
:mod:`ast` module (no import required) and generates multi-column HTML/LaTeX tables
via the ``.. grid-component-list::`` and ``.. ts-field-list::`` directives.
"""

import ast
import os
import re

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


def _extract_class_names(grid_path: str):
    """Return a sorted list of class names defined in *grid_path*."""
    with open(grid_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=grid_path)
    return sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    )


def _extract_ts_fields(ts_fields_path: str):
    """Extract TS field information from ts_fields.py.

    Returns a dict mapping category names to lists of (field_name, pw_name, description) tuples.
    """
    with open(ts_fields_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=ts_fields_path)

    categories = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TS":
            # Find nested classes inside TS
            for child in node.body:
                if isinstance(child, ast.ClassDef):
                    category_name = child.name
                    fields = []
                    for item in child.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    field_name = target.name
                                    # Extract TSField arguments
                                    if isinstance(item.value, ast.Call):
                                        args = item.value.args
                                        pw_name = ""
                                        description = ""
                                        if len(args) >= 1 and isinstance(args[0], ast.Constant):
                                            pw_name = args[0].value
                                        if len(args) >= 2 and isinstance(args[1], ast.Constant):
                                            description = args[1].value
                                        fields.append((field_name, pw_name, description))
                    if fields:
                        categories[category_name] = sorted(fields, key=lambda x: x[0])
    return categories


def _build_compact_table(items, n_cols, table_class, is_monospace=False):
    """Build a compact multi-column table node.

    Args:
        items: List of strings to display in cells
        n_cols: Number of columns
        table_class: CSS class for the table
        is_monospace: If True, use literal nodes for monospace display

    Returns:
        A docutils table node
    """
    rows = [items[i : i + n_cols] for i in range(0, len(items), n_cols)]
    # Pad the last row
    if rows and len(rows[-1]) < n_cols:
        rows[-1].extend([""] * (n_cols - len(rows[-1])))

    table = nodes.table()
    table["classes"].append(table_class)
    tgroup = nodes.tgroup(cols=n_cols)
    table += tgroup

    for _ in range(n_cols):
        tgroup += nodes.colspec(colwidth=1)

    tbody = nodes.tbody()
    tgroup += tbody

    for row_data in rows:
        row_node = nodes.row()
        for cell_text in row_data:
            entry = nodes.entry()
            if cell_text:
                if is_monospace:
                    entry += nodes.paragraph("", "", nodes.literal(text=cell_text))
                else:
                    entry += nodes.paragraph(text=cell_text)
            else:
                entry += nodes.paragraph(text="")
            row_node += entry
        tbody += row_node

    return table


class GridComponentList(Directive):
    """Render all grid component class names as a compact table."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self):
        # Locate esapp/components/grid.py relative to the docs/ directory
        docs_dir = os.path.dirname(self.state.document.settings.env.app.srcdir)
        grid_path = os.path.join(docs_dir, "esapp", "components", "grid.py")

        if not os.path.isfile(grid_path):
            error = self.state_machine.reporter.error(
                f"grid_list: cannot find {grid_path}",
                nodes.literal_block(self.block_text, self.block_text),
                line=self.lineno,
            )
            return [error]

        names = _extract_class_names(grid_path)
        table = _build_compact_table(names, n_cols=4, table_class="grid-component-table")

        # Add a count note above the table
        count_para = nodes.paragraph()
        count_para += nodes.strong(text=f"{len(names)} component types available")
        return [count_para, table]


class TSFieldList(Directive):
    """Render TS field constants as compact tables organized by category."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "category": lambda x: x,  # Optional: filter to single category
    }

    def run(self):
        # Locate esapp/components/ts_fields.py relative to the docs/ directory
        docs_dir = os.path.dirname(self.state.document.settings.env.app.srcdir)
        ts_path = os.path.join(docs_dir, "esapp", "components", "ts_fields.py")

        if not os.path.isfile(ts_path):
            error = self.state_machine.reporter.error(
                f"ts_field_list: cannot find {ts_path}",
                nodes.literal_block(self.block_text, self.block_text),
                line=self.lineno,
            )
            return [error]

        categories = _extract_ts_fields(ts_path)
        filter_category = self.options.get("category")

        result_nodes = []
        total_fields = 0

        # Sort categories for consistent output
        cat_order = ["Area", "Branch", "Bus", "Gen", "InjectionGroup", "Load", "Shunt", "Substation", "System"]
        sorted_cats = [c for c in cat_order if c in categories]
        # Add any remaining categories not in the predefined order
        sorted_cats.extend([c for c in sorted(categories.keys()) if c not in cat_order])

        for cat_name in sorted_cats:
            if filter_category and cat_name != filter_category:
                continue

            fields = categories[cat_name]
            total_fields += len(fields)

            # Create section for this category
            section = nodes.section()
            section["ids"].append(f"ts-{cat_name.lower()}-fields")

            # Add heading
            title = nodes.title(text=f"TS.{cat_name}")
            section += title

            # Build a 3-column table: Field, PowerWorld Name, Description
            table = nodes.table()
            table["classes"].append("ts-field-table")
            table["classes"].append("longtable")
            tgroup = nodes.tgroup(cols=3)
            table += tgroup

            # Column specs - give description more space
            tgroup += nodes.colspec(colwidth=15)
            tgroup += nodes.colspec(colwidth=20)
            tgroup += nodes.colspec(colwidth=65)

            # Header
            thead = nodes.thead()
            tgroup += thead
            header_row = nodes.row()
            for header_text in ["Field", "PowerWorld Name", "Description"]:
                entry = nodes.entry()
                entry += nodes.paragraph(text=header_text)
                header_row += entry
            thead += header_row

            # Body
            tbody = nodes.tbody()
            tgroup += tbody

            for field_name, pw_name, description in fields:
                row = nodes.row()

                # Field name (monospace)
                entry1 = nodes.entry()
                entry1 += nodes.paragraph("", "", nodes.literal(text=field_name))
                row += entry1

                # PowerWorld name (monospace, smaller)
                entry2 = nodes.entry()
                entry2 += nodes.paragraph("", "", nodes.literal(text=pw_name))
                row += entry2

                # Description - clean up DSC:: prefixes
                entry3 = nodes.entry()
                desc_clean = description
                if desc_clean.startswith("DSC::"):
                    desc_clean = desc_clean[5:]  # Remove DSC:: prefix
                entry3 += nodes.paragraph(text=desc_clean if desc_clean else "—")
                row += entry3

                tbody += row

            section += table
            result_nodes.append(section)

        # Add summary at the top
        summary = nodes.paragraph()
        summary += nodes.strong(text=f"{total_fields} TS field constants across {len(sorted_cats)} categories")

        return [summary] + result_nodes


def setup(app: Sphinx):
    app.add_directive("grid-component-list", GridComponentList)
    app.add_directive("ts-field-list", TSFieldList)

    # Add custom CSS for better table rendering
    app.add_css_file("custom_tables.css")

    return {"version": "0.2", "parallel_read_safe": True}
