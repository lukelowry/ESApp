"""Sphinx extension that renders a compact list of grid component classes.

Parses ``esapp/components/grid.py`` with the :mod:`ast` module (no import required) and
generates a multi-column HTML/LaTeX table of class names via the
``.. grid-component-list::`` directive.
"""

import ast
import os

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

        # Build a 4-column table
        n_cols = 4
        rows = [names[i : i + n_cols] for i in range(0, len(names), n_cols)]
        # Pad the last row
        if rows and len(rows[-1]) < n_cols:
            rows[-1].extend([""] * (n_cols - len(rows[-1])))

        table = nodes.table()
        table["classes"].append("grid-component-table")
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
                    entry += nodes.paragraph(text=cell_text)
                else:
                    entry += nodes.paragraph(text="")
                row_node += entry
            tbody += row_node

        # Add a count note above the table
        count_para = nodes.paragraph()
        count_para += nodes.strong(text=f"{len(names)} component types available")
        return [count_para, table]


def setup(app: Sphinx):
    app.add_directive("grid-component-list", GridComponentList)
    return {"version": "0.1", "parallel_read_safe": True}
