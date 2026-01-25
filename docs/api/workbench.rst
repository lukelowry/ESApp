GridWorkBench
=============

The ``GridWorkBench`` is the high-level entry point for interacting with PowerWorld via ESA++. It wraps
SimAuto with a Pythonic interface for case management, data access, and analysis helpers. For concepts and
usage patterns, see :doc:`../guide/usage`. This page lists the full API surface.

.. currentmodule:: esapp.workbench

.. autoclass:: GridWorkBench

.. rubric:: Methods

.. autosummary::
   :toctree: generated/
   :template: method.rst

   ~GridWorkBench.areas
   ~GridWorkBench.auto_insert_contingencies
   ~GridWorkBench.branch_admittance
   ~GridWorkBench.buscoords
   ~GridWorkBench.busmap
   ~GridWorkBench.calculate_gic
   ~GridWorkBench.close
   ~GridWorkBench.close_branch
   ~GridWorkBench.command
   ~GridWorkBench.create
   ~GridWorkBench.deenergize
   ~GridWorkBench.delete
   ~GridWorkBench.diff_mode
   ~GridWorkBench.edit_mode
   ~GridWorkBench.energize
   ~GridWorkBench.fault
   ~GridWorkBench.flatstart
   ~GridWorkBench.generations
   ~GridWorkBench.gens_above_pmax
   ~GridWorkBench.gens_above_qmax
   ~GridWorkBench.gic_clear
   ~GridWorkBench.gic_load_b3d
   ~GridWorkBench.gic_storm
   ~GridWorkBench.gmatrix
   ~GridWorkBench.incidence_matrix
   ~GridWorkBench.islands
   ~GridWorkBench.jacobian
   ~GridWorkBench.lines
   ~GridWorkBench.loads
   ~GridWorkBench.load_aux
   ~GridWorkBench.load_script
   ~GridWorkBench.lodf
   ~GridWorkBench.log
   ~GridWorkBench.mismatches
   ~GridWorkBench.network_cut
   ~GridWorkBench.open
   ~GridWorkBench.open_branch
   ~GridWorkBench.path_distance
   ~GridWorkBench.pflow
   ~GridWorkBench.print_log
   ~GridWorkBench.ptdf
   ~GridWorkBench.radial_paths
   ~GridWorkBench.refresh_onelines
   ~GridWorkBench.reset
   ~GridWorkBench.run_contingency
   ~GridWorkBench.run_mode
   ~GridWorkBench.save
   ~GridWorkBench.scale_gen
   ~GridWorkBench.scale_load
   ~GridWorkBench.select
   ~GridWorkBench.set_as_base_case
   ~GridWorkBench.set_esa
   ~GridWorkBench.set_gen
   ~GridWorkBench.set_load
   ~GridWorkBench.set_voltages
   ~GridWorkBench.shortest_path
   ~GridWorkBench.shunts
   ~GridWorkBench.shunt_admittance
   -GridWorkBench.solve_contingencies
   ~GridWorkBench.solve_opf
   ~GridWorkBench.transformers
   ~GridWorkBench.unselect
   ~GridWorkBench.violations
   ~GridWorkBench.voltage
   ~GridWorkBench.voltages
   ~GridWorkBench.write_voltage
   ~GridWorkBench.ybus
   ~GridWorkBench.zones
