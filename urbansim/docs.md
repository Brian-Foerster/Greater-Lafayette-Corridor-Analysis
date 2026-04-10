Docs for the `urbansim/` package

Purpose
- Describe how package modules map to the project: accounts, developer tools, maps, models, utils.

Structure
- `urbansim/developer/` - developer utilities (sqft proforma, etc.)
- `urbansim/maps/` - explorer helper and dframe_explorer server
- `urbansim/models/` - model code for supply-demand, transitions, relocation, regression
- `urbansim/utils/` - shared utilities: yaml I/O, sampling, networks, testing helpers

How to run
- Run package module-level tests with `pytest -q`.
- For interactive work, start a kernel with the conda environment and run `notebooks/Exploration.ipynb`.

Tests
- Unit tests live in `urbansim/tests/` and should be updated when package internals change.