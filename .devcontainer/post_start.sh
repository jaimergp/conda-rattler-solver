#!/bin/bash

# This script assumes we are running in a Miniconda container where:
# - /opt/conda is the Miniconda or Miniforge installation directory
# - https://github.com/conda/conda is mounted at /workspaces/conda
# - https://github.com/conda/conda-rattler-solver is mounted at
#   /workspaces/conda-rattler-solver
# - https://github.com/mamba-org/mamba is (optionally) mounted at
#   /workspaces/mamba

set -euo pipefail

BASE_CONDA=${BASE_CONDA:-/opt/conda}
SRC_CONDA=${SRC_CONDA:-/workspaces/conda}
SRC_CONDA_RATTLER_SOLVER=${SRC_CONDA_RATTLER_SOLVER:-/workspaces/conda-rattler-solver}

cd "$SRC_CONDA"
echo "Initializing conda in dev mode..."
"$BASE_CONDA/bin/python" -m conda init --dev bash
cd -

echo "Installing conda-rattler-solver in dev mode..."
"$BASE_CONDA/bin/python" -m pip install -e "$SRC_CONDA_RATTLER_SOLVER" --no-deps

set -x
conda list -p "$BASE_CONDA"
conda info
conda config --show-sources
