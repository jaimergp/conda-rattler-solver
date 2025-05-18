# Copyright (C) 2022 Anaconda, Inc
# Copyright (C) 2023 conda
# SPDX-License-Identifier: BSD-3-Clause
"""
Ensure the right User-Agent headers are set by `conda.context`.
These will be sent to the server on each request.
"""

import json
import os
from importlib.metadata import version
from subprocess import check_output

import pytest


@pytest.mark.parametrize("solver", ("classic", "rattler"))
def test_user_agent_conda_info(solver):
    env = os.environ.copy()
    env["CONDA_SOLVER"] = solver
    out = check_output(["conda", "info", "--json"], env=env)
    info = json.loads(out)
    assert "conda/" in info["user_agent"]
    if solver == "classic":
        assert "solver/" not in info["user_agent"]
    elif solver == "rattler":
        assert "solver/rattler" in info["user_agent"]
        assert f"conda-rattler-solver/{version('conda-rattler-solver')}" in info["user_agent"]
        assert f"rattler/{version('py-rattler')}" in info["user_agent"]
