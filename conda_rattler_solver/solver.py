import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import rattler
from conda.base.constants import ChannelPriority
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import Spinner
from conda.exceptions import PackagesNotFoundError
from conda.models.match_spec import MatchSpec
from conda_libmamba_solver.solver import LibMambaSolver
from conda_libmamba_solver.state import SolverInputState, SolverOutputState
from rattler import __version__ as rattler_version
from rattler.exceptions import SolverError as RattlerSolverError

from . import __version__
from .exceptions import RattlerUnsatisfiableError
from .index import RattlerIndexHelper
from .utils import rattler_record_to_conda_record

log = logging.getLogger(f"conda.{__name__}")


class RattlerSolver(LibMambaSolver):
    @staticmethod
    @lru_cache(maxsize=None)
    def user_agent():
        """
        Expose this identifier to allow conda to extend its user agent if required
        """
        return f"conda-rattler-solver/{__version__} py-rattler/{rattler_version}"

    def solve_final_state(
        self,
        update_modifier=NULL,
        deps_modifier=NULL,
        prune=NULL,
        ignore_pinned=NULL,
        force_remove=NULL,
        should_retry_solve=False,
    ):
        in_state = SolverInputState(
            prefix=self.prefix,
            requested=self.specs_to_add or self.specs_to_remove,
            update_modifier=update_modifier,
            deps_modifier=deps_modifier,
            prune=prune,
            ignore_pinned=ignore_pinned,
            force_remove=force_remove,
            command=self._command,
        )

        out_state = SolverOutputState(solver_input_state=in_state)

        # These tasks do _not_ require a solver...
        # TODO: Abstract away in the base class?
        none_or_final_state = out_state.early_exit()
        if none_or_final_state is not None:
            return none_or_final_state

        all_channels = [
            *self.channels,
            *in_state.channels_from_specs(),
            *in_state.maybe_free_channel(),
        ]

        with Spinner(
            self._spinner_msg_metadata(all_channels),
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            index = RattlerIndexHelper(all_channels, self.subdirs, self._repodata_fn)

        with Spinner(
            self._spinner_msg_solving(),
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            # This function will copy and mutate `out_state`
            # Make sure we get the latest copy to return the correct solution below
            out_state = self._solving_loop(in_state, out_state, index)
            self.neutered_specs = tuple(out_state.neutered.values())
            solution = out_state.current_solution

        # Check whether conda can be updated; this is normally done in .solve_for_diff()
        # but we are doing it now so we can reuse in_state and friends
        self._notify_conda_outdated(None, index, solution)

        return solution

    def _solving_loop(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: RattlerIndexHelper,
    ):
        solution = None
        out_state.check_for_pin_conflicts(index)
        for attempt in range(1, self._max_attempts(in_state) + 1):
            log.debug("Starting solver attempt %s", attempt)
            solution = self._solve_attempt(in_state, out_state, index, attempt=attempt)
            if solution is not None:
                break
            out_state = SolverOutputState(
                solver_input_state=in_state,
                records=dict(out_state.records),
                for_history=dict(out_state.for_history),
                neutered=dict(out_state.neutered),
                conflicts=dict(out_state.conflicts),
                pins=dict(out_state.pins),
            )
        else:
            # Didn't find a solution yet, let's unfreeze everything
            out_state.conflicts.update(
                {
                    name: record.to_match_spec()
                    for name, record in in_state.installed.items()
                    if not record.is_unmanageable
                }
            )
            solution = self._solve_attempt(in_state, out_state, index, attempt=attempt + 1)
            if solution is None:
                raise RattlerUnsatisfiableError("Could not find solution")

        # We didn't fail? Nice, let's return the calculated state
        self._export_solved_records(solution, out_state)

        # Run post-solve tasks
        out_state.post_solve(solver=self)

        return out_state

    def _solve_attempt(
        self,
        in_state: SolverInputState,
        out_state: SolverOutputState,
        index: RattlerIndexHelper,
        attempt: int = 0,
    ):
        tasks = self._specs_to_tasks(in_state, out_state)
        virtual_packages = [
            rattler.GenericVirtualPackage(
                rattler.PackageName(pkg.name), rattler.Version(pkg.version), pkg.build
            )
            for pkg in in_state.virtual.values()
        ]
        remove = [
            MatchSpec(spec).name
            for (task_name, _), task_specs in tasks.items()
            for spec in task_specs
            if task_name.startswith("ERASE")
        ]

        specs = []
        constrained_specs = []
        pinned_packages = []
        locked_packages = []
        for (task_name, _), task_specs in tasks.items():
            if task_name in ("INSTALL", "UPDATE"):
                specs.extend(task_specs)
            elif task_name.startswith("ERASE"):
                constrained_specs.extend(
                    [f"{MatchSpec(spec).name}<0.0.0a0" for spec in task_specs]
                )
            elif task_name == "ADD_PIN":
                constrained_specs.extend(task_specs)
            elif task_name in "USERINSTALLED":
                for spec in task_specs:
                    if MatchSpec(spec).name in remove:
                        continue
                    for record in in_state.installed.values():
                        if MatchSpec(spec).match(record):
                            locked_packages.append(
                                self._prefix_record_to_rattler_prefix_record(record)
                            )
            elif task_name == "LOCK":
                for spec in task_specs:
                    if MatchSpec(spec).name in remove:
                        continue
                    for record in in_state.installed.values():
                        if MatchSpec(spec).match(record):
                            pinned_packages.append(
                                self._prefix_record_to_rattler_prefix_record(record)
                            )
        # print("specs=", *[rattler.MatchSpec(s) for s in specs])
        # print("locked_packages=", *locked_packages)
        # print("pinned_packages=", *pinned_packages)
        # print("virtual_packages=", *virtual_packages)
        # print("constraints=", *[rattler.MatchSpec(s) for s in constrained_specs])
        try:
            solution = asyncio.run(
                rattler.solve_with_sparse_repodata(
                    specs=[rattler.MatchSpec(s) for s in specs],
                    sparse_repodata=[info.repo for info in index._index.values()],
                    locked_packages=locked_packages,
                    pinned_packages=pinned_packages,
                    virtual_packages=virtual_packages,
                    channel_priority=rattler.ChannelPriority.Strict
                    if context.channel_priority == ChannelPriority.STRICT
                    else rattler.ChannelPriority.Disabled,
                    strategy="highest",
                    constraints=[rattler.MatchSpec(s) for s in constrained_specs],
                )
            )
        except RattlerSolverError as exc:
            self._maybe_raise_for_problems(str(exc), out_state)
        else:
            out_state.conflicts.clear()
            return solution

    def _maybe_raise_for_problems(self, problems: str, out_state: SolverOutputState):
        unsatisfiable = {}
        not_found = {}
        for line in problems.splitlines():
            line = line.strip(" ─│└├")
            words = line.split()
            if "is locked, but another version is required as reported above" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1]}")
            elif "which cannot be installed because there are no viable options" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1].strip(',')}")
            elif "cannot be installed because there are no viable options" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1]}")

        if not unsatisfiable and not_found:
            log.debug(
                "Inferred PackagesNotFoundError %s from conflicts:\n%s",
                tuple(not_found.keys()),
                problems,
            )
            # This is not a conflict, but a missing package in the channel
            exc = PackagesNotFoundError(tuple(not_found.values()), tuple(self.channels))
            exc.allow_retry = False
            raise exc

        previous = out_state.conflicts or {}
        previous_set = set(previous.values())
        current_set = set(unsatisfiable.values())

        diff = current_set.difference(previous_set)
        if len(diff) > 1 and "python" in unsatisfiable:
            # Only report python as conflict if it's the only conflict reported
            # This helps us prioritize neutering for other dependencies first
            unsatisfiable.pop("python")

        if (previous and (previous_set == current_set)) or len(diff) >= 10:
            # We have same or more (up to 10) unsatisfiable now! Abort to avoid recursion
            exc = RattlerUnsatisfiableError(problems)
            # do not allow conda.cli.install to try more things
            exc.allow_retry = False
            raise exc

        log.debug("Attempt failed with %s conflicts:\n%s", len(unsatisfiable), problems)
        out_state.conflicts.update(unsatisfiable)

    def _export_solved_records(self, records, out_state):
        for rattler_record in records:
            conda_record = rattler_record_to_conda_record(rattler_record)
            out_state.records[conda_record.name] = conda_record

    @lru_cache(maxsize=None)
    def _prefix_record_to_rattler_prefix_record(self, record):
        # TODO: This is a hack to get the installed packages into the solver
        # but rattler doesn't allow PrefixRecords to be built through the API yet
        with NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(record.dump(), tmp)
        try:
            return rattler.PrefixRecord.from_path(tmp.name)
        finally:
            Path(tmp.name).unlink()
