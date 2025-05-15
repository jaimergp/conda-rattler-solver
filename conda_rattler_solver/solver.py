from __future__ import annotations

import asyncio
import json
import logging
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import rattler
from conda.base.constants import ChannelPriority
from conda.base.context import context
from conda.common.constants import NULL
from conda.exceptions import PackagesNotFoundError
from conda.models.match_spec import MatchSpec
from conda.reporters import get_spinner
from conda_libmamba_solver.solver import LibMambaSolver
from conda_libmamba_solver.state import SolverInputState, SolverOutputState
from libmambapy.solver import Request
from rattler import __version__ as rattler_version
from rattler.exceptions import SolverError as RattlerSolverError

from . import __version__
from .exceptions import RattlerUnsatisfiableError
from .index import RattlerIndexHelper
from .utils import rattler_record_to_conda_record

if TYPE_CHECKING:
    from boltons.setutils import IndexedSet
    from conda.auxlib import _Null
    from conda.base.constants import (
        DepsModifier,
        UpdateModifier,
    )
    from conda.models.records import PackageRecord

log = logging.getLogger(f"conda.{__name__}")


class RattlerSolver(LibMambaSolver):
    @staticmethod
    @cache
    def user_agent() -> str:
        """
        Expose this identifier to allow conda to extend its user agent if required
        """
        return f"conda-rattler-solver/{__version__} py-rattler/{rattler_version}"

    def solve_final_state(
        self,
        update_modifier: UpdateModifier | _Null = NULL,
        deps_modifier: DepsModifier | _Null = NULL,
        prune: bool | _Null = NULL,
        ignore_pinned: bool | _Null = NULL,
        force_remove: bool | _Null = NULL,
        should_retry_solve: bool = False,
    ) -> IndexedSet[PackageRecord]:
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

        channels = self._collect_channel_list(in_state)
        conda_build_channels = self._collect_channels_subdirs_from_conda_build(
            seen=set(channels)
        )
        with get_spinner(
            self._collect_all_metadata_spinner_message(channels, conda_build_channels),
        ):
            index = RattlerIndexHelper(all_channels, self.subdirs, self._repodata_fn)

        with get_spinner(
            self._solving_loop_spinner_message(),
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
    ) -> SolverOutputState:
        solution = None
        out_state.check_for_pin_conflicts(index)
        for attempt in range(1, self._max_attempts(in_state) + 1):
            log.debug("Starting solver attempt %s", attempt)
            solution = self._solve_attempt(in_state, out_state, index, attempt=attempt)
            if solution is not None and not isinstance(solution, Exception):
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
            solution = self._solve_attempt(
                in_state, out_state, index, attempt=attempt + 1
            )
            if isinstance(solution, Exception) or solution is None:
                exc = RattlerUnsatisfiableError(solution or "Could not find solution")
                exc.allow_retry = False
                raise exc

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
    ) -> RattlerSolverError | list[rattler.RepoDataRecord]:
        if in_state.is_removing:
            jobs = self._specs_to_request_jobs_remove(in_state, out_state)
            jobs = {Request.Remove: jobs.pop(Request.Remove, ()), **jobs}
        elif self._called_from_conda_build():
            jobs = self._specs_to_request_jobs_conda_build(in_state, out_state)
        else:
            jobs = self._specs_to_request_jobs_add(in_state, out_state)
        virtual_packages = [
            rattler.GenericVirtualPackage(
                rattler.PackageName(pkg.name), rattler.Version(pkg.version), pkg.build
            )
            for pkg in in_state.virtual.values()
        ]
        # Request.Pin,
        # Request.Install,
        # Request.Update,
        # Request.Keep,
        # Request.Freeze,
        # Request.Remove,
        remove = [
            MatchSpec(spec).name
            for request, task_specs in jobs.items()
            for spec in task_specs
            if request in (Request.Remove, Request.Update)
            and "*" not in MatchSpec(spec).name
        ]

        specs = []
        constrained_specs = []
        pinned_packages = []
        locked_packages = []
        for request, task_specs in jobs.items():
            if request in (Request.Install, Request.Update):
                specs.extend(task_specs)
            elif request == Request.Remove:
                for spec in task_specs:
                    match_spec = MatchSpec(spec)
                    if "*" in match_spec.name:
                        # TODO: Improve logic here; too many loops
                        for pkg_name, pkg in in_state.installed.items():
                            if match_spec.match(pkg):
                                constrained_specs.append(f"{pkg_name}<0.0.0a0")
                                remove.append(pkg_name)
                    else:
                        remove.append(match_spec.name)
                        constrained_specs.append(f"{match_spec.name}<0.0.0a0")
            elif request == Request.Pin:
                constrained_specs.extend(task_specs)
            elif request == Request.Keep:
                for spec in task_specs:
                    if MatchSpec(spec).name in remove:
                        continue
                    if not in_state.is_removing:
                        specs.append(spec)
                    for record in in_state.installed.values():
                        if MatchSpec(spec).match(record):
                            locked_packages.append(
                                self._prefix_record_to_rattler_prefix_record(record)
                            )
            elif request == Request.Freeze:
                for spec in task_specs:
                    if MatchSpec(spec).name in remove:
                        continue
                    for record in in_state.installed.values():
                        if MatchSpec(spec).match(record):
                            pinned_packages.append(
                                self._prefix_record_to_rattler_prefix_record(record)
                            )
        # remove any packages that should be updated from the locked_packages
        locked_packages = [
            record for record in locked_packages if record.name not in remove
        ]

        # print("specs=", *[rattler.MatchSpec(s) for s in specs])
        # print("locked_packages=", *locked_packages)
        # print("pinned_packages=", *pinned_packages)
        # print("virtual_packages=", *virtual_packages)
        # print("constraints=", *[rattler.MatchSpec(s) for s in constrained_specs])
        # print("index=", index._index)
        try:
            solution = asyncio.run(
                rattler.solve_with_sparse_repodata(
                    specs=[
                        rattler.MatchSpec(str(s).rstrip("=").replace("=[", "["))
                        for s in specs
                    ],
                    sparse_repodata=[info.repo for info in index._index.values()],
                    locked_packages=locked_packages,
                    pinned_packages=pinned_packages,
                    virtual_packages=virtual_packages,
                    channel_priority=(
                        rattler.ChannelPriority.Strict
                        if context.channel_priority == ChannelPriority.STRICT
                        else rattler.ChannelPriority.Disabled
                    ),
                    strategy="highest",
                    constraints=[rattler.MatchSpec(str(s)) for s in constrained_specs],
                )
            )
        except RattlerSolverError as exc:
            self._maybe_raise_for_problems(str(exc), out_state)
            return exc
        else:
            out_state.conflicts.clear()
            return solution

    def _maybe_raise_for_problems(self, problems: str, out_state: SolverOutputState):
        unsatisfiable = {}
        not_found = {}
        for line in problems.splitlines():
            line = line.strip(" ─│└├")
            if line.startswith("Cannot solve the request because of:"):
                line = line.split(":", 1)[1]
            words = line.split()
            if "is locked, but another version is required as reported above" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1]}")
            elif (
                "which cannot be installed because there are no viable options" in line
            ):
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1].strip(',')}")
            elif "cannot be installed because there are no viable options" in line:
                unsatisfiable[words[0]] = MatchSpec(f"{words[0]} {words[1]}")
            elif "the constraint" in line and "cannot be fulfilled" in line:
                unsatisfiable[words[2]] = MatchSpec(" ".join(words[2:-3]))
            elif (
                "can be installed with any of the following options" in line
                and "which" not in line
            ):
                position = line.index(" can be installed with")
                unsatisfiable[words[0]] = MatchSpec(line[:position])
            elif "No candidates were found for" in line:
                position = line.index("No candidates were found for ")
                position += len("No candidates were found for ")
                spec = line[position:]
                not_found[spec.split()[0]] = MatchSpec(spec)
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
        out_state.records.clear()
        for rattler_record in records:
            conda_record = rattler_record_to_conda_record(rattler_record)
            out_state.records[conda_record.name] = conda_record

    @cache
    def _prefix_record_to_rattler_prefix_record(self, record):
        # TODO: This is a hack to get the installed packages into the solver
        # but rattler doesn't allow PrefixRecords to be built through the API yet
        with NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            json.dump(record.dump(), tmp)
        try:
            return rattler.PrefixRecord.from_path(tmp.name)
        finally:
            Path(tmp.name).unlink()
