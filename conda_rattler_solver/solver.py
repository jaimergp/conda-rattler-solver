from functools import lru_cache
from pathlib import Path

from boltons.setutils import IndexedSet
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import Spinner
from conda.core.subdir_data import SubdirData
from conda.models.channel import Channel
from conda.models.match_spec import MatchSpec
from conda.models.prefix_graph import PrefixGraph
from conda_libmamba_solver.solver import LibMambaSolver
from conda_libmamba_solver.state import SolverInputState, SolverOutputState
from rattler import __version__ as rattler_version
from rattler import (
    solve,
    MatchSpec as RattlerMatchSpec,
    PrefixRecord as RattlerPrefixRecord,
    VirtualPackage,
)

from . import __version__
from .exceptions import RattlerUnsatisfiableError
from .index import RattlerIndexHelper

from . import __version__


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
            "Solving environment",
            enabled=not context.verbosity and not context.quiet,
            json=context.json,
        ):
            try:
                records = self._solve_attempt(in_state, out_state, index)
                self._export_solved_records(records, out_state)
            except Exception as exc:
                if exc.__class__.__name__.split(".")[-1] == "SolverException":
                    exc2 = RattlerUnsatisfiableError(str(exc))
                    exc2.allow_retry = False
                    raise exc2
                raise exc

        # Run post-solve tasks
        out_state.post_solve(solver=self)
        self.neutered_specs = tuple(out_state.neutered.values())

        return out_state.current_solution

    def _solve_attempt(self, in_state, out_state, index):
        out_state.prepare_specs(index)
        tasks = self._specs_to_tasks(in_state, out_state)
        # TODO: This is a hack to get the installed packages into the solver
        # but rattler doesn't allow PrefixRecords to be passed in yet
        # rattler_installed = {}
        # for json_path in Path(self.prefix).glob("conda-meta/*.json"):
        #     name = json_path.stem.rsplit("-", 2)[0]
        #     record = RattlerPrefixRecord.from_path(json_path)
        #     rattler_installed[name] = record

        specs = []
        pins = []
        locked = []
        for (task_name, _), task_specs in tasks.items():
            if task_name in ("INSTALL", "UPDATE"):
                specs.extend(task_specs)
            # # TODO
            # elif task_name in ("ADD_PIN", "USERINSTALLED"):
            #     for spec in task_specs:
            #         for record in in_state.installed.values():
            #             if MatchSpec(spec).match(record):
            #                 pins.append(rattler_installed[record.name])
            # elif task_name == "LOCK":
            #     for spec in task_specs:
            #         for record in in_state.installed.values():
            #             if MatchSpec(spec).match(record):
            #                 locked.append(rattler_installed[record.name])

        return solve(
            specs=[RattlerMatchSpec(s) for s in specs],
            available_packages=[info.repo for info in index._index.values()],
            # locked_packages=locked,  # TODO
            # pinned_packages=pins,  # TODO
            virtual_packages=[p.into_generic() for p in VirtualPackage.current()],
        )

    def _export_solved_records(self, records, out_state):
        # rattler doesn't return the full record, so we need to query conda's SubdirData
        # which has a huge overhead
        for record in records:
            sd = SubdirData(Channel(record.url.rsplit("/", 1)[0]), repodata_fn=self._repodata_fn)
            conda_records = list(sd.query_all(str(record.package_record)))
            assert len(conda_records) == 1, conda_records
            conda_record = conda_records[0]
            out_state.records.set(
                conda_record.name, conda_record, reason="Part of solution calculated by rattler"
            )
