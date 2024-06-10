import asyncio
from functools import lru_cache
from pathlib import Path

import rattler
from conda.base.constants import ChannelPriority
from conda.base.context import context
from conda.common.constants import NULL
from conda.common.io import Spinner
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord
from conda_libmamba_solver.solver import LibMambaSolver
from conda_libmamba_solver.state import SolverInputState, SolverOutputState
from rattler import __version__ as rattler_version
from rattler.exceptions import SolverError as RattlerSolverError

from . import __version__
from .exceptions import RattlerUnsatisfiableError
from .index import RattlerIndexHelper


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
            except RattlerSolverError as exc:
                exc2 = RattlerUnsatisfiableError(str(exc))
                exc2.allow_retry = False
                raise exc2 from exc

        # Run post-solve tasks
        out_state.post_solve(solver=self)
        self.neutered_specs = tuple(out_state.neutered.values())

        return out_state.current_solution

    def _solve_attempt(
        self, in_state: SolverInputState, out_state: SolverOutputState, index: RattlerIndexHelper
    ):
        out_state.check_for_pin_conflicts(index)
        tasks = self._specs_to_tasks(in_state, out_state)
        # TODO: This is a hack to get the installed packages into the solver
        # but rattler doesn't allow PrefixRecords to be built through the API yet
        rattler_installed = {}
        for json_path in Path(self.prefix).glob("conda-meta/*.json"):
            name = json_path.stem.rsplit("-", 2)[0]
            record = rattler.PrefixRecord.from_path(json_path)
            rattler_installed[name] = record

        specs = []
        pinned_packages = []
        locked_packages = []
        for (task_name, _), task_specs in tasks.items():
            if task_name in ("INSTALL", "UPDATE"):
                specs.extend(task_specs)
            elif task_name in ("ADD_PIN", "USERINSTALLED"):
                for spec in task_specs:
                    for record in in_state.installed.values():
                        if MatchSpec(spec).match(record):
                            pinned_packages.append(rattler_installed[record.name])
            elif task_name == "LOCK":
                for spec in task_specs:
                    for record in in_state.installed.values():
                        if MatchSpec(spec).match(record):
                            locked_packages.append(rattler_installed[record.name])
        virtual_packages = []
        for pkg in in_state.virtual.values():
            virtual_packages.append(
                rattler.GenericVirtualPackage(
                    rattler.PackageName(pkg.name), rattler.Version(pkg.version), pkg.build
                )
            )

        return asyncio.run(
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
            )
        )

    def _export_solved_records(self, records, out_state):
        for record in records:
            if timestamp := record.timestamp:
                timestamp = int(timestamp.timestamp() * 1000)
            else:
                timestamp = 0
            out_state.records[record.name] = PackageRecord(
                name=record.name.source,
                version=str(record.version),
                build=record.build,
                build_number=record.build_number,
                channel=record.channel,
                subdir=record.subdir,
                fn=record.file_name,
                md5=record.md5,
                legacy_bz2_md5=record.legacy_bz2_md5,
                legacy_bz2_size=record.legacy_bz2_size,
                url=record.url,
                sha256=record.sha256,
                arch=record.arch,
                platform=record.platform,
                depends=record.depends or (),
                constrains=record.constrains or (),
                track_features=record.track_features or (),
                features=record.features or (),
                noarch=record.noarch,
                # preferred_env=record.preferred_env,
                license=record.license,
                license_family=record.license_family,
                # package_type=record.package_type,
                timestamp=timestamp,
                # date=record.date,
                size=record.size or 0,
            )
