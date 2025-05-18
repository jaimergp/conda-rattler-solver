from __future__ import annotations

import json
import os
import re
import sys
from contextlib import suppress
from logging import getLogger
from textwrap import dedent
from typing import TYPE_CHECKING

import rattler
from conda import __version__ as _conda_version
from conda.base.constants import KNOWN_SUBDIRS, REPODATA_FN, UNKNOWN_CHANNEL
from conda.base.context import context
from conda.common.path import paths_equal
from conda.models.match_spec import MatchSpec
from conda.models.records import PackageRecord, PrefixRecord
from conda.models.version import VersionOrder

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from conda.common.path import PathType

    from .index import RattlerIndexHelper

log = getLogger(f"conda.{__name__}")


def _hash_to_str(bytes_or_str: bytes | str | None) -> None | str:
    if not bytes_or_str:
        return None
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.hex()
    return bytes_or_str.lower()


def rattler_record_to_conda_record(record: rattler.PackageRecord) -> PackageRecord:
    if timestamp := record.timestamp:
        timestamp = int(timestamp.timestamp() * 1000)
    else:
        timestamp = 0

    if record.noarch.none:
        noarch = None
    elif record.noarch.python:
        noarch = "python"
    elif record.noarch.generic:
        noarch = "generic"
    else:
        raise ValueError(f"Unknown noarch type: {record.noarch}")

    if record.channel.endswith(
        (
            "noarch",
            *KNOWN_SUBDIRS,
        )
    ):
        channel_url = record.channel
    else:
        channel_url = (f"{record.channel}/{record.subdir}",)

    return PackageRecord(
        name=record.name.source,
        version=str(record.version),
        build=record.build,
        build_number=record.build_number,
        channel=channel_url,
        subdir=record.subdir,
        fn=record.file_name,
        md5=_hash_to_str(record.md5),
        legacy_bz2_md5=_hash_to_str(record.legacy_bz2_md5),
        legacy_bz2_size=record.legacy_bz2_size,
        url=record.url,
        sha256=_hash_to_str(record.sha256),
        arch=record.arch,
        platform=record.platform,
        depends=record.depends or (),
        constrains=record.constrains or (),
        track_features=record.track_features or (),
        features=record.features or (),
        noarch=noarch,
        # preferred_env=record.preferred_env,
        license=record.license,
        license_family=record.license_family,
        # package_type=record.package_type,
        timestamp=timestamp,
        # date=record.date,
        size=record.size or 0,
        python_site_packages_path=record.python_site_packages_path,
    )


def conda_prefix_record_to_rattler_prefix_record(
    record: PrefixRecord,
) -> rattler.PrefixRecord:
    package_record = rattler.PackageRecord(
        name=record.name,
        version=record.version,
        build=record.build,
        build_number=record.build_number,
        subdir=record.subdir,
        arch=record.get("arch"),
        platform=record.get("platform"),
        noarch=rattler.NoArchType(record.get("noarch")),  # BUGGY
        depends=record.get("depends"),
        constrains=record.get("constrains"),
        sha256=bytes.fromhex(record.get("sha256") or "") or None,
        md5=bytes.fromhex(record.get("md5", "") or "") or None,
        size=record.get("size"),
        features=record.get("features") or None,
        legacy_bz2_md5=bytes.fromhex(record.get("legacy_bz2_md5", "") or "") or None,
        legacy_bz2_size=bytes.fromhex(record.get("legacy_bz2_size", "") or "") or None,
        license=record.get("license"),
        license_family=record.get("license_family"),
        python_site_packages_path=record.get("python_site_packages_path"),
    )
    repodata_record = rattler.RepoDataRecord(
        package_record=package_record,
        file_name=record.fn,
        url=record.url,
        channel=record.channel.base_url,
    )
    paths_data = rattler.PrefixPaths()
    path_entries = []
    for path in record.paths_data.paths:
        kwargs = {
            "relative_path": path.path,
            "path_type": rattler.PrefixPathType(str(path.path_type)),
            "prefix_placeholder": getattr(path, "prefix_placeholder", None),
            "sha256": bytes.fromhex(getattr(path, "sha256", "")) or None,
            "sha256_in_prefix": bytes.fromhex(getattr(path, "sha256_in_prefix", "")) or None,
            "size_in_bytes": getattr(path, "size_in_bytes", None),
        }
        if file_mode := str(getattr(path, "file_mode", "")):
            kwargs["file_mode"] = rattler.FileMode(file_mode)
        path_entries.append(rattler.PrefixPathsEntry(**kwargs))
    paths_data.paths = path_entries
    return rattler.PrefixRecord(
        repodata_record=repodata_record,
        paths_data=paths_data,
        # missing 'link' argument
        package_tarball_full_path=record.package_tarball_full_path,
        extracted_package_dir=record.extracted_package_dir,
        requested_spec=record.get("requested_spec"),
        files=record.files,
    )


def empty_repodata_dict(subdir: str, **info_kwargs) -> dict[str, Any]:
    return {
        "info": {
            "subdir": subdir,
            **info_kwargs,
        },
        "packages": {},
        "packages.conda": {},
    }


def maybe_ignore_current_repodata(repodata_fn) -> str:
    is_repodata_fn_set = False
    for config in context.collect_all().values():
        for key, value in config.items():
            if key == "repodata_fns" and value:
                is_repodata_fn_set = True
                break
    if repodata_fn == "current_repodata.json" and not is_repodata_fn_set:
        log.debug(
            "Ignoring repodata_fn='current_repodata.json', defaulting to %s",
            REPODATA_FN,
        )
        return REPODATA_FN
    return repodata_fn


def notify_conda_outdated(
    prefix: PathType | None = None,
    index: RattlerIndexHelper | None = None,
    final_state: Iterable[PackageRecord] | None = None,
) -> None:
    """
    We are overriding the base class implementation, which gets called in
    Solver.solve_for_diff() once 'link_precs' is available. However, we
    are going to call it before (in .solve_final_state(), right after the solve).
    That way we can reuse the IndexHelper and SolverOutputState instances we have
    around, which contains the channel and env information we need, before losing them.
    """
    if prefix is None and index is None and final_state is None:
        # The parent class 'Solver.solve_for_diff()' method will call this method again
        # with only 'link_precs' as the argument, because that's the original method signature.
        # We have added two optional kwargs (index and final_state) so we can call this method
        # earlier, in .solve_final_state(), while we still have access to the index helper
        # (which allows us to query the available packages in the channels quickly, without
        # reloading the channels with conda) and the final_state (which gives the list of
        # packages to be installed). So, if both index and final_state are None, we return
        # because that means that the method is being called from .solve_for_diff() and at
        # that point we will have already called it from .solve_for_state().
        return
    if not context.notify_outdated_conda or context.quiet:
        # This check can be silenced with a specific option in the context or in quiet mode
        return

    # manually check base prefix since `PrefixData(...).get("conda", None) is expensive
    # once prefix data is lazy this might be a different situation
    current_conda_prefix_rec = None
    conda_meta_prefix_directory = os.path.join(context.conda_prefix, "conda-meta")
    with suppress(OSError, ValueError):
        if os.path.lexists(conda_meta_prefix_directory):
            for entry in os.scandir(conda_meta_prefix_directory):
                if (
                    entry.is_file()
                    and entry.name.endswith(".json")
                    and entry.name.rsplit("-", 2)[0] == "conda"
                ):
                    with open(entry.path) as f:
                        current_conda_prefix_rec = PrefixRecord(**json.loads(f.read()))
                    break
    if not current_conda_prefix_rec:
        # We are checking whether conda can be found in the environment conda is
        # running from. Unless something is really wrong, this should never happen.
        return

    channel_name = current_conda_prefix_rec.channel.canonical_name
    if channel_name in (UNKNOWN_CHANNEL, "@", "<develop>", "pypi"):
        channel_name = "defaults"

    # only check the loaded index if it contains the channel conda should come from
    # otherwise ignore
    index_channels = {getattr(chn, "canonical_name", chn) for chn in index.channels}
    if channel_name not in index_channels:
        return

    # we only want to check if a newer conda is available in the channel we installed it from
    conda_newer_str = f"{channel_name}::conda>{_conda_version}"
    conda_newer_spec = MatchSpec(conda_newer_str)

    # if target prefix is the same conda is running from
    # maybe the solution we are proposing already contains
    # an updated conda! in that case, we don't need to check further
    if paths_equal(prefix, context.conda_prefix):
        if any(conda_newer_spec.match(record) for record in final_state):
            return

    # check if the loaded index contains records that match a more recent conda version
    conda_newer_records = index.search(conda_newer_str)

    # print instructions to stderr if we found a newer conda
    if conda_newer_records:
        newest = max(conda_newer_records, key=lambda x: VersionOrder(x.version))
        print(
            dedent(
                f"""

                    ==> WARNING: A newer version of conda exists. <==
                        current version: {_conda_version}
                        latest version: {newest.version}

                    Please update conda by running

                        $ conda update -n base -c {channel_name} conda

                    """
            ),
            file=sys.stderr,
        )


def fix_version_field_for_conda_build(spec: MatchSpec) -> MatchSpec:
    """Fix taken from mambabuild"""
    if spec.version:
        only_dot_or_digit_re = re.compile(r"^[\d\.]+$")
        version_str = str(spec.version)
        if re.match(only_dot_or_digit_re, version_str):
            spec_fields = spec.conda_build_form().split()
            if version_str.count(".") <= 1:
                spec_fields[1] = version_str + ".*"
            else:
                spec_fields[1] = version_str + "*"
            return MatchSpec(" ".join(spec_fields))
    return spec
