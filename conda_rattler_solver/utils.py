from __future__ import annotations

from typing import TYPE_CHECKING

import rattler
from conda.models.records import PackageRecord

if TYPE_CHECKING:
    from conda.models.records import PrefixRecord


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

    return PackageRecord(
        name=record.name.source,
        version=str(record.version),
        build=record.build,
        build_number=record.build_number,
        channel=record.channel,
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
            "sha256_in_prefix": bytes.fromhex(getattr(path, "sha256_in_prefix", ""))
            or None,
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
