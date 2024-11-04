from __future__ import annotations

import rattler
from conda.models.records import PackageRecord


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
