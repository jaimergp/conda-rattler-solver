import rattler
from conda.models.records import PackageRecord

def rattler_record_to_conda_record(record: rattler.PackageRecord) -> PackageRecord:
    if timestamp := record.timestamp:
        timestamp = int(timestamp.timestamp() * 1000)
    else:
        timestamp = 0
    return PackageRecord(
        name=record.name.source,
        version=str(record.version),
        build=record.build,
        build_number=record.build_number,
        channel=record.channel,
        subdir=record.subdir,
        fn=record.file_name,
        md5=record.md5.lower() if record.md5 else None,
        legacy_bz2_md5=record.legacy_bz2_md5.lower() if record.legacy_bz2_md5 else None,
        legacy_bz2_size=record.legacy_bz2_size,
        url=record.url,
        sha256=record.sha256.lower() if record.sha256 else None,
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