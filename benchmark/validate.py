#!/usr/bin/env python3
"""Validate benchmark record integrity independently of numerical success."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics
import tomllib


HEADERS = {
    "catalog": "id group scenario estimator variant phase suite inputs measure target".split(),
    "metrics": ("id group phase rep status samples ok_samples median_ns mean_ns sd_ns "
                "min_ns max_ns p95_ns gc_share median_bytes mean_bytes min_bytes "
                "max_bytes measure target worst_value median_value data_hash").split(),
    "samples": "id rep sample time_ns gc_ns bytes value status".split(),
    "errors": "id rep status exception stack".split(),
}
# Ordered by severity; a case takes the most severe status among its observations.
STATUSES = ("ok", "target_missed", "uncertifiable", "timeout", "error")
MEASURES = {"relative_gap", "kkt_residual"}
RECORD = "record"
# The sources of the loaded packages: ProximalBase and ProximalSolvers are
# identified by commit, tree state and version, which must not change within a
# series and, for a comparison, must agree between the series. The package
# hash identifies the measured revision: constant within a series, free between series.
SOURCES = ("proximalbase", "proximalsolvers")
SOURCE_FIELDS = tuple(f"{name}_{field}" for name in SOURCES for field in ("sha", "dirty", "version"))
SERIES_SETTINGS = (
    "suite", "groups", "harness_hash", "julia_threads", "blas_threads", "blas",
    "cpu_model", "samples_target", "seconds_limit", "case_seconds", "profile_seconds",
    "seed_scheme", "dependencies", "package_hash",
) + SOURCE_FIELDS
MATCH_SETTINGS = tuple(key for key in SERIES_SETTINGS if key != "package_hash")
START_KEYS = (
    "julia", "rep", "suite", "groups", "harness_hash", "package_path", "package_sha",
    "package_dirty", "package_version", "package_hash", "expected_package_root", "julia_threads",
    "blas_threads", "blas", "cpu", "cpu_model", "samples_target", "seconds_limit",
    "case_seconds", "profile_seconds", "seed_scheme", "job_id", "dependencies",
) + tuple(f"{name}_{field}" for name in SOURCES for field in ("path", "sha", "dirty", "version"))
FINAL_KEYS = ("cases", "ok_cases", "nonok_cases", "elapsed_s", "maxrss_bytes", "finished")
SUFFIXES = tuple(f"-{kind}.tsv" for kind in HEADERS) + (
    "-metadata-start.toml", "-metadata.toml", "-profiles.txt",
)
PREFIX = re.compile(r"julia-(.+)-rep([0-9]{2})")
TIME_FIELDS = ("median_ns", "mean_ns", "sd_ns", "min_ns", "max_ns", "p95_ns",
               "gc_share", "median_bytes", "mean_bytes", "min_bytes", "max_bytes")


def process_prefixes(root):
    prefixes = set()
    for path in Path(root).iterdir():
        if path.is_file():
            for suffix in SUFFIXES:
                if path.name.endswith(suffix):
                    prefixes.add(path.name.removesuffix(suffix))
                    break
    return prefixes


def discover_versions(root):
    if not Path(root).is_dir():
        return []
    return sorted({match[1] for prefix in process_prefixes(root)
                   if (match := PREFIX.fullmatch(prefix))})


def records(path, kind):
    # The runner writes fields verbatim, with tabs and line breaks replaced by spaces.
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != HEADERS[kind]:
            raise ValueError(f"wrong {kind} header in {Path(path).name}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"wrong field count in {Path(path).name}")
    return rows


def metadata(path):
    try:
        with Path(path).open("rb") as stream:
            return tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid metadata in {Path(path).name}: {error}") from error


def number(value):
    return float(value) if value != "" else math.nan


def same_number(left, right):
    return left == right or (math.isnan(left) and math.isnan(right))


def target_value(value, case=""):
    """Return a positive finite threshold, or `record` for accuracy without one."""
    if value == RECORD:
        return RECORD
    try:
        threshold = float(value)
    except ValueError:
        threshold = math.nan
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"invalid target in {case}")
    return threshold


def catalog_signature(rows):
    return {row["id"]: tuple(target_value(row[key], row["id"]) if key == "target" else row[key]
                             for key in HEADERS["catalog"]) for row in rows}


def dependency_differences(before, after):
    """List (package, field, before, after) for every differing version or tree hash."""
    differences = []
    for package in sorted(before.keys() | after.keys()):
        old, new = before.get(package, {}), after.get(package, {})
        for field in sorted(old.keys() | new.keys()):
            if old.get(field) != new.get(field):
                differences.append((package, field, old.get(field), new.get(field)))
    return differences


def fresh_output(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a fresh output directory to avoid stale reports")
    output.mkdir(parents=True, exist_ok=True)
    return output


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"invalid integer {label}: {value!r}")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"invalid {label}: {value!r}")
    return parsed


def check_dependencies(dependencies):
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError("malformed dependencies: expected a nonempty table")
    for package, entry in dependencies.items():
        if not isinstance(entry, dict) or not {"version", "tree_hash"} <= entry.keys():
            raise ValueError(f"malformed dependencies: {package} needs version and tree_hash")
        version, tree = entry["version"], entry["tree_hash"]
        # A package tracked by path has no tree hash in the manifest.
        if not isinstance(version, str) or not version or not isinstance(tree, str) or (
            tree and not re.fullmatch(r"[0-9a-fA-F]{40}", tree)
        ):
            raise ValueError(f"malformed dependencies: {package} version or tree_hash")


def check_metadata(meta, version, rep, final=True):
    missing = set(START_KEYS + (FINAL_KEYS if final else ())) - meta.keys()
    if missing:
        raise ValueError(f"missing metadata keys: {sorted(missing)}")
    if meta["julia"] != version or integer(meta["rep"], "rep", 1) != rep:
        raise ValueError("metadata version/repetition disagrees with filename")
    if final and meta["finished"] is not True:
        raise ValueError("process is not finished")
    if meta["suite"] not in ("quick", "extended"):
        raise ValueError("unknown suite")
    for key in ("julia_threads", "blas_threads"):
        if integer(meta[key], key, 1) != 1:
            raise ValueError("expected one Julia and one BLAS thread")
    for key in ("harness_hash", "package_hash", "package_sha",
                *(f"{name}_sha" for name in SOURCES)):
        if not isinstance(meta[key], str) or not re.fullmatch(r"[0-9a-fA-F]{40}", meta[key]):
            raise ValueError(f"malformed provenance hash: {key}")
    for key in ("package", *SOURCES):
        if not isinstance(meta[f"{key}_dirty"], bool):
            raise ValueError(f"{key}_dirty must be boolean")
        for field in ("path", "version"):
            if not isinstance(meta[f"{key}_{field}"], str) or not meta[f"{key}_{field}"]:
                raise ValueError(f"malformed provenance field: {key}_{field}")
    if meta["expected_package_root"] and (
        Path(meta["package_path"]).resolve() != Path(meta["expected_package_root"]).resolve()
    ):
        raise ValueError("loaded package root differs from expected_package_root")
    check_dependencies(meta["dependencies"])
    integer(meta["samples_target"], "samples_target", 1)
    for key in ("seconds_limit", "case_seconds", "profile_seconds"):
        value = float(meta[key])
        if not math.isfinite(value) or value < 0 or (key == "case_seconds" and value == 0):
            raise ValueError(f"invalid {key}")
    if final:
        for key in ("cases", "ok_cases", "nonok_cases", "maxrss_bytes"):
            integer(meta[key], key)
        if not math.isfinite(float(meta["elapsed_s"])) or meta["elapsed_s"] < 0:
            raise ValueError("invalid elapsed_s")


def check_catalog(rows, meta):
    catalog = {row["id"]: row for row in rows}
    if not catalog or len(catalog) != len(rows) or len(catalog) != meta["cases"]:
        raise ValueError("catalog duplicates, empty catalog or case-count mismatch")
    for row in rows:
        if row["id"] != "/".join(row[key] for key in ("group", "estimator", "variant")):
            raise ValueError(f"catalog id disagrees with its fields: {row['id']}")
        if not row["scenario"]:
            raise ValueError(f"missing scenario in {row['id']}")
        if row["phase"] != "call":
            raise ValueError(f"unknown phase in {row['id']}")
        if row["suite"] not in ("quick", "extended") or (
            meta["suite"] == "quick" and row["suite"] != "quick"
        ):
            raise ValueError(f"invalid suite in {row['id']}")
        if row["inputs"] not in ("fixed", "draw") or row["measure"] not in MEASURES:
            raise ValueError(f"invalid inputs/measure in {row['id']}")
        target_value(row["target"], row["id"])
    return catalog


def sample_statistics(rows):
    if not rows:
        return dict.fromkeys(TIME_FIELDS, math.nan)
    times = sorted(int(row["time_ns"]) for row in rows)
    allocations = [int(row["bytes"]) for row in rows]
    position = (len(times) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    return dict(
        median_ns=statistics.median(times), mean_ns=statistics.mean(times),
        sd_ns=statistics.pstdev(times),
        min_ns=min(times), max_ns=max(times),
        p95_ns=times[lower] + (times[upper] - times[lower]) * (position - lower),
        gc_share=sum(int(row["gc_ns"]) for row in rows) / sum(times) if sum(times) else 0.0,
        median_bytes=statistics.median(allocations), mean_bytes=statistics.mean(allocations),
        min_bytes=min(allocations), max_bytes=max(allocations),
    )


def check_process(root, prefix, rep, meta):
    tables = {kind: records(root / f"{prefix}-{kind}.tsv", kind) for kind in HEADERS}
    catalog = check_catalog(tables["catalog"], meta)
    metrics = tables["metrics"]
    if len(metrics) != len(catalog) or {row["id"] for row in metrics} != catalog.keys():
        raise ValueError("metrics do not cover the catalog exactly once")
    if any(row["status"] not in STATUSES for row in metrics):
        raise ValueError("unknown metric status")
    samples = defaultdict(dict)
    for row in tables["samples"]:
        case = row["id"]
        if case not in catalog or integer(row["rep"], "sample rep", 1) != rep:
            raise ValueError("foreign sample id/repetition")
        index = integer(row["sample"], "sample index", 1)
        if index in samples[case]:
            raise ValueError(f"duplicate sample in {case}")
        if row["status"] not in STATUSES:
            raise ValueError(f"unknown sample status in {case}")
        for key in ("time_ns", "gc_ns", "bytes"):
            if row[key] != "" or row["status"] != "error":
                integer(row[key], key)
        if row["gc_ns"] != "" and row["time_ns"] != "" and int(row["gc_ns"]) > int(row["time_ns"]):
            raise ValueError(f"GC time exceeds elapsed time in {case}")
        value = number(row["value"])
        if row["status"] == "ok" and (not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid ok certificate value in {case}")
        threshold = target_value(catalog[case]["target"], case)
        if row["status"] == "ok" and threshold != RECORD and value > threshold:
            raise ValueError(f"ok sample exceeds target in {case}")
        samples[case][index] = row
    errors = {row["id"]: row for row in tables["errors"]}
    nonok = {row["id"] for row in metrics if row["status"] != "ok"}
    if len(errors) != len(tables["errors"]) or errors.keys() != nonok:
        raise ValueError("errors do not cover non-ok cases exactly once")
    for row in metrics:
        case = row["id"]
        if integer(row["rep"], "metric rep", 1) != rep:
            raise ValueError(f"wrong metric repetition in {case}")
        for key in ("group", "phase", "measure"):
            if row[key] != catalog[case][key]:
                raise ValueError(f"metric {key} differs from catalog in {case}")
        if target_value(row["target"], case) != target_value(catalog[case]["target"], case):
            raise ValueError(f"metric target differs from catalog in {case}")
        count = integer(row["samples"], "samples")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", row["data_hash"]) and not (
            row["data_hash"] == "" and row["status"] == "error" and count == 0
        ):
            raise ValueError(f"malformed data_hash in {case}")
        if set(samples[case]) != set(range(1, count + 1)):
            raise ValueError(f"sample coverage mismatch in {case}")
        if row["status"] == "ok" and count < 3:
            raise ValueError(f"fewer than three samples in ok case {case}")
        good = [sample for sample in samples[case].values() if sample["status"] == "ok"]
        if integer(row["ok_samples"], "ok_samples") != len(good):
            raise ValueError(f"ok_samples disagrees with samples in {case}")
        if row["status"] == "ok" and (not good or len(good) != count):
            raise ValueError(f"ok case has missing or non-ok samples in {case}")
        # Validation can fail before sampling, so it can raise the case's severity.
        if any(STATUSES.index(sample["status"]) > STATUSES.index(row["status"])
               for sample in samples[case].values()):
            raise ValueError(f"case status understates sample failure in {case}")
        # Only ok samples are timing-eligible, whatever the case status.
        for key, observed in sample_statistics(good).items():
            reported = number(row[key])
            tolerance = 0.51 if key.endswith(("_ns", "_bytes")) else 1e-12
            if not (same_number(reported, observed) or math.isclose(
                reported, observed, rel_tol=1e-12, abs_tol=tolerance
            )):
                raise ValueError(f"{key} disagrees with ok samples in {case}")
        values = [number(sample["value"]) for sample in samples[case].values()]
        values = [value for value in values if math.isfinite(value)]
        accuracy = dict(worst_value=max(values) if values else math.nan,
                        median_value=statistics.median(values) if values else math.nan)
        for key, observed in accuracy.items():
            reported = number(row[key])
            if not (same_number(reported, observed) or math.isclose(
                reported, observed, rel_tol=1e-12, abs_tol=1e-15
            )):
                raise ValueError(f"{key} disagrees with samples in {case}")
        if case in errors:
            error = errors[case]
            if integer(error["rep"], "error rep", 1) != rep or error["status"] != row["status"]:
                raise ValueError(f"error status/repetition differs from metrics in {case}")
    if meta["ok_cases"] != len(metrics) - len(nonok) or meta["nonok_cases"] != len(nonok):
        raise ValueError("metadata status counts disagree with metrics")
    return tables


def validate(root, expected=1, versions=None):
    root = Path(root)
    versions = discover_versions(root) if versions is None else sorted(set(versions))
    issues, configurations, catalogs = [], {}, {}
    counts = dict(processes=0, measurements=0, samples=0, ok_samples=0, nonok_cases=0)
    statuses, nonok_cases = Counter(), []
    case_statuses = defaultdict(Counter)
    if not isinstance(expected, int) or isinstance(expected, bool) or not 1 <= expected <= 10:
        issues.append("expected runs must be between 1 and 10")
        expected = 0
    if not root.is_dir():
        issues.append("results directory does not exist")
    found = process_prefixes(root) if root.is_dir() else set()
    wanted = {f"julia-{version}-rep{rep:02d}"
              for version in versions for rep in range(1, expected + 1)}
    if not versions:
        issues.append("no process records found")
    if wanted != found:
        issues.append(f"process coverage: missing={sorted(wanted - found)}, extra={sorted(found - wanted)}")
    for prefix in sorted(found):
        match = PREFIX.fullmatch(prefix)
        if not match:
            issues.append(f"invalid process filename prefix: {prefix}")
            continue
        version, rep = match[1], int(match[2])
        try:
            final = root / f"{prefix}-metadata.toml"
            start = root / f"{prefix}-metadata-start.toml"
            if start.exists() and not final.exists():
                raise ValueError("interrupted process: start metadata without final metadata")
            meta = metadata(final)
            check_metadata(meta, version, rep)
            initial = metadata(start)
            check_metadata(initial, version, rep, final=False)
            if any(initial[key] != meta[key] for key in START_KEYS):
                raise ValueError("start and final metadata disagree")
            if version in configurations:
                changed = dependency_differences(configurations[version]["dependencies"],
                                                 meta["dependencies"])
                if changed:
                    raise ValueError("dependency versions or tree hashes changed within a "
                                     f"Julia version (package, field, before, after): {changed}")
                if any(configurations[version][key] != meta[key] for key in SERIES_SETTINGS):
                    raise ValueError("series settings changed within a Julia version")
            configurations[version] = {key: meta[key] for key in SERIES_SETTINGS}
            tables = check_process(root, prefix, rep, meta)
            signature = catalog_signature(tables["catalog"])
            if version in catalogs and signature != catalogs[version]:
                raise ValueError("catalog changed between repetitions")
            catalogs[version] = signature
            counts["processes"] += 1
            counts["measurements"] += len(tables["metrics"])
            counts["samples"] += len(tables["samples"])
            counts["ok_samples"] += sum(row["status"] == "ok" for row in tables["samples"])
            for row in tables["metrics"]:
                statuses[row["status"]] += 1
                case_statuses[(version, row["id"])][row["status"]] += 1
                if row["status"] != "ok":
                    nonok_cases.append(dict(julia=version, id=row["id"], rep=rep,
                                            status=row["status"], worst_value=row["worst_value"]))
        except (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error) as error:
            issues.append(f"{prefix}: {error}")
    counts["nonok_cases"] = len(nonok_cases)
    return dict(valid=not issues, issues=issues, counts=counts,
                status_counts={status: statuses[status] for status in STATUSES},
                nonok_cases=nonok_cases, expected_runs=expected, versions=versions,
                case_status_counts=[dict(julia=version, id=case,
                                         status_counts={status: values[status] for status in STATUSES})
                                    for (version, case), values in sorted(case_statuses.items())],
                cases_per_version={version: len(catalog) for version, catalog in catalogs.items()})


def load_series(root):
    rows, configs, catalogs = {}, {}, {}
    for path in sorted(Path(root).glob("*-metadata.toml")):
        prefix = path.name.removesuffix("-metadata.toml")
        meta = metadata(path)
        version = meta["julia"]
        configs[version] = meta
        catalogs[version] = catalog_signature(records(path.with_name(prefix + "-catalog.tsv"), "catalog"))
        for row in records(path.with_name(prefix + "-metrics.tsv"), "metrics"):
            rows[(version, row["id"], int(row["rep"]))] = row
    return rows, configs, catalogs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("out_json")
    parser.add_argument("--expected-runs", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--versions", nargs="+")
    args = parser.parse_args()
    result = validate(args.root, args.expected_runs, args.versions)
    try:
        Path(args.out_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        parser.exit(3, f"Cannot write validation JSON: {error}\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 2)
