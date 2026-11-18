#!/usr/bin/env python3
"""Run analytic controls and synthetic record-integrity checks without Julia."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

# Leave no bytecode cache beside the scripts.
sys.dont_write_bytecode = True

from compare import FIELDS, compare
from summarize import mean_interval, summarize
from validate import HEADERS, SERIES_SETTINGS, STATUSES, records, validate


VERSION = "1.10.0"
DEFAULT_CASE = "lasso/lasso/control"
FIXTURE_HEADERS = {
    "catalog": "id group scenario estimator variant phase suite inputs measure target".split(),
    "metrics": ("id group phase rep status samples ok_samples median_ns mean_ns sd_ns "
                "min_ns max_ns p95_ns gc_share median_bytes mean_bytes min_bytes "
                "max_bytes measure target worst_value median_value data_hash").split(),
    "samples": "id rep sample time_ns gc_ns bytes value status".split(),
    "errors": "id rep status exception stack".split(),
}
DEPENDENCIES = {
    "StatsAPI": dict(version="1.8.0", tree_hash="1" * 40),
    # Packages tracked by path have no tree hash.
    "ProximalBase": dict(version="0.6.0", tree_hash=""),
    "ProximalSolvers": dict(version="0.3.2", tree_hash=""),
}


def write_rows(path, kind, rows):
    # Like the runner: fields verbatim and unquoted; a tab or line break is an error.
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, FIXTURE_HEADERS[kind], delimiter="\t", lineterminator="\n",
                                quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path, delimiter="\t"):
    # Runner TSVs are unquoted; the analysis CSVs use standard quoting.
    quoting = csv.QUOTE_NONE if delimiter == "\t" else csv.QUOTE_MINIMAL
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter, quoting=quoting))


def write_meta(path, values):
    def literal(value):
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, dict):
            return "{" + ", ".join(f"{json.dumps(key)} = {literal(item)}"
                                   for key, item in value.items()) + "}"
        return json.dumps(value) if isinstance(value, str) else str(value)
    lines = [f"{key} = {literal(value)}" for key, value in values.items()
             if not isinstance(value, dict)]
    for key, table in values.items():
        if isinstance(table, dict):
            lines += ["", f"[{key}]"] + [f"{json.dumps(name)} = {literal(entry)}"
                                         for name, entry in table.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select(value, rep, case):
    return value(rep, case) if callable(value) else value


def fixture(folder, factor=4, runs=3, allocated=0, cases=(DEFAULT_CASE,), status="ok",
            accuracy=1e-9, version=VERSION, measure="relative_gap", target=1e-6):
    folder.mkdir(exist_ok=True)
    for rep in range(1, runs + 1):
        prefix = folder / f"julia-{version}-rep{rep:02d}"
        catalog, metrics, samples, errors = [], [], [], []
        for case in cases:
            group, estimator, variant = case.split("/")
            catalog.append(dict(id=case, group=group, scenario=f"control-{variant}",
                                estimator=estimator, variant=variant, phase="call",
                                suite="quick", inputs="draw", measure=measure, target=target))
            outcome = select(status, rep, case)
            value = select(accuracy, rep, case) * rep
            if outcome == "target_missed":
                value = target * 2
            elif outcome in ("uncertifiable", "error"):
                value = math.nan
            base = select(factor, rep, case) * 100 * rep
            allocation = select(allocated, rep, case)
            count = 0 if outcome == "error" else 5
            for index in range(1, count + 1):
                samples.append(dict(id=case, rep=rep, sample=index,
                                    time_ns=int(base * (index + 1) / 4), gc_ns=0,
                                    bytes=allocation, value=value, status=outcome))
            good = outcome == "ok"
            # The five times are base times (1/2, 3/4, 1, 5/4, 3/2).
            metrics.append(dict(
                id=case, group=group, phase="call", rep=rep, status=outcome,
                samples=count, ok_samples=5 if good else 0,
                median_ns=base if good else math.nan, mean_ns=base if good else math.nan,
                sd_ns=base * math.sqrt(1 / 8) if good else math.nan,
                min_ns=base / 2 if good else math.nan, max_ns=base * 1.5 if good else math.nan,
                p95_ns=base * 1.45 if good else math.nan, gc_share=0 if good else math.nan,
                median_bytes=allocation if good else math.nan,
                mean_bytes=allocation if good else math.nan,
                min_bytes=allocation if good else math.nan, max_bytes=allocation if good else math.nan,
                measure=measure, target=target, worst_value=value, median_value=value,
                data_hash=hashlib.sha256(f"{case}:{rep}".encode()).hexdigest(),
            ))
            if not good:
                errors.append(dict(id=case, rep=rep, status=outcome,
                                   exception="control" if outcome == "error" else "", stack=""))
        for kind, rows in (("catalog", catalog), ("metrics", metrics),
                           ("samples", samples), ("errors", errors)):
            write_rows(Path(str(prefix) + f"-{kind}.tsv"), kind, rows)
        meta = dict(
            julia=version, rep=rep, suite="quick", groups="lasso", harness_hash="a" * 40,
            package_path="/synthetic/package", package_sha="b" * 40, package_dirty=False,
            package_version="0.2.0", package_hash="c" * 40, expected_package_root="/synthetic/package",
            proximalbase_path="/synthetic/ProximalBase", proximalbase_sha="d" * 40,
            proximalbase_dirty=False, proximalbase_version="0.6.0",
            proximalsolvers_path="/synthetic/ProximalSolvers", proximalsolvers_sha="e" * 40,
            proximalsolvers_dirty=False, proximalsolvers_version="0.3.2",
            julia_threads=1, blas_threads=1, blas="control", cpu="control", cpu_model="control",
            samples_target=5, seconds_limit=1.0, case_seconds=60.0, profile_seconds=0.0,
            seed_scheme="control", job_id="none",
            dependencies={name: dict(entry) for name, entry in DEPENDENCIES.items()},
        )
        write_meta(Path(str(prefix) + "-metadata-start.toml"), meta)
        meta.update(cases=len(catalog), ok_cases=sum(row["status"] == "ok" for row in metrics),
                    nonok_cases=len(errors), elapsed_s=1.0, maxrss_bytes=0, finished=True)
        write_meta(Path(str(prefix) + "-metadata.toml"), meta)


def edit_rows(folder, kind, change, rep=None):
    pattern = f"*-rep{rep:02d}-{kind}.tsv" if rep else f"*-{kind}.tsv"
    for path in sorted(folder.glob(pattern)):
        rows = read_rows(path)
        change(rows)
        write_rows(path, kind, rows)


def edit_metadata(folder, change, rep=None, pattern="*-metadata*.toml"):
    for path in folder.glob(pattern):
        if rep is not None and f"-rep{rep:02d}-" not in path.name:
            continue
        with path.open("rb") as stream:
            meta = tomllib.load(stream)
        change(meta)
        write_meta(path, meta)


def edit_meta(folder, key, value, rep=None):
    edit_metadata(folder, lambda meta: meta.update({key: value}), rep)


def edit_dependency(folder, package, field, value, rep=None):
    edit_metadata(folder, lambda meta: meta["dependencies"].setdefault(package, {}).update(
        {field: value}), rep)


def expect_error(action, message):
    try:
        action()
    except ValueError as error:
        assert message in str(error), str(error)
        return str(error)
    raise AssertionError(f"Expected ValueError containing {message!r}")


def invalid_mutation(folder, change, message, expected=3):
    saved = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
    try:
        change()
        result = validate(folder, expected)
        assert result["valid"] is False, message
        assert any(message in issue for issue in result["issues"]), result["issues"]
    finally:
        for path in folder.iterdir():
            if path.is_file() and path not in saved:
                path.unlink()
        for path, contents in saved.items():
            path.write_bytes(contents)
    assert validate(folder, expected)["valid"]


def analytic_controls(root):
    assert HEADERS == FIXTURE_HEADERS
    for runs, critical in ((1, None), (3, 4.303), (5, 2.776), (10, 2.262)):
        before, after = root / f"before{runs}", root / f"after{runs}"
        fixture(before, runs=runs, allocated=64)
        fixture(after, factor=2, runs=runs, allocated=16, accuracy=2e-9)
        checked = validate(before, runs)
        assert checked["valid"], checked["issues"]
        assert checked["counts"] == dict(processes=runs, measurements=runs, samples=5 * runs,
                                           ok_samples=5 * runs, nonok_cases=0)
        assert checked["status_counts"] == {status: runs if status == "ok" else 0 for status in STATUSES}
        assert checked["nonok_cases"] == []
        summary = root / f"summary{runs}"
        assert summarize(before, summary, runs)
        row = read_rows(summary / "summary.csv", ",")[0]
        average = 200 * (runs + 1)
        assert float(row["mean_ns"]) == average
        assert float(row["mean_bytes"]) == 64
        assert float(row["worst_value"]) == 1e-9 * runs
        assert int(row["status_ok"]) == runs and int(row["timing_runs"]) == runs
        if runs == 1:
            assert math.isnan(float(row["ci95_low_ns"]))
            assert math.isnan(float(row["ci95_high_ns"]))
        else:
            sd = 400 * math.sqrt(runs * (runs + 1) / 12)
            half = critical * sd / math.sqrt(runs)
            assert math.isclose(float(row["sd_between_runs_ns"]), sd)
            assert math.isclose(float(row["ci95_low_ns"]), max(0, average - half))
            assert math.isclose(float(row["ci95_high_ns"]), average + half)
        assert "HighDimensionalModels benchmark summary" in (summary / "SUMMARY.md").read_text()
        output = root / f"comparison{runs}"
        rows = compare(before, after, output, runs)
        assert len(rows) == 1
        row = rows[0]
        assert list(row) == FIELDS and row["phase"] == "call"
        assert math.isclose(row["paired_time_ratio"], 0.5)
        assert row["before_mean_bytes"] == 64 and row["after_mean_bytes"] == 16
        assert row["mean_bytes_change"] == -48
        assert row["before_max_bytes"] == 64 and row["after_max_bytes"] == 16
        assert row["max_bytes_change"] == -48
        assert math.isclose(row["before_worst_value"], runs * 1e-9)
        assert math.isclose(row["after_worst_value"], runs * 2e-9)
        assert math.isclose(row["worst_value_change"], runs * 1e-9)
        if runs == 1:
            assert math.isnan(row["ci95_low"]) and math.isnan(row["ci95_high"])
            assert row["timing_assessment"] == "single run; exploratory"
        else:
            assert math.isclose(row["ci95_low"], 0.5) and math.isclose(row["ci95_high"], 0.5)
            assert row["timing_assessment"] == "faster"
        # Vary the ratios so the logarithmic interval has a nonzero known width.
        varying = root / f"varying{runs}"
        fixture(varying, factor=lambda rep, case: 2 ** rep, runs=runs)
        row = compare(before, varying, root / f"varying-out{runs}", runs)[0]
        mu = ((runs + 1) / 2 - 2) * math.log(2)
        assert math.isclose(row["paired_time_ratio"], math.exp(mu))
        if runs > 1:
            half = critical * math.sqrt(runs * (runs + 1) / 12) * math.log(2) / math.sqrt(runs)
            assert math.isclose(row["ci95_low"], math.exp(mu - half))
            assert math.isclose(row["ci95_high"], math.exp(mu + half))
            assert row["timing_assessment"] == ("inconclusive" if mu - half <= 0 <= mu + half else "slower")
        report = (output / "REPORT.md").read_text()
        for sentence in ("HighDimensionalModels revision comparison",
                         "Single-run ratios are exploratory", "containing one is inconclusive",
                         "Excluded cases are not ties", "Maximum bytes before / after"):
            assert sentence in report
        assert "unexpected" not in report.lower()
    # The three-repetition interval contains one, so it is inconclusive.
    row = compare(root / "before3", root / "varying3", root / "varying-check", 3)[0]
    assert row["ci95_low"] < 1 < row["ci95_high"] and row["timing_assessment"] == "inconclusive"
    assert math.isnan(mean_interval([1.0]))
    record = root / "record-target"
    fixture(record, target="record")
    assert validate(record, 3)["valid"]
    assert len(compare(record, record, root / "record-out", 3)) == 1
    assert summarize(record, root / "record-summary", 3)
    assert read_rows(root / "record-summary" / "summary.csv", ",")[0]["target"] == "record"


def allocation_controls(root):
    grows, shrinks = "lasso/lasso/grows", "lasso/lasso/shrinks"
    cases = (grows, shrinks)
    before, after = root / "bytes-before", root / "bytes-after"
    # Repetition medians 0, 0, 96 separate the mean (32) from the median (0).
    fixture(before, cases=cases, allocated=lambda rep, case: (
        0 if case == grows else 96 * (rep == 3)))
    fixture(after, cases=cases, allocated=lambda rep, case: 16 * rep if case == grows else 0)
    output = root / "allocation-out"
    rows = {row["id"]: row for row in compare(before, after, output, 3)}
    assert set(rows) == set(cases)
    row = rows[grows]
    assert row["before_mean_bytes"] == 0 and row["after_mean_bytes"] == 32
    assert row["mean_bytes_change"] == 32
    assert row["before_max_bytes"] == 0 and row["after_max_bytes"] == 48
    assert row["max_bytes_change"] == 48
    row = rows[shrinks]
    assert row["before_mean_bytes"] == 32 and row["after_mean_bytes"] == 0
    assert row["mean_bytes_change"] == -32
    assert row["before_max_bytes"] == 96 and row["after_max_bytes"] == 0
    assert row["max_bytes_change"] == -96
    assert all(row["timing_assessment"] == "inconclusive" for row in rows.values())
    written = read_rows(output / "comparison.csv", ",")
    assert [float(row["max_bytes_change"]) for row in written] == [48, -96]


def sample_controls(root):
    folder = root / "asymmetric"
    fixture(folder)
    def skew_samples(rows):
        base = 400 * int(rows[0]["rep"])
        for index, row in enumerate(rows):
            row.update(time_ns=base if index < 4 else 100 * base,
                       gc_ns=base // 10 if index < 3 else 0, bytes=0 if index < 4 else 64)
    def skew_metrics(rows):
        base = 400 * int(rows[0]["rep"])
        rows[0].update(median_ns=base, mean_ns=20.8 * base, sd_ns=39.6 * base,
                       min_ns=base, max_ns=100 * base, p95_ns=80.2 * base,
                       gc_share=0.3 / 104, median_bytes=0, mean_bytes=12.8, min_bytes=0, max_bytes=64)
    edit_rows(folder, "samples", skew_samples)
    edit_rows(folder, "metrics", skew_metrics)
    assert validate(folder, 3)["valid"]
    output = root / "asymmetric-summary"
    assert summarize(folder, output, 3)
    row = read_rows(output / "summary.csv", ",")[0]
    assert float(row["mean_ns"]) == 800 and float(row["mean_bytes"]) == 0
    assert float(row["max_bytes"]) == 64
    row = compare(folder, folder, root / "asymmetric-out", 3)[0]
    assert row["paired_time_ratio"] == 1 and row["timing_assessment"] == "inconclusive"
    assert row["before_mean_bytes"] == 0 and row["before_max_bytes"] == 64
    assert row["mean_bytes_change"] == 0 and row["max_bytes_change"] == 0
    # Integer-nanosecond rounding does not invalidate recomputable statistics.
    edit_rows(folder, "metrics", lambda rows: rows[0].update(
        sd_ns=round(float(rows[0]["sd_ns"])), p95_ns=round(float(rows[0]["p95_ns"]))))
    assert validate(folder, 3)["valid"]
    partial = root / "partial-samples"
    fixture(partial, status="target_missed")
    def partial_samples(rows):
        for row in rows[:4]:
            row.update(status="ok", value=1e-9)
    def partial_metrics(rows):
        base = 400 * int(rows[0]["rep"])
        rows[0].update(ok_samples=4, median_ns=0.875 * base, mean_ns=0.875 * base,
                       sd_ns=base * math.sqrt(5 / 64), min_ns=0.5 * base,
                       max_ns=1.25 * base, p95_ns=1.2125 * base, gc_share=0,
                       median_bytes=0, mean_bytes=0, min_bytes=0, max_bytes=0, median_value=1e-9)
    edit_rows(partial, "samples", partial_samples)
    edit_rows(partial, "metrics", partial_metrics)
    checked = validate(partial, 3)
    assert checked["valid"], checked["issues"]
    assert checked["counts"]["ok_samples"] == 12 and checked["status_counts"]["target_missed"] == 3
    assert summarize(partial, root / "partial-summary", 3)
    row = read_rows(root / "partial-summary" / "summary.csv", ",")[0]
    assert float(row["mean_ns"]) == 700 and int(row["timing_runs"]) == 3
    assert int(row["status_target_missed"]) == 3
    assert compare(partial, partial, root / "partial-out", 3) == []
    thrown = root / "thrown-sample"
    fixture(thrown, status="error")
    for rep in range(1, 4):
        path = thrown / f"julia-{VERSION}-rep{rep:02d}-samples.tsv"
        write_rows(path, "samples", [dict(id=DEFAULT_CASE, rep=rep, sample=1, time_ns="",
                                         gc_ns="", bytes="", value="", status="error")])
    edit_rows(thrown, "metrics", lambda rows: rows[0].update(samples=1))
    checked = validate(thrown, 3)
    assert checked["valid"], checked["issues"]
    assert checked["counts"]["samples"] == 3 and checked["status_counts"]["error"] == 3
    assert compare(thrown, thrown, root / "thrown-out", 3) == []
    construction = root / "construction-error"
    fixture(construction, status="error")
    edit_rows(construction, "metrics", lambda rows: rows[0].update(data_hash=""))
    assert validate(construction, 3)["valid"]
    assert compare(construction, construction, root / "construction-out", 3) == []


def failure_controls(root):
    good = "lasso/good/control"
    mixed = "lasso/mixed/control"
    cases = (good, mixed)
    before, after = root / "mixed-before", root / "mixed-after"
    fixture(before, cases=cases, status=lambda rep, case: (
        "target_missed" if case == mixed and rep == 2 else
        "uncertifiable" if case == mixed and rep == 3 else "ok"))
    fixture(after, cases=cases, status=lambda rep, case: (
        "uncertifiable" if case == mixed and rep == 1 else "ok"))
    checked = validate(before, 3)
    assert checked["valid"] and checked["counts"]["nonok_cases"] == 2
    assert checked["status_counts"]["ok"] == 4
    assert checked["status_counts"]["target_missed"] == 1
    assert checked["status_counts"]["uncertifiable"] == 1
    assert checked["case_status_counts"] == [
        dict(julia=VERSION, id=good, status_counts={status: 3 if status == "ok" else 0 for status in STATUSES}),
        dict(julia=VERSION, id=mixed, status_counts={status: 1 if status in (
            "ok", "target_missed", "uncertifiable") else 0 for status in STATUSES})]
    assert [(row["rep"], row["status"]) for row in checked["nonok_cases"]] == [
        (2, "target_missed"), (3, "uncertifiable")]
    out = root / "mixed-out"
    rows = compare(before, after, out, 3)
    assert [row["id"] for row in rows] == [good]
    coverage = json.loads((out / "coverage.json").read_text())
    assert coverage["excluded"] == [dict(julia=VERSION, id=mixed,
                                         before=["rep02: target_missed", "rep03: uncertifiable"],
                                         after=["rep01: uncertifiable"])]
    summary = root / "mixed-summary"
    assert summarize(before, summary, 3)
    row = next(row for row in read_rows(summary / "summary.csv", ",") if row["id"] == mixed)
    assert math.isnan(float(row["mean_ns"])) and int(row["timing_runs"]) == 1
    assert int(row["runs"]) == 3 and int(row["status_target_missed"]) == 1
    assert int(row["status_uncertifiable"]) == 1
    for status in STATUSES[1:]:
        failed_before, failed_after = root / f"failed-before-{status}", root / f"failed-after-{status}"
        fixture(failed_before, status=status)
        fixture(failed_after, status=status)
        edit_rows(failed_before, "metrics", lambda rows: rows[0].update(median_ns=""))
        checked = validate(failed_before, 3)
        assert checked["valid"], checked["issues"]
        assert checked["status_counts"][status] == 3 and checked["counts"]["ok_samples"] == 0
        output = root / f"failed-out-{status}"
        assert compare(failed_before, failed_after, output, 3) == []
        assert {path.name for path in output.iterdir()} == {"REPORT.md", "comparison.csv", "coverage.json"}
        assert (output / "comparison.csv").read_text().strip() == ",".join(FIELDS)
        excluded = json.loads((output / "coverage.json").read_text())["excluded"]
        reasons = [f"rep{rep:02d}: {status}" for rep in range(1, 4)]
        assert excluded == [dict(julia=VERSION, id=DEFAULT_CASE, before=reasons, after=reasons)]
        assert "No eligible cases" in (output / "REPORT.md").read_text()
        assert summarize(failed_before, root / f"failed-summary-{status}", 3)


def target_missed_controls(root):
    # An under-iterated case misses its target in every sample, beside an ok case.
    tight, missed = "lasso/lasso/tight", "controls/lasso/underiterated"
    cases = (tight, missed)
    def status(rep, case):
        return "target_missed" if case == missed else "ok"
    before, after = root / "missed-before", root / "missed-after"
    fixture(before, cases=cases, status=status, allocated=64)
    fixture(after, cases=cases, status=status, factor=2, allocated=64)
    checked = validate(before, 3)
    assert checked["valid"], checked["issues"]
    assert checked["counts"]["samples"] == 30 and checked["counts"]["ok_samples"] == 15
    assert checked["status_counts"]["ok"] == 3 and checked["status_counts"]["target_missed"] == 3
    assert [(row["id"], row["rep"], row["status"]) for row in checked["nonok_cases"]] == [
        (missed, rep, "target_missed") for rep in range(1, 4)]
    summary = root / "missed-summary"
    assert summarize(before, summary, 3)
    rows = {row["id"]: row for row in read_rows(summary / "summary.csv", ",")}
    assert int(rows[missed]["timing_runs"]) == 0 and int(rows[missed]["status_target_missed"]) == 3
    assert all(math.isnan(float(rows[missed][key]))
               for key in ("mean_ns", "ci95_low_ns", "ci95_high_ns", "mean_bytes", "max_bytes"))
    assert float(rows[missed]["worst_value"]) == 2e-6
    assert float(rows[tight]["mean_ns"]) == 800 and int(rows[tight]["timing_runs"]) == 3
    output = root / "missed-out"
    rows = compare(before, after, output, 3)
    assert [row["id"] for row in rows] == [tight] and math.isclose(rows[0]["paired_time_ratio"], 0.5)
    reasons = [f"rep{rep:02d}: target_missed" for rep in range(1, 4)]
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["excluded"] == [dict(julia=VERSION, id=missed, before=reasons, after=reasons)]
    report = (output / "REPORT.md").read_text()
    assert "Included cases: 1. Excluded cases: 1." in report
    assert f"| {missed} | {VERSION} | {'; '.join(reasons)} |" in report
    # Timings of missed samples cannot enter the metrics, nor can those samples pass as ok.
    def time_missed(rows):
        for row in rows:
            if row["id"] == missed:
                row.update(median_ns=400 * int(row["rep"]))
    invalid_mutation(before, lambda: edit_rows(before, "metrics", time_missed),
                     "median_ns disagrees with ok samples")
    def promote(rows):
        for row in rows:
            if row["id"] == missed:
                row.update(status="ok")
    invalid_mutation(before, lambda: edit_rows(before, "samples", promote), "ok sample exceeds target")
    invalid_mutation(before, lambda: (edit_rows(before, "metrics", promote),
                                      edit_rows(before, "errors", list.clear),
                                      edit_meta(before, "ok_cases", 2),
                                      edit_meta(before, "nonok_cases", 0)),
                     "ok case has missing or non-ok samples")


def quoting_controls(root):
    # Exception text is unquoted: leading, doubled and unbalanced quotes and backslashes
    # are literal, and none of them may merge fields or rows.
    first, second = "lasso/lasso/quoted", "lasso/lasso/unbalanced"
    folder = root / "quoted-errors"
    fixture(folder, cases=(first, second), status="error")
    # Read as quoted, the first exception would run on over the tab into its stack.
    text = {first: ('"MethodError: no method matching ""f""(::Int) \\ end',
                    '[1] f(x) at "cases.jl":3 \\'),
            second: ('"control" failed: "unbalanced', "'single' \\\"quote")}
    def quote(rows):
        for row in rows:
            row.update(exception=text[row["id"]][0], stack=text[row["id"]][1])
    edit_rows(folder, "errors", quote)
    path = folder / f"julia-{VERSION}-rep01-errors.tsv"
    assert all(f"{exception}\t{stack}\n" in path.read_text() for exception, stack in text.values())
    checked = validate(folder, 3)
    assert checked["valid"], checked["issues"]
    assert checked["status_counts"]["error"] == 6
    for rep in range(1, 4):
        path = folder / f"julia-{VERSION}-rep{rep:02d}-errors.tsv"
        for rows in (records(path, "errors"), read_rows(path)):
            assert {row["id"]: (row["exception"], row["stack"]) for row in rows} == text
    assert compare(folder, folder, root / "quoted-out", 3) == []
    assert summarize(folder, root / "quoted-summary", 3)


def integrity_controls(root):
    folder = root / "integrity"
    fixture(folder)
    prefix = folder / f"julia-{VERSION}-rep01"
    final = Path(str(prefix) + "-metadata.toml")
    start = Path(str(prefix) + "-metadata-start.toml")
    invalid_mutation(folder, final.unlink, "interrupted process")
    invalid_mutation(folder, start.unlink, "No such file")
    for kind in HEADERS:
        path = Path(str(prefix) + f"-{kind}.tsv")
        invalid_mutation(folder, path.unlink, "No such file")
        invalid_mutation(folder, lambda path=path: path.write_text("wrong\theader\n"), "wrong")
    invalid_mutation(folder, lambda: final.write_text("not valid toml ["), "invalid metadata")
    invalid_mutation(folder, lambda: edit_meta(folder, "finished", False), "not finished")
    invalid_mutation(folder, lambda: edit_metadata(folder, lambda meta: meta.pop("finished", None), rep=1),
                     "missing metadata keys")
    invalid_mutation(folder, lambda: edit_meta(folder, "julia_threads", 2), "one Julia and one BLAS thread")
    invalid_mutation(folder, lambda: edit_meta(folder, "blas_threads", 2), "one Julia and one BLAS thread")
    for key in ("harness_hash", "package_hash", "package_sha", "proximalbase_sha", "proximalsolvers_sha"):
        invalid_mutation(folder, lambda key=key: edit_meta(folder, key, "bad"), "provenance hash")
    for key in ("package", "proximalbase", "proximalsolvers"):
        invalid_mutation(folder, lambda key=key: edit_meta(folder, f"{key}_dirty", "no"),
                         f"{key}_dirty must be boolean")
        for field in ("path", "version"):
            invalid_mutation(folder, lambda key=key, field=field: edit_meta(folder, f"{key}_{field}", ""),
                             f"malformed provenance field: {key}_{field}")
    invalid_mutation(folder, lambda: edit_metadata(folder, lambda meta: meta.pop("proximalsolvers_sha")),
                     "missing metadata keys")
    invalid_mutation(folder, lambda: edit_metadata(folder, lambda meta: meta.pop("dependencies")),
                     "missing metadata keys")
    invalid_mutation(folder, lambda: edit_meta(folder, "dependencies", {}), "malformed dependencies")
    invalid_mutation(folder, lambda: edit_metadata(folder, lambda meta: meta["dependencies"]["StatsAPI"].pop(
        "tree_hash")), "malformed dependencies")
    invalid_mutation(folder, lambda: edit_dependency(folder, "StatsAPI", "tree_hash", "bad"),
                     "malformed dependencies")
    invalid_mutation(folder, lambda: edit_dependency(folder, "StatsAPI", "version", ""),
                     "malformed dependencies")
    invalid_mutation(folder, lambda: edit_metadata(
        folder, lambda meta: meta["dependencies"]["StatsAPI"].update(version="1.9.1"),
        pattern="*-rep01-metadata-start.toml"), "start and final metadata disagree")
    # Dependency identity is part of the series: no version or tree hash may change.
    for field, value in (("version", "0.6.1"), ("tree_hash", "e" * 40)):
        message = "dependency versions or tree hashes changed within a Julia version"
        invalid_mutation(folder, lambda field=field, value=value: edit_dependency(
            folder, "ProximalBase", field, value, rep=2), message)
    invalid_mutation(folder, lambda: (edit_dependency(folder, "Extra", "version", "1.0.0", rep=2),
                                      edit_dependency(folder, "Extra", "tree_hash", "f" * 40, rep=2)),
                     "dependency versions or tree hashes changed within a Julia version")
    invalid_mutation(folder, lambda: edit_meta(folder, "package_hash", "e" * 40, rep=2),
                     "series settings changed")
    # The sources of ProximalBase and ProximalSolvers are constant within a series.
    for key, value in (("proximalbase_sha", "f" * 40), ("proximalbase_dirty", True),
                       ("proximalbase_version", "0.6.1"), ("proximalsolvers_sha", "f" * 40),
                       ("proximalsolvers_dirty", True), ("proximalsolvers_version", "0.3.3")):
        invalid_mutation(folder, lambda key=key, value=value: edit_meta(folder, key, value, rep=2),
                         "series settings changed")
    invalid_mutation(folder, lambda: edit_rows(folder, "metrics", lambda rows: rows.append(rows[0])),
                     "metrics do not cover")
    invalid_mutation(folder, lambda: edit_rows(folder, "metrics", lambda rows: rows[0].update(rep=9)),
                     "wrong metric repetition")
    invalid_mutation(folder, lambda: edit_rows(folder, "metrics", lambda rows: rows[0].update(samples=4)),
                     "sample coverage mismatch")
    invalid_mutation(folder, lambda: edit_rows(folder, "metrics", lambda rows: rows[0].update(ok_samples=4)),
                     "ok_samples disagrees")
    def keep_two(rows):
        del rows[2:]
    invalid_mutation(folder, lambda: (edit_rows(folder, "samples", keep_two),
                                      edit_rows(folder, "metrics", lambda rows: rows[0].update(samples=2))),
                     "fewer than three samples")
    for key in ("median_ns", "mean_ns", "sd_ns", "min_ns", "max_ns", "p95_ns", "gc_share",
                "median_bytes", "mean_bytes", "min_bytes", "max_bytes"):
        invalid_mutation(folder, lambda key=key: edit_rows(folder, "metrics", lambda rows: rows[0].update({key: 999})),
                         f"{key} disagrees")
    for key in ("worst_value", "median_value"):
        invalid_mutation(folder, lambda key=key: edit_rows(folder, "metrics", lambda rows: rows[0].update({key: 1})),
                         f"{key} disagrees")
    for kind in ("metrics", "samples"):
        invalid_mutation(folder, lambda kind=kind: edit_rows(folder, kind, lambda rows: rows[0].update(status="unknown")),
                         "unknown")
    invalid_mutation(folder, lambda: edit_rows(folder, "samples", lambda rows: rows.append(rows[0])), "duplicate sample")
    invalid_mutation(folder, lambda: edit_rows(folder, "samples", lambda rows: rows[0].update(bytes=-1)), "invalid integer bytes")
    invalid_mutation(folder, lambda: edit_rows(folder, "metrics", lambda rows: rows[0].update(data_hash="a" * 40)),
                     "malformed data_hash")
    invalid_mutation(folder, lambda: edit_rows(folder, "catalog", lambda rows: rows[0].update(scenario="control-other"), rep=2),
                     "catalog changed")
    invalid_mutation(folder, lambda: edit_rows(folder, "catalog", lambda rows: rows[0].update(estimator="other")),
                     "catalog id disagrees")
    invalid_mutation(folder, lambda: edit_rows(folder, "catalog", lambda rows: rows[0].update(scenario="")),
                     "missing scenario")
    invalid_mutation(folder, lambda: edit_rows(folder, "catalog", lambda rows: rows[0].update(phase="workspace")),
                     "unknown phase")
    invalid_mutation(folder, lambda: edit_rows(folder, "catalog", lambda rows: rows[0].update(measure="self_consistency")),
                     "invalid inputs/measure")
    # A target is a positive finite threshold or `record`; NaN and empty fields are neither.
    for value in ("NaN", "", "0", "-1e-6", "Inf", "loose"):
        invalid_mutation(folder, lambda value=value: edit_rows(folder, "catalog", lambda rows: rows[0].update(target=value)),
                         "invalid target")
    invalid_mutation(folder, lambda: edit_rows(folder, "metrics", lambda rows: rows[0].update(target="record")),
                     "metric target differs from catalog")
    invalid_mutation(folder, lambda: edit_meta(folder, "seconds_limit", 2.0, rep=2), "series settings changed")
    invalid_mutation(folder, lambda: edit_meta(folder, "expected_package_root", "/other/package"), "loaded package root")
    missing_process = folder / f"julia-{VERSION}-rep03-metadata-start.toml"
    invalid_mutation(folder, lambda: [path.unlink() for path in folder.glob(f"julia-{VERSION}-rep03-*")], "process coverage")
    assert missing_process.exists()
    assert not validate(folder, 2)["valid"]
    assert not validate(folder, 4)["valid"]
    assert not validate(folder, 0)["valid"]
    assert not validate(root / "absent")["valid"]
    assert not validate(folder, 3, ["9.9.9"])["valid"]
    fixture(folder, version="1.11.0")
    assert validate(folder, 3, [VERSION, "1.11.0"])["valid"]
    assert validate(folder, 3)["counts"]["processes"] == 6
    assert not validate(folder, 3, [VERSION])["valid"]


def compatibility_controls(root):
    before, after = root / "compatible-before", root / "compatible-after"
    fixture(before)
    fixture(after)
    edit_meta(after, "package_hash", "e" * 40)
    edit_meta(after, "package_sha", "f" * 40)
    edit_meta(after, "package_dirty", True)
    edit_meta(after, "package_version", "0.2.1")
    # The paths of the dependencies do not identify them; the commits do.
    edit_meta(after, "proximalbase_path", "/elsewhere/ProximalBase")
    assert len(compare(before, after, root / "changed-package", 3)) == 1
    replacements = dict(suite="extended", groups="lasso,controls", harness_hash="e" * 40,
                        blas="other", cpu_model="other", samples_target=10, seconds_limit=2.0,
                        case_seconds=30.0, profile_seconds=1.0, seed_scheme="other",
                        proximalbase_sha="f" * 40, proximalbase_dirty=True, proximalbase_version="0.6.1",
                        proximalsolvers_sha="f" * 40, proximalsolvers_dirty=True,
                        proximalsolvers_version="0.3.3")
    assert set(replacements) == set(SERIES_SETTINGS) - {
        "package_hash", "julia_threads", "blas_threads", "dependencies"}
    for key, value in replacements.items():
        folder = root / f"incompatible-{key}"
        fixture(folder)
        edit_meta(folder, key, value)
        assert validate(folder, 3)["valid"]
        message = expect_error(lambda: compare(before, folder, root / f"reject-{key}", 3), "series settings differ")
        assert key in message
    # A consistent series with other dependencies is valid but not comparable.
    dependency_changes = (("ProximalBase", "version", "0.6.1"), ("StatsAPI", "tree_hash", "e" * 40),
                          ("Extra", "version", "1.0.0"))
    for package, field, value in dependency_changes:
        folder = root / f"dependency-{package}-{field}"
        fixture(folder)
        if package == "Extra":
            edit_dependency(folder, package, "tree_hash", "f" * 40)
        edit_dependency(folder, package, field, value)
        assert validate(folder, 3)["valid"]
        message = expect_error(lambda: compare(before, folder, root / f"reject-{package}-{field}", 3),
                               "Comparison dependencies differ in version or tree hash")
        old = DEPENDENCIES.get(package, {}).get(field)
        assert repr((VERSION, package, field, old, value)) in message, message
    after2 = root / "two-repetitions"
    fixture(after2, runs=2)
    message = expect_error(lambda: compare(before, after2, root / "reject-missing-rep", 3), "Invalid results")
    assert "process coverage" in message and f"julia-{VERSION}-rep03" in message
    expect_error(lambda: compare(after2, before, root / "reject-extra-rep", 2), "Invalid results")
    folder = root / "hash-mismatch"
    fixture(folder)
    edit_rows(folder, "metrics", lambda rows: rows[0].update(data_hash="e" * 64), rep=2)
    assert validate(folder, 3)["valid"]
    message = expect_error(lambda: compare(before, folder, root / "reject-hash", 3), "data_hash mismatch")
    assert repr((VERSION, DEFAULT_CASE, 2)) in message
    for field, value in (("measure", "kkt_residual"), ("target", 1e-5), ("target", "record")):
        folder = root / f"different-{field}-{value}"
        fixture(folder, **{field: value})
        assert validate(folder, 3)["valid"]
        expect_error(lambda: compare(before, folder, root / f"reject-{field}-{value}", 3),
                     "certificate definitions differ")
    unmatched = root / "unmatched"
    other = "lasso/lasso/other"
    fixture(unmatched, cases=(other,))
    output = root / "reject-unmatched"
    expect_error(lambda: compare(before, unmatched, output, 3), "Case coverage differs")
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["before_only"] == [[VERSION, DEFAULT_CASE, rep] for rep in range(1, 4)]
    assert coverage["after_only"] == [[VERSION, other, rep] for rep in range(1, 4)]
    assert coverage["excluded"] == [
        dict(julia=VERSION, id=DEFAULT_CASE, before=[], after=[f"rep{rep:02d}: missing case" for rep in range(1, 4)]),
        dict(julia=VERSION, id=other, before=[f"rep{rep:02d}: missing case" for rep in range(1, 4)], after=[])]
    occupied = root / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep.txt"
    sentinel.write_text("preserve\n")
    expect_error(lambda: summarize(before, occupied, 3), "fresh output directory")
    expect_error(lambda: compare(before, after, occupied, 3), "fresh output directory")
    assert {path.name for path in occupied.iterdir()} == {"keep.txt"}
    assert sentinel.read_text() == "preserve\n"


def cli_controls(root):
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    def run(script, *args):
        return subprocess.run([sys.executable, str(scripts / script), *map(str, args)],
                              capture_output=True, text=True, env=env, check=False)
    for name in ("validate.py", "summarize.py", "compare.py"):
        assert run(name, "--help").returncode == 0
    valid = root / "failed-before-target_missed"
    report = root / "cli-validation.json"
    result = run("validate.py", valid, report, "--expected-runs", 3)
    assert result.returncode == 0, result.stderr
    checked = json.loads(report.read_text())
    assert checked["valid"] and checked["status_counts"]["target_missed"] == 3
    result = run("validate.py", root / "integrity", report, "--expected-runs", 3,
                 "--versions", VERSION, "1.11.0")
    assert result.returncode == 0, result.stderr
    result = run("validate.py", root / "missing", report)
    assert result.returncode == 2 and json.loads(report.read_text())["valid"] is False
    result = run("validate.py", valid, root / "nonexistent" / "out.json", "--expected-runs", 3)
    assert result.returncode == 3 and "Cannot write validation JSON" in result.stderr
    result = run("compare.py", valid, valid, root / "cli-failed", "--expected-runs", 3)
    assert result.returncode == 0, result.stderr
    result = run("summarize.py", valid, root / "cli-summary", "--expected-runs", 3)
    assert result.returncode == 0, result.stderr
    result = run("summarize.py", root / "integrity", root / "cli-versions", "--expected-runs", 3,
                 "--versions", VERSION)
    assert result.returncode == 2 and "Invalid results" in result.stderr
    result = run("compare.py", root / "missing", valid, root / "cli-invalid", "--expected-runs", 3)
    assert result.returncode == 2 and "Invalid results" in result.stderr
    result = run("compare.py", root / "compatible-before", root / "dependency-ProximalBase-version",
                 root / "cli-dependencies", "--expected-runs", 3)
    assert result.returncode == 2 and "dependencies differ" in result.stderr


def main():
    if len(sys.argv) != 1:
        raise SystemExit("check_analysis.py takes no arguments")
    with tempfile.TemporaryDirectory(prefix="hdm-analysis-control-") as temporary:
        root = Path(temporary)
        analytic_controls(root)
        allocation_controls(root)
        sample_controls(root)
        failure_controls(root)
        target_missed_controls(root)
        quoting_controls(root)
        integrity_controls(root)
        compatibility_controls(root)
        cli_controls(root)
    print("ANALYSIS_CONTROLS_PASSED repetitions=1,3,5,10; medians,t-intervals,paired-ratios,"
          "allocations,statuses,target-missed-exclusion,all-failed,quoted-text,interrupted,corruption,"
          "coverage,provenance,series-settings,dependencies,data-hashes,certificates,"
          "fresh-output,CLI")


if __name__ == "__main__":
    main()
