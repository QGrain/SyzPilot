#!/usr/bin/env python3
"""Collect syzkaller 5/10-run experiment results into the benchmark CSV format."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FIELDNAMES = [
    "ID",
    "Hit Count(24h)",
    "Avg Hit Count(24h)",
    "TTH(24h)",
    "μTTH(24h)",
    "Hit Count(48h)",
    "Avg Hit Count(48h)",
    "TTH(48h)",
    "μTTH(48h)",
    "TTE(24h)",
    "μTTE(24h)",
    "TTE(48h)",
    "μTTE(48h)",
]
COMMENTS = [
    "# Collect the original results of 10 runs and caculate the average results for them.",
    "# The format or original results is as RUN1/RUN2/RUN3/.../RUN10",
    "# μTTH is the average of Time-to-Hit, μTTE is the average of Time-to-Expire",
]
RAW_COLUMNS = [
    "Hit Count(24h)",
    "TTH(24h)",
    "Hit Count(48h)",
    "TTH(48h)",
    "TTE(24h)",
    "TTE(48h)",
]
AVG_COLUMNS = {
    "Hit Count(24h)": "Avg Hit Count(24h)",
    "TTH(24h)": "μTTH(24h)",
    "Hit Count(48h)": "Avg Hit Count(48h)",
    "TTH(48h)": "μTTH(48h)",
    "TTE(24h)": "μTTE(24h)",
    "TTE(48h)": "μTTE(48h)",
}
HOURS_24 = 24.0
HOURS_48 = 48.0
SECONDS_24 = int(HOURS_24 * 3600)
SECONDS_48 = int(HOURS_48 * 3600)
SIMILAR_REPORT_LIMIT = 3

logger = logging.getLogger(__name__)


_IN_RE = re.compile(r"\s+in\s+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_FUNCTION_OFFSET_RE = re.compile(
    r"(?:\+0x[0-9a-f]+(?:/0x[0-9a-f]+)?)$", re.IGNORECASE
)
_CASE_RE = re.compile(r"^case_(\d+)$")


@dataclass(frozen=True)
class BugTitle:
    raw: str
    normalized: str
    bug_type: str
    function: str | None
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class CrashCandidate:
    description: str
    parsed: BugTitle
    cosine: float
    edit_similarity: float


@dataclass
class RunMetrics:
    hit_count_24: int
    hit_count_48: int
    tth_24: float
    tth_48: float
    tte_24: float
    tte_48: float
    exact_exposure: bool
    similar_candidates: list[CrashCandidate]


def _normalize_function(value: str) -> str | None:
    function = value.strip().split(None, 1)[0] if value.strip() else ""
    function = function.split("(", 1)[0].rstrip("():,.;")
    function = _FUNCTION_OFFSET_RE.sub("", function)
    return function.casefold() or None


def parse_bug_title(title: str) -> BugTitle:
    raw = title.strip()
    normalized = _SPACE_RE.sub(" ", raw).casefold()
    parts = _IN_RE.split(raw)
    if len(parts) > 1:
        bug_type = _SPACE_RE.sub(" ", " in ".join(parts[:-1]).strip()).casefold()
        function = _normalize_function(parts[-1])
    else:
        bug_type = normalized
        function = None
    return BugTitle(
        raw=raw,
        normalized=normalized,
        bug_type=bug_type,
        function=function,
        tokens=tuple(_TOKEN_RE.findall(normalized)),
    )


def cosine_similarity(left: str | BugTitle, right: str | BugTitle) -> float:
    lhs = left if isinstance(left, BugTitle) else parse_bug_title(left)
    rhs = right if isinstance(right, BugTitle) else parse_bug_title(right)
    left_counts = Counter(lhs.tokens)
    right_counts = Counter(rhs.tokens)
    if not left_counts or not right_counts:
        return 0.0
    dot = sum(count * right_counts[token] for token, count in left_counts.items())
    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    return min(1.0, max(0.0, dot / (left_norm * right_norm)))


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def edit_similarity(left: str | BugTitle, right: str | BugTitle) -> float:
    lhs = left.normalized if isinstance(left, BugTitle) else parse_bug_title(left).normalized
    rhs = right.normalized if isinstance(right, BugTitle) else parse_bug_title(right).normalized
    width = max(len(lhs), len(rhs))
    if width == 0:
        return 1.0
    return 1.0 - levenshtein_distance(lhs, rhs) / width


def natural_case_key(case_id: str) -> tuple[int, int | str]:
    match = _CASE_RE.match(case_id)
    if match:
        return (0, int(match.group(1)))
    return (1, case_id)


def format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    number = float(value)
    if math.isclose(number, round(number), abs_tol=1e-9):
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def parse_number(value: str) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def average_slot_values(values: list[str]) -> str:
    nums = [parse_number(value) for value in values if str(value).strip()]
    nums = [num for num in nums if num is not None]
    if not nums:
        return ""
    return format_number(sum(nums) / len(nums))


def join_slots(values: list[str]) -> str:
    return "/".join(values)


def split_slots(value: str, total_runs: int | None = None) -> list[str]:
    slots = str(value or "").split("/")
    if total_runs is not None:
        if len(slots) < total_runs:
            slots += [""] * (total_runs - len(slots))
        elif len(slots) > total_runs:
            raise ValueError(
                f"slot count {len(slots)} exceeds expected total-runs {total_runs}"
            )
    return slots


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def iter_json_objects(path: Path) -> Iterable[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            obj, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        index = next_index
        if isinstance(obj, dict):
            yield obj


def pc_low32(value: str) -> int | None:
    """Parse a hexadecimal PC using the fuzzer's low-32-bit address contract."""
    try:
        return int(str(value).strip(), 16) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None


def reachability_value(obj: dict, target_pc: str) -> int | None:
    reach_values: list[tuple[str, int]] = []
    for key, value in obj.items():
        if not isinstance(key, str) or not key.startswith("reachability 0x"):
            continue
        token = key[len("reachability ") :].strip()
        try:
            reach_values.append((token, int(value)))
        except (TypeError, ValueError):
            continue

    if not reach_values:
        return None

    target_low32 = pc_low32(target_pc)
    if target_low32 is None:
        return None

    for token, value in reach_values:
        if pc_low32(token) == target_low32:
            return value

    return None


def parse_bench_metrics(bench_path: Path, target_pc: str) -> tuple[int, int, float, float]:
    hit_count_24 = 0
    hit_count_48 = 0
    first_hit_seconds: float | None = None
    saw_target_stat = False

    for obj in iter_json_objects(bench_path):
        try:
            uptime = float(obj.get("uptime"))
        except (TypeError, ValueError):
            continue

        value = reachability_value(obj, target_pc)
        if value is None:
            continue
        saw_target_stat = True

        if uptime <= SECONDS_24:
            hit_count_24 = value
        if uptime <= SECONDS_48:
            hit_count_48 = value
        if value > 0 and first_hit_seconds is None:
            first_hit_seconds = uptime

    if not saw_target_stat:
        raise ValueError(
            f"{bench_path}: no reachability statistic for final target {target_pc}"
        )

    tth_24 = first_hit_seconds / 3600.0 if first_hit_seconds is not None and first_hit_seconds <= SECONDS_24 else HOURS_24
    tth_48 = first_hit_seconds / 3600.0 if first_hit_seconds is not None and first_hit_seconds <= SECONDS_48 else HOURS_48
    return hit_count_24, hit_count_48, tth_24, tth_48


def stat_birth_time(path: Path) -> int | None:
    try:
        output = subprocess.check_output(
            ["stat", "--format=%W", str(path)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if output and output not in {"0", "-1"}:
            return int(output)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def earliest_mtime(path: Path) -> int | None:
    try:
        if path.is_file():
            return int(path.stat().st_mtime)
        mtimes = [int(child.stat().st_mtime) for child in path.iterdir()]
    except OSError:
        return None
    return min(mtimes) if mtimes else None


def crash_time_seconds(description_path: Path) -> int | None:
    birth = stat_birth_time(description_path)
    if birth is not None:
        return birth
    return earliest_mtime(description_path.parent)


def iter_crash_descriptions(crashes_dir: Path) -> Iterable[tuple[Path, str]]:
    if not crashes_dir.is_dir():
        return
    for description_path in sorted(crashes_dir.glob("*/description")):
        if not description_path.is_file():
            continue
        try:
            description = description_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            continue
        if description:
            yield description_path, description


def collect_exposure(
    cfg_path: Path,
    crashes_dir: Path,
    task_name: str,
    edit_threshold: float,
    cosine_threshold: float,
) -> tuple[float | None, list[CrashCandidate]]:
    parsed_target = parse_bug_title(task_name)
    target_norm = parsed_target.normalized
    try:
        cfg_time = int(cfg_path.stat().st_mtime)
    except OSError:
        cfg_time = None

    exact_times: list[float] = []
    similar_candidates: list[CrashCandidate] = []

    for description_path, description in iter_crash_descriptions(crashes_dir):
        parsed = parse_bug_title(description)

        if parsed.normalized == target_norm:
            crash_time = crash_time_seconds(description_path)
            if cfg_time is not None and crash_time is not None:
                exact_times.append(max(0.0, (crash_time - cfg_time) / 3600.0))
            continue

        if not parsed_target.function or parsed.function != parsed_target.function:
            continue

        candidate = CrashCandidate(
            description=description,
            parsed=parsed,
            cosine=cosine_similarity(parsed_target, parsed),
            edit_similarity=edit_similarity(parsed_target, parsed),
        )
        if (
            candidate.edit_similarity >= edit_threshold
            or candidate.cosine >= cosine_threshold
        ):
            similar_candidates.append(candidate)

    exact_tte = min(exact_times) if exact_times else None
    if exact_tte is not None:
        return exact_tte, []

    sorted_candidates = sorted(
        similar_candidates,
        key=lambda item: (
            -max(item.edit_similarity, item.cosine),
            -item.edit_similarity,
            -item.cosine,
            item.description.casefold(),
        )
    )
    deduped_candidates: list[CrashCandidate] = []
    seen_titles: set[str] = set()
    for candidate in sorted_candidates:
        if candidate.parsed.normalized in seen_titles:
            continue
        seen_titles.add(candidate.parsed.normalized)
        deduped_candidates.append(candidate)
        if len(deduped_candidates) >= SIMILAR_REPORT_LIMIT:
            break
    return None, deduped_candidates


def read_run_config(cfg_path: Path) -> tuple[str, str]:
    cfg = read_json(cfg_path)
    syzpilot = cfg.get("SyzPilot", {})
    target_pcs = syzpilot.get("target_pcs") or []
    if not target_pcs:
        raise ValueError(f"{cfg_path}: missing SyzPilot.target_pcs")
    task_name = syzpilot.get("task_name")
    if not task_name:
        raise ValueError(f"{cfg_path}: missing SyzPilot.task_name")
    return str(task_name), str(target_pcs[-1])


def collect_run_metrics(
    case_dir: Path,
    actual_run: int,
    edit_threshold: float,
    cosine_threshold: float,
) -> RunMetrics:
    cfg_path = case_dir / f"{actual_run}.cfg"
    bench_path = case_dir / f"{actual_run}.bench.log"
    if not cfg_path.is_file():
        raise ValueError(f"missing cfg: {cfg_path}")
    if not bench_path.is_file():
        raise ValueError(f"missing bench log: {bench_path}")

    task_name, target_pc = read_run_config(cfg_path)
    hit_count_24, hit_count_48, tth_24, tth_48 = parse_bench_metrics(
        bench_path, target_pc
    )
    exact_tte, similar_candidates = collect_exposure(
        cfg_path,
        case_dir / str(actual_run) / "crashes",
        task_name,
        edit_threshold,
        cosine_threshold,
    )
    tte_24 = exact_tte if exact_tte is not None and exact_tte <= HOURS_24 else HOURS_24
    tte_48 = exact_tte if exact_tte is not None and exact_tte <= HOURS_48 else HOURS_48
    return RunMetrics(
        hit_count_24=hit_count_24,
        hit_count_48=hit_count_48,
        tth_24=tth_24,
        tth_48=tth_48,
        tte_24=tte_24,
        tte_48=tte_48,
        exact_exposure=exact_tte is not None,
        similar_candidates=similar_candidates,
    )


def discover_actual_runs(case_dir: Path) -> list[int]:
    runs = []
    for cfg_path in case_dir.glob("*.cfg"):
        stem = cfg_path.name[: -len(".cfg")]
        if stem.isdigit() and (case_dir / f"{stem}.bench.log").is_file():
            runs.append(int(stem))
    return sorted(set(runs))


def empty_case_slots(total_runs: int) -> dict[str, list[str]]:
    return {column: [""] * total_runs for column in RAW_COLUMNS}


def row_from_slots(case_id: str, slots: dict[str, list[str]]) -> dict[str, str]:
    row = {"ID": case_id}
    for column in RAW_COLUMNS:
        values = slots[column]
        row[column] = join_slots(values)
        row[AVG_COLUMNS[column]] = average_slot_values(values)
    return row


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_result_csv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        for comment in COMMENTS:
            f.write(comment + "\n")
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_similar_report(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "ID",
        "slot",
        "actual_run",
        "rank",
        "target_title",
        "candidate_description",
        "edit_similarity",
        "cosine_similarity",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slot_numbers(row: dict[str, str], column: str) -> list[float]:
    return [
        number
        for value in split_slots(row.get(column, ""))
        if (number := parse_number(value)) is not None
    ]


def summarize_result_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    summary = {
        "cases": len(rows),
        "runs": 0,
        "cases_hit_24h": 0,
        "cases_hit_48h": 0,
        "hit_runs_24h": 0,
        "hit_runs_48h": 0,
        "cases_exposed_24h": 0,
        "cases_exposed_48h": 0,
        "expose_runs_24h": 0,
        "expose_runs_48h": 0,
    }
    for row in rows:
        hit_24 = slot_numbers(row, "Hit Count(24h)")
        hit_48 = slot_numbers(row, "Hit Count(48h)")
        tte_24 = slot_numbers(row, "TTE(24h)")
        tte_48 = slot_numbers(row, "TTE(48h)")

        hit_runs_24 = sum(1 for value in hit_24 if value > 0)
        hit_runs_48 = sum(1 for value in hit_48 if value > 0)
        expose_runs_24 = sum(1 for value in tte_24 if value < HOURS_24)
        expose_runs_48 = sum(1 for value in tte_48 if value < HOURS_48)

        summary["runs"] += len(hit_48)
        summary["hit_runs_24h"] += hit_runs_24
        summary["hit_runs_48h"] += hit_runs_48
        summary["expose_runs_24h"] += expose_runs_24
        summary["expose_runs_48h"] += expose_runs_48
        summary["cases_hit_24h"] += int(hit_runs_24 > 0)
        summary["cases_hit_48h"] += int(hit_runs_48 > 0)
        summary["cases_exposed_24h"] += int(expose_runs_24 > 0)
        summary["cases_exposed_48h"] += int(expose_runs_48 > 0)
    return summary


def log_result_summary(label: str, rows: list[dict[str, str]]) -> None:
    summary = summarize_result_rows(rows)
    logger.info(
        "%s summary: cases=%d runs=%d",
        label,
        summary["cases"],
        summary["runs"],
    )
    logger.info(
        "%s reachability: 24h cases=%d/%d hit_runs=%d/%d; "
        "48h cases=%d/%d hit_runs=%d/%d",
        label,
        summary["cases_hit_24h"],
        summary["cases"],
        summary["hit_runs_24h"],
        summary["runs"],
        summary["cases_hit_48h"],
        summary["cases"],
        summary["hit_runs_48h"],
        summary["runs"],
    )
    logger.info(
        "%s exposure: 24h cases=%d/%d expose_runs=%d/%d; "
        "48h cases=%d/%d expose_runs=%d/%d",
        label,
        summary["cases_exposed_24h"],
        summary["cases"],
        summary["expose_runs_24h"],
        summary["runs"],
        summary["cases_exposed_48h"],
        summary["cases"],
        summary["expose_runs_48h"],
        summary["runs"],
    )


def collect_command(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not root.is_dir():
        raise ValueError(f"result root does not exist: {root}")

    total_runs = args.total_runs
    if total_runs <= 0:
        raise ValueError("--total-runs must be positive")
    if args.run_offset < 0:
        raise ValueError("--run-offset must be non-negative")

    out_path = Path(args.out)
    similar_report = Path(args.similar_report) if args.similar_report else out_path.with_suffix(".similar_matches.csv")
    ensure_parent_dir(out_path)
    ensure_parent_dir(similar_report)
    logger.info("Collecting syzkaller results: root=%s out=%s", root, out_path)
    logger.info("Similar-title report: %s", similar_report)

    rows: list[dict[str, str]] = []
    similar_rows: list[dict[str, str]] = []
    warnings: list[str] = []

    case_dirs = [
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("case_")
    ]
    for case_dir in sorted(case_dirs, key=lambda path: natural_case_key(path.name)):
        slots = empty_case_slots(total_runs)
        for actual_run in discover_actual_runs(case_dir):
            slot = actual_run + args.run_offset
            if slot < 1 or slot > total_runs:
                warnings.append(
                    f"{case_dir.name}/run{actual_run}: slot {slot} outside 1..{total_runs}; skipped"
                )
                continue

            try:
                metrics = collect_run_metrics(
                    case_dir,
                    actual_run,
                    args.similar_edit_threshold,
                    args.similar_cosine_threshold,
                )
                task_name, _ = read_run_config(case_dir / f"{actual_run}.cfg")
            except ValueError as exc:
                warnings.append(str(exc))
                continue

            index = slot - 1
            slots["Hit Count(24h)"][index] = str(metrics.hit_count_24)
            slots["Hit Count(48h)"][index] = str(metrics.hit_count_48)
            slots["TTH(24h)"][index] = format_number(metrics.tth_24)
            slots["TTH(48h)"][index] = format_number(metrics.tth_48)
            slots["TTE(24h)"][index] = format_number(metrics.tte_24)
            slots["TTE(48h)"][index] = format_number(metrics.tte_48)

            for rank, candidate in enumerate(metrics.similar_candidates, 1):
                similar_rows.append(
                    {
                        "ID": case_dir.name,
                        "slot": str(slot),
                        "actual_run": str(actual_run),
                        "rank": str(rank),
                        "target_title": task_name,
                        "candidate_description": candidate.description,
                        "edit_similarity": f"{candidate.edit_similarity:.4f}",
                        "cosine_similarity": f"{candidate.cosine:.4f}",
                    }
                )

        rows.append(row_from_slots(case_dir.name, slots))

    write_result_csv(out_path, rows)
    write_similar_report(similar_report, similar_rows)

    log_result_summary("collect", rows)
    logger.info("similar_candidates_reported=%d", len(similar_rows))
    logger.info("output_csv=%s", out_path)
    logger.info("similar_report=%s", similar_report)
    for warning in warnings[:50]:
        logger.warning(warning)
    if len(warnings) > 50:
        logger.warning("%d additional warnings omitted", len(warnings) - 50)
    return 0


def read_result_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)
    if reader.fieldnames != FIELDNAMES:
        raise ValueError(f"{path}: unexpected CSV header: {reader.fieldnames}")
    return list(reader)


def infer_total_runs(input_rows: Iterable[dict[str, str]]) -> int:
    total = 0
    for row in input_rows:
        for column in RAW_COLUMNS:
            total = max(total, len(str(row.get(column, "")).split("/")))
    return total


def equivalent_slot_values(left: str, right: str) -> bool:
    if left == right:
        return True
    left_num = parse_number(left)
    right_num = parse_number(right)
    if left_num is not None and right_num is not None:
        return math.isclose(left_num, right_num, rel_tol=0.0, abs_tol=1e-9)
    return False


def merge_command(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    ensure_parent_dir(out_path)
    logger.info("Merging result CSVs: inputs=%s out=%s", ",".join(args.inputs), out_path)

    all_input_rows: list[tuple[Path, dict[str, str]]] = []
    for input_value in args.inputs:
        path = Path(input_value)
        for row in read_result_csv(path):
            all_input_rows.append((path, row))

    total_runs = args.total_runs or infer_total_runs(row for _, row in all_input_rows)
    if total_runs <= 0:
        raise ValueError("could not infer total runs")

    merged: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: empty_case_slots(total_runs)
    )
    conflicts: list[str] = []

    for path, row in all_input_rows:
        case_id = row["ID"]
        for column in RAW_COLUMNS:
            slots = split_slots(row.get(column, ""), total_runs=total_runs)
            for index, value in enumerate(slots):
                value = value.strip()
                if not value:
                    continue
                existing = merged[case_id][column][index]
                if existing and not equivalent_slot_values(existing, value):
                    conflicts.append(
                        f"{case_id} {column} slot{index + 1}: {existing} vs {value} from {path}"
                    )
                    continue
                merged[case_id][column][index] = format_number(parse_number(value))

    if conflicts:
        for conflict in conflicts[:50]:
            logger.error("merge conflict: %s", conflict)
        if len(conflicts) > 50:
            logger.error("%d additional conflicts omitted", len(conflicts) - 50)
        return 2

    warnings = []
    output_rows = []
    for case_id in sorted(merged, key=natural_case_key):
        for column in RAW_COLUMNS:
            missing = [i + 1 for i, value in enumerate(merged[case_id][column]) if not value]
            if missing:
                warnings.append(f"{case_id} {column}: missing slots {missing}")
        output_rows.append(row_from_slots(case_id, merged[case_id]))

    write_result_csv(out_path, output_rows)
    log_result_summary("merge", output_rows)
    logger.info("total_runs=%d", total_runs)
    logger.info("output_csv=%s", out_path)
    for warning in warnings[:50]:
        logger.warning(warning)
    if len(warnings) > 50:
        logger.warning("%d additional warnings omitted", len(warnings) - 50)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect/merge syzkaller experiment results in benchmark CSV format"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect one result directory")
    collect.add_argument("--root", required=True, help="path to exp/syzkaller")
    collect.add_argument("--out", required=True, help="output CSV path")
    collect.add_argument(
        "--run-offset",
        type=int,
        default=0,
        help="offset added to actual run ids when writing output slots",
    )
    collect.add_argument(
        "--total-runs",
        type=int,
        default=10,
        help="number of slash-separated run slots in output",
    )
    collect.add_argument(
        "--similar-report",
        default=None,
        help="CSV path for high-similarity non-exact exposure candidates",
    )
    collect.add_argument(
        "--similar-edit-threshold",
        type=float,
        default=0.80,
        help="report non-exact candidates with edit similarity at or above this value",
    )
    collect.add_argument(
        "--similar-cosine-threshold",
        type=float,
        default=0.80,
        help="report non-exact candidates with cosine similarity at or above this value",
    )
    collect.set_defaults(func=collect_command)

    merge = subparsers.add_parser("merge", help="merge partial result CSV files")
    merge.add_argument("--inputs", nargs="+", required=True, help="partial CSV inputs")
    merge.add_argument("--out", required=True, help="merged CSV output path")
    merge.add_argument(
        "--total-runs",
        type=int,
        default=None,
        help="expected total slots; inferred from inputs when omitted",
    )
    merge.set_defaults(func=merge_command)

    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
