#!/usr/bin/env python3
"""Standalone syzbot upstream reproducer statistics script.

External dependencies:
    python -m pip install requests beautifulsoup4 lxml rich
"""

import csv
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup
import requests
from rich.console import Console
from rich.table import Table


LOGGER = logging.getLogger(__name__)
USER_AGENT = "syzbot-repro-stats/1.0"
OUTPUT_HEADERS = (
    "Status",
    "C+Syz Repro",
    "Syz Repro",
    "No Repro",
    "Total",
)
PAGES = (
    ("Open", "https://syzkaller.appspot.com/upstream"),
    ("Fixed", "https://syzkaller.appspot.com/upstream/fixed"),
    ("Invalid", "https://syzkaller.appspot.com/upstream/invalid"),
)


class ParseError(RuntimeError):
    """Raised when a syzbot list page cannot be parsed safely."""


class FetchError(RuntimeError):
    """Raised when a syzbot list page cannot be downloaded."""


class OutputError(RuntimeError):
    """Raised when a statistics snapshot cannot be written."""


@dataclass(frozen=True)
class ReproStats:
    """Counts for the three possible values in a syzbot Repro column."""

    c_repro: int = 0
    syz_repro: int = 0
    no_repro: int = 0

    @property
    def total(self) -> int:
        return self.c_repro + self.syz_repro + self.no_repro


def parse_repro_stats(html: str) -> ReproStats:
    """Parse and combine all vulnerability tables in one syzbot page."""
    soup = BeautifulSoup(html, "lxml")
    c_repro = 0
    syz_repro = 0
    no_repro = 0

    for table in soup.select("table.list_table"):
        headers = [
            cell.get_text(" ", strip=True)
            for cell in table.select("thead th")
        ]
        if "Title" not in headers or "Repro" not in headers:
            continue

        repro_index = headers.index("Repro")
        for row in table.select("tbody > tr"):
            cells = row.find_all("td", recursive=False)
            if repro_index >= len(cells):
                raise ParseError("vulnerability row is missing Repro cell")

            value = cells[repro_index].get_text(" ", strip=True)
            if value == "C":
                c_repro += 1
            elif value == "syz":
                syz_repro += 1
            elif value == "":
                no_repro += 1
            else:
                raise ParseError(f"unknown Repro value: {value!r}")

    stats = ReproStats(c_repro, syz_repro, no_repro)
    if stats.total == 0:
        raise ParseError("no vulnerability rows found")
    return stats


def sum_stats(items: Iterable[ReproStats]) -> ReproStats:
    """Add reproducer counts without mutating the source results."""
    c_repro = 0
    syz_repro = 0
    no_repro = 0
    for stats in items:
        c_repro += stats.c_repro
        syz_repro += stats.syz_repro
        no_repro += stats.no_repro
    return ReproStats(c_repro, syz_repro, no_repro)


def format_stat(count: int, total: int) -> str:
    """Format a count and its percentage of a row total."""
    percentage = count / total * 100 if total else 0.0
    return f"{count} ({percentage:.2f}%)"


def _display_rows(
    results: Sequence[Tuple[str, ReproStats]],
) -> List[Tuple[str, str, str, str, str]]:
    """Create the shared display rows for terminal and CSV output."""
    total_stats = sum_stats(stats for _, stats in results)
    stats_rows = list(results) + [("Total", total_stats)]
    return [
        (
            status,
            format_stat(stats.c_repro, stats.total),
            format_stat(stats.syz_repro, stats.total),
            format_stat(stats.no_repro, stats.total),
            str(stats.total),
        )
        for status, stats in stats_rows
    ]


def default_output_path(now: Optional[datetime] = None) -> Path:
    """Return a timestamped CSV path in the current working directory."""
    timestamp = now or datetime.now()
    filename = timestamp.strftime(
        "syzbot_upstream_repro_stats_%Y%m%d_%H%M%S.csv"
    )
    return Path.cwd() / filename


def write_csv(
    results: Sequence[Tuple[str, ReproStats]], output_path: Path
) -> Path:
    """Write the formatted statistics table to a UTF-8 CSV file."""
    output_path = Path(output_path)
    try:
        with output_path.open(
            "x", newline="", encoding="utf-8"
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(OUTPUT_HEADERS)
            writer.writerows(_display_rows(results))
    except OSError as error:
        raise OutputError(f"failed to write CSV to {output_path}") from error
    return output_path.resolve()


def fetch_page(
    session: requests.Session,
    url: str,
    attempts: int = 3,
    timeout: float = 30.0,
) -> str:
    """Fetch one page with bounded retries for request failures."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    session.headers.update({"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt < attempts:
                wait_seconds = 2 ** (attempt - 1)
                LOGGER.warning(
                    "Request failed for %s (attempt %d/%d): %s; "
                    "retrying in %ds",
                    url,
                    attempt,
                    attempts,
                    error,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

    raise FetchError(
        f"failed to fetch {url} after {attempts} attempts"
    ) from last_error


def build_table(results: Sequence[Tuple[str, ReproStats]]) -> Table:
    """Build the final Rich table, including an aggregate Total row."""
    table = Table(
        title="Syzbot Upstream Reproducer Statistics",
        header_style="bold cyan",
    )
    table.add_column(OUTPUT_HEADERS[0], style="bold")
    for header in OUTPUT_HEADERS[1:-1]:
        table.add_column(header, justify="right")
    table.add_column(OUTPUT_HEADERS[-1], justify="right", style="bold")

    display_rows = _display_rows(results)
    for index, row in enumerate(display_rows):
        if index == len(results):
            table.add_section()
        table.add_row(*row, style="bold" if row[0] == "Total" else None)
    return table


def run(
    session: Optional[requests.Session] = None,
    console: Optional[Console] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, ReproStats]:
    """Fetch, parse, aggregate, and print all upstream status pages."""
    started_at = time.perf_counter()
    console = console or Console()
    owns_session = session is None
    session = session or requests.Session()
    page_results: List[Tuple[str, ReproStats]] = []

    try:
        for status, url in PAGES:
            LOGGER.info("Fetching %s: %s", status, url)
            html = fetch_page(session, url)
            stats = parse_repro_stats(html)
            page_results.append((status, stats))
            LOGGER.info("Parsed %s: %d vulnerabilities", status, stats.total)
    finally:
        if owns_session:
            session.close()

    total_stats = sum_stats(stats for _, stats in page_results)
    results = dict(page_results)
    results["Total"] = total_stats

    if output_path is None:
        output_path = default_output_path()
    saved_path = write_csv(page_results, output_path)
    LOGGER.info("Saved results to %s", saved_path)

    console.print()
    console.print(build_table(page_results))
    elapsed = time.perf_counter() - started_at
    console.print(f"\n[bold]Elapsed time:[/bold] {elapsed:.2f} seconds")
    return results


def main() -> int:
    """Run the command-line script and return its process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        run()
    except (FetchError, OutputError, ParseError) as error:
        LOGGER.error("Statistics failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
