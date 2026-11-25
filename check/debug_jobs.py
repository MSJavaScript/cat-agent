#!/usr/bin/env python3
"""Print the jobs known to a running jobmanager as a table.

The jobmanager exposes its durable job list at ``GET /jobs`` (see
``catagent/jobmanager/server.py``).  This is a read-only debug helper: it sends
that request and renders the JSON records as a table so you can see which jobs
exist and how far each one got.

Examples
--------
    python -m check.debug_jobs
    python -m check.debug_jobs --url http://127.0.0.1:8123
    python -m check.debug_jobs --status running --status failed
    python -m check.debug_jobs --full          # full UUID, submit/finalize state
    python -m check.debug_jobs --sub-jobs      # also print the per-group rows
    python -m check.debug_jobs --json          # raw /jobs payload
    python -m check.debug_jobs --watch 2       # refresh every two seconds

The default URL can also be set with CATAGENT_JOBMANAGER_URL.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional, Sequence

DEFAULT_URL = os.environ.get("CATAGENT_JOBMANAGER_URL", "http://127.0.0.1:8000")
GAP = "  "
ELLIPSIS = "\u2026"

ANSI = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}
RESET = "\033[0m"

STATUS_COLORS = {
    "completed": "green",
    "success": "green",
    "failed": "red",
    "error": "red",
    "cancelled": "red",
    "running": "yellow",
    "submitting": "yellow",
    "planning": "yellow",
    "finalizing": "yellow",
    "ready": "yellow",
    "submitted": "cyan",
    "pending": "cyan",
    "unknown": "dim",
    "none": "dim",
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        raise ValueError("jobmanager URL is empty")
    if "://" not in url:
        url = "http://" + url
    return url


def jobs_url(base_url: str, limit: int) -> str:
    url = f"{base_url}/jobs"
    if limit > 0:
        url += "?" + urllib.parse.urlencode({"limit": limit})
    return url


def fetch_jobs(base_url: str, limit: int, timeout: float) -> list[dict[str, Any]]:
    url = jobs_url(base_url, limit)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise RuntimeError(
            f"GET {url} failed with HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach jobmanager at {url}: {exc.reason}"
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(
            f"GET {url} returned {type(payload).__name__}, expected a JSON list"
        )
    return [item for item in payload if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Small text/table helpers (stdlib only; East-Asian widths are handled)
# ---------------------------------------------------------------------------
def char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def text_width(text: str) -> int:
    return sum(char_width(char) for char in text)


def truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if text_width(text) <= width:
        return text
    if width == 1:
        return ELLIPSIS
    kept: list[str] = []
    used = 0
    for char in text:
        size = char_width(char)
        if used + size > width - 1:
            break
        kept.append(char)
        used += size
    return "".join(kept) + ELLIPSIS


def pad(text: str, width: int, align: str = "left") -> str:
    text = truncate(text, width)
    fill = " " * max(0, width - text_width(text))
    return fill + text if align == "right" else text + fill


def paint(text: str, color: Optional[str], enabled: bool) -> str:
    if not enabled or not color:
        return text
    return f"{ANSI.get(color, '')}{text}{RESET}"


def one_line(value: Any) -> str:
    if value is None:
        return "-"
    text = " ".join(str(value).split())
    return text or "-"


def short_uuid(value: Any, length: int = 8) -> str:
    """Show a UUID prefix without an ellipsis (git-style short id)."""
    text = str(value or "-")
    return text if len(text) <= length else text[:length]


def fmt_timestamp(value: Any) -> str:
    if not value:
        return "-"
    text = str(value)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if moment.tzinfo is not None:
            moment = moment.astimezone(timezone.utc)
        return moment.strftime("%m-%d %H:%M:%S")
    except ValueError:
        return text[:14]


def status_color(value: str) -> str:
    return STATUS_COLORS.get(value.lower(), "dim")


class Column:
    def __init__(
        self,
        title: str,
        getter: Callable[[dict[str, Any]], str],
        width: int,
        align: str = "left",
        color: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.title = title
        self.getter = getter
        self.width = width
        self.align = align
        self.color = color


def render_table(
    columns: Sequence[Column],
    rows: Sequence[dict[str, Any]],
    use_color: bool,
) -> list[str]:
    last = len(columns) - 1

    def cell(column: Column, index: int, value: str) -> str:
        if index == last:
            text = truncate(value, column.width)
            color = column.color(value) if column.color else None
            return paint(text, color, use_color)
        text = pad(value, column.width, column.align)
        color = column.color(value) if column.color else None
        return paint(text, color, use_color)

    lines: list[str] = []
    header_cells: list[str] = []
    for index, column in enumerate(columns):
        text = column.title if index == last else pad(column.title, column.width)
        header_cells.append(paint(text, "bold", use_color))
    lines.append(GAP.join(header_cells).rstrip())

    last_column = columns[-1]
    last_width = text_width(last_column.title)
    for row in rows:
        last_width = max(last_width, text_width(last_column.getter(row)))
    last_width = min(last_width, last_column.width)
    underline = [column.width for column in columns[:-1]] + [last_width]
    lines.append(GAP.join("\u2500" * width for width in underline).rstrip())
    for row in rows:
        values = [column.getter(row) for column in columns]
        lines.append(
            GAP.join(
                cell(column, index, value)
                for index, (column, value) in enumerate(zip(columns, values))
            )
        )
    return lines


def build_job_columns(
    *,
    full: bool,
    show_description: bool,
    description_width: Optional[int],
    terminal_width: int,
) -> list[Column]:
    columns = [
        Column(
            "JOB",
            lambda job: short_uuid(job.get("job_uuid"), 36 if full else 8),
            36 if full else 8,
        ),
        Column(
            "STATUS",
            lambda job: str(job.get("status") or "-"),
            9,
            color=status_color,
        ),
    ]
    if full:
        columns.extend(
            [
                Column(
                    "SUBMIT",
                    lambda job: str(job.get("submission_state") or "-"),
                    10,
                ),
                Column(
                    "FINALIZE",
                    lambda job: str(job.get("finalize_state") or "-"),
                    10,
                ),
                Column(
                    "GROUPS",
                    lambda job: str(len(job.get("sub_jobs") or [])),
                    6,
                    align="right",
                ),
            ]
        )
    columns.append(
        Column(
            "RESULT",
            lambda job: str(job.get("result_status") or "-"),
            7,
            color=status_color,
        )
    )
    columns.append(
        Column(
            "PROVIDER / ENV",
            lambda job: f"{job.get('provider_id') or '-'}/"
            f"{job.get('job_env_name') or '-'}",
            24,
        )
    )
    columns.extend(
        [
            Column(
                "ATT",
                lambda job: str(job.get("attempt", 0)),
                3,
                align="right",
            ),
            Column(
                "UPDATED (UTC)",
                lambda job: fmt_timestamp(job.get("updated_at")),
                14,
            ),
        ]
    )

    if show_description:
        if description_width is not None:
            width = description_width
        else:
            fixed = sum(column.width for column in columns) + len(GAP) * len(columns)
            width = max(12, terminal_width - fixed)
        columns.append(
            Column(
                "DESCRIPTION",
                lambda job: one_line(job.get("description")),
                width,
            )
        )
    return columns


def build_sub_job_columns(description_width: int) -> list[Column]:
    return [
        Column("JOB", lambda sub: short_uuid(sub.get("job_uuid"), 8), 8),
        Column(
            "GROUP", lambda sub: str(sub.get("group_index", "-")), 5, align="right"
        ),
        Column("REMOTE ID", lambda sub: str(sub.get("remote_id") or "-"), 14),
        Column(
            "STATUS",
            lambda sub: str(sub.get("status") or "-"),
            9,
            color=status_color,
        ),
        Column(
            "REMOTE DIR",
            lambda sub: str(sub.get("remote_dir") or "-"),
            24,
        ),
        Column(
            "UPDATED (UTC)",
            lambda sub: fmt_timestamp(sub.get("updated_at")),
            14,
        ),
        Column(
            "ERROR",
            lambda sub: one_line(sub.get("error")),
            max(12, description_width),
        ),
    ]


def sub_job_rows(jobs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        for sub in job.get("sub_jobs") or []:
            if isinstance(sub, dict):
                merged = dict(sub)
                merged.setdefault("job_uuid", job.get("job_uuid"))
                rows.append(merged)
    return rows


# ---------------------------------------------------------------------------
# Filtering / printing
# ---------------------------------------------------------------------------
def filter_jobs(
    jobs: Iterable[dict[str, Any]],
    statuses: Sequence[str],
    providers: Sequence[str],
    grep: Optional[str],
) -> list[dict[str, Any]]:
    wanted_status = {value.lower() for value in statuses}
    wanted_provider = {value.lower() for value in providers}
    needle = (grep or "").lower()
    selected: list[dict[str, Any]] = []
    for job in jobs:
        status = str(job.get("status", "")).lower()
        if wanted_status and status not in wanted_status:
            continue
        provider = str(job.get("provider_id", "")).lower()
        if wanted_provider and provider not in wanted_provider:
            continue
        if needle:
            haystack = " ".join(
                str(job.get(key, ""))
                for key in (
                    "job_uuid",
                    "description",
                    "provider_id",
                    "job_env_name",
                    "local_dir",
                    "result_error",
                    "error",
                )
            ).lower()
            if needle not in haystack:
                continue
        selected.append(job)
    return selected


def print_jobs(
    jobs: list[dict[str, Any]],
    *,
    all_jobs: list[dict[str, Any]],
    base_url: str,
    limit: int,
    full: bool,
    show_sub_jobs: bool,
    use_color: bool,
    terminal_width: int,
    description_width: Optional[int],
    no_description: bool,
) -> None:
    if not jobs:
        print(f"No jobs found at {jobs_url(base_url, limit)}")
        if all_jobs:
            print(f"({len(all_jobs)} job(s) returned, all filtered out)")
        return

    columns = build_job_columns(
        full=full,
        show_description=not no_description,
        description_width=description_width,
        terminal_width=terminal_width,
    )
    for line in render_table(columns, jobs, use_color):
        print(line)

    counts = Counter(str(job.get("status") or "unknown") for job in jobs)
    summary = ", ".join(f"{status}: {count}" for status, count in counts.most_common())
    print()
    print(f"{len(jobs)} job(s) - {summary}")
    print(f"source: GET {jobs_url(base_url, limit)}")

    if show_sub_jobs:
        subs = sub_job_rows(jobs)
        print()
        if not subs:
            print("No sub-jobs.")
            return
        sub_columns = build_sub_job_columns(
            columns[-1].width if not no_description else 24
        )
        for line in render_table(sub_columns, subs, use_color):
            print(line)
        print()
        print(f"{len(subs)} sub-job(s)")


def describe_filters(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.status:
        parts.append("status=" + ",".join(args.status))
    if args.provider:
        parts.append("provider=" + ",".join(args.provider))
    if args.grep:
        parts.append(f"grep={args.grep!r}")
    return " ".join(parts)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Print the jobs registered in a running jobmanager (GET /jobs).",
        epilog=__doc__.split("Examples\n--------\n", 1)[-1],
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"jobmanager base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        metavar="N",
        help="maximum number of jobs to request (default: 200, 0 = server default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="HTTP timeout (default: 10)",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        metavar="STATUS",
        help="only show this job status; may be repeated",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="NAME",
        help="only show jobs from this provider; may be repeated",
    )
    parser.add_argument(
        "--grep",
        metavar="TEXT",
        help="only show jobs whose UUID, description, provider, env or path "
        "contains TEXT",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="show the full UUID, submit/finalize state and group count",
    )
    parser.add_argument(
        "--sub-jobs",
        action="store_true",
        help="also print one row per sub-job (remote id, status, error, ...)",
    )
    parser.add_argument(
        "--description-width",
        type=int,
        metavar="N",
        help="fixed width of the DESCRIPTION/ERROR column",
    )
    parser.add_argument(
        "--no-description",
        action="store_true",
        help="hide the DESCRIPTION column",
    )
    parser.add_argument(
        "--width",
        type=int,
        metavar="N",
        help="table width (default: current terminal width)",
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="refresh the table every SECONDS until Ctrl-C",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the raw JSON records instead of a table",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="disable status colors (also disabled when NO_COLOR is set)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        base_url = normalize_url(args.url)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    use_color = (
        not args.no_color
        and "NO_COLOR" not in os.environ
        and sys.stdout.isatty()
    )
    terminal_width = args.width or shutil.get_terminal_size(fallback=(120, 24)).columns

    while True:
        try:
            fetched = fetch_jobs(base_url, args.limit, args.timeout)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if exc.__cause__ and isinstance(exc.__cause__, urllib.error.URLError):
                print(
                    "hint: pass --url or set CATAGENT_JOBMANAGER_URL "
                    "(the server prints its host/port from start_job_manager.sh).",
                    file=sys.stderr,
                )
            if args.watch is None:
                return 1

        else:
            selected = filter_jobs(fetched, args.status, args.provider, args.grep)
            if args.watch is not None and sys.stdout.isatty():
                sys.stdout.write("\033[2J\033[H")

            if args.json:
                print(json.dumps(selected, indent=2, ensure_ascii=False))
            else:
                if args.watch is not None:
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    filters = describe_filters(args)
                    suffix = f"  [{filters}]" if filters else ""
                    print(f"{stamp}  {base_url}{suffix}")
                    print()
                print_jobs(
                    selected,
                    all_jobs=fetched,
                    base_url=base_url,
                    limit=args.limit,
                    full=args.full,
                    show_sub_jobs=args.sub_jobs,
                    use_color=use_color,
                    terminal_width=terminal_width,
                    description_width=args.description_width,
                    no_description=args.no_description,
                )

        if args.watch is None:
            return 0
        try:
            time.sleep(max(0.2, args.watch))
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
