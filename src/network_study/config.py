from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import read_json


DEFAULT_BASE_URL = "https://www.moltbook.com/api/v1"


@dataclass
class SafetyConfig:
    timeout_seconds: int = 45
    max_retries: int = 6
    min_delay_ms: int = 800
    jitter_ms: int = 500
    backoff_base_seconds: float = 2.0
    cool_down_when_remaining_at_or_below: int = 1
    honor_rate_limit_headers: bool = True
    continue_on_error: bool = True


@dataclass
class MoltbookConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "MOLTBOOK_API_KEY"
    api_key: str = ""
    default_limit: int = 50
    max_pages: int = 20
    post_sorts: list[str] = field(default_factory=lambda: ["new", "hot", "top"])
    comment_sorts: list[str] = field(default_factory=lambda: ["new", "top"])
    comment_limit: int = 100
    comment_max_pages: int = 20
    hydrate_comments: bool = True
    hydrate_recent_posts: int = 500
    recrawl_offsets_hours: list[int] = field(default_factory=lambda: [1, 6, 24, 72, 168])
    submolt_query_param: str = "submolt"
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def resolved_api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") or self.api_key


@dataclass
class StudyConfig:
    database_path: Path = Path("data/study.sqlite")
    raw_dir: Path = Path("raw")
    reports_dir: Path = Path("reports")
    moltbook: MoltbookConfig = field(default_factory=MoltbookConfig)


def _safety_from_dict(data: dict[str, Any]) -> SafetyConfig:
    return SafetyConfig(
        timeout_seconds=int(data.get("timeout_seconds", data.get("requestTimeoutSec", 45))),
        max_retries=int(data.get("max_retries", data.get("maxRetries", 6))),
        min_delay_ms=int(data.get("min_delay_ms", data.get("minDelayMs", 800))),
        jitter_ms=int(data.get("jitter_ms", data.get("jitterMs", 500))),
        backoff_base_seconds=float(data.get("backoff_base_seconds", data.get("backoffBaseSeconds", 2.0))),
        cool_down_when_remaining_at_or_below=int(
            data.get(
                "cool_down_when_remaining_at_or_below",
                data.get("coolDownWhenRemainingAtOrBelow", 1),
            )
        ),
        honor_rate_limit_headers=bool(
            data.get("honor_rate_limit_headers", data.get("honorRateLimitHeaders", True))
        ),
        continue_on_error=bool(data.get("continue_on_error", data.get("continueOnError", True))),
    )


def _moltbook_from_dict(data: dict[str, Any]) -> MoltbookConfig:
    safety = _safety_from_dict(data.get("safety", {}))
    return MoltbookConfig(
        base_url=str(data.get("base_url", data.get("baseUrl", DEFAULT_BASE_URL))),
        api_key_env=str(data.get("api_key_env", "MOLTBOOK_API_KEY")),
        api_key=str(data.get("api_key", data.get("apiKey", ""))),
        default_limit=int(data.get("default_limit", data.get("defaultLimit", 50))),
        max_pages=int(data.get("max_pages", data.get("maxPages", 20))),
        post_sorts=list(data.get("post_sorts", data.get("postSorts", ["new", "hot", "top"]))),
        comment_sorts=list(data.get("comment_sorts", data.get("commentSorts", ["new", "top"]))),
        comment_limit=int(data.get("comment_limit", data.get("commentLimit", 100))),
        comment_max_pages=int(data.get("comment_max_pages", data.get("commentMaxPages", 20))),
        hydrate_comments=bool(data.get("hydrate_comments", data.get("hydrateComments", True))),
        hydrate_recent_posts=int(data.get("hydrate_recent_posts", data.get("hydrateRecentPosts", 500))),
        recrawl_offsets_hours=list(
            data.get("recrawl_offsets_hours", data.get("recrawlOffsetsHours", [1, 6, 24, 72, 168]))
        ),
        submolt_query_param=str(data.get("submolt_query_param", data.get("submoltQueryParam", "submolt"))),
        safety=safety,
    )


def load_config(path: Path | None) -> StudyConfig:
    if path is None:
        return StudyConfig()
    data = read_json(path)
    root = path.parent
    db = Path(data.get("database_path", data.get("databasePath", "data/study.sqlite")))
    raw = Path(data.get("raw_dir", data.get("rawDir", "raw")))
    reports = Path(data.get("reports_dir", data.get("reportsDir", "reports")))
    if not db.is_absolute():
        db = root / db
    if not raw.is_absolute():
        raw = root / raw
    if not reports.is_absolute():
        reports = root / reports
    return StudyConfig(
        database_path=db,
        raw_dir=raw,
        reports_dir=reports,
        moltbook=_moltbook_from_dict(data.get("moltbook", {})),
    )
