"""Tests for pure web search helpers."""

from datetime import datetime

from web_search_tool.web_search import WebSearchClient, get_link_domain, parse_lead_time


def test_parse_lead_time_relative_days() -> None:
    """Parses day ranges using the maximum lead time."""
    assert parse_lead_time("Ships in 3-5 days") == 5


def test_parse_lead_time_mm_dd() -> None:
    """Parses MM-DD delivery dates against an explicit current date."""
    now = datetime(2026, 5, 26, 12, 0)
    assert parse_lead_time("05-28", now=now) == 1


def test_get_link_domain_normalizes_www() -> None:
    """Normalizes domains from URLs."""
    assert get_link_domain("https://www.supplyhouse.com/product/123") == "supplyhouse.com"


def test_build_product_query_ignores_generic_values() -> None:
    """Builds a clean structured product query."""
    query = WebSearchClient._build_product_query("fuse", "FLQ01-5", "generic")
    assert query == "FLQ01-5 fuse"

