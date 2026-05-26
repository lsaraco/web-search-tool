"""Functional tests for SerpAPI web search."""

from pathlib import Path

from dotenv import load_dotenv

from web_search_tool import WebSearchClient


def load_test_environment() -> None:
    """Load this project's .env with precedence over system variables."""
    project_dir = Path(__file__).resolve().parents[1]
    load_dotenv(project_dir / ".env", override=True)


def test_serpapi_search_thermostat_honeywell() -> None:
    """Search SerpAPI for Honeywell thermostats and return product information."""
    load_test_environment()

    client = WebSearchClient()
    results = client.search(
        query="thermostat honeywell",
        method="google_shopping",
        limit=3,
        include_delivery_details=False,
    )

    print("\nSerpAPI results for 'thermostat honeywell':")
    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"  Title: {result.get('product_title')}")
        print(f"  Vendor: {result.get('preferred_site')}")
        print(f"  Price: {result.get('price')}")
        print(f"  URL: {result.get('source_url')}")
        print(f"  Delivery cost: {result.get('delivery_cost')}")

    assert results, "SerpAPI should return at least one result"
    assert all(result["status"] == "success" for result in results)
    assert any("honeywell" in result.get("product_title", "").lower() for result in results)
    assert any(float(result.get("price") or 0) > 0 for result in results)
    assert any(result.get("source_url") and result["source_url"] != "NOT FOUND" for result in results)
