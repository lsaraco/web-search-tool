"""Reusable web search client extracted from the quote agent."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import urlparse

import requests
from openai import OpenAI
from tavily import TavilyClient

logger = logging.getLogger(__name__)

SearchMethod = Literal["google_shopping", "google_search", "tavily"]


@dataclass(frozen=True)
class WebSearchConfig:
    """Configuration for search providers."""

    serpapi_api_key: str | None
    zyte_api_key: str | None
    openai_api_key: str | None
    tavily_api_key: str | None
    search_sites: list[str]
    llm_extraction_model: str
    default_method: SearchMethod

    @classmethod
    def from_env(cls) -> "WebSearchConfig":
        """Build config from environment variables."""
        sites = os.getenv("WEB_SEARCH_SITES", "supplyhouse.com")
        return cls(
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            zyte_api_key=os.getenv("ZYTE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            search_sites=[site.strip() for site in sites.split(",") if site.strip()],
            llm_extraction_model=os.getenv("WEB_SEARCH_LLM_EXTRACTION_MODEL", "gpt-4.1-mini"),
            default_method=os.getenv("WEB_SEARCH_DEFAULT_METHOD", "google_shopping"),
        )


def parse_lead_time(lead_time: str | None, now: datetime | None = None) -> int | None:
    """Parse a delivery string into days from today."""
    if not lead_time:
        return None

    now = now or datetime.now()
    text = lead_time.strip().lower()

    mm_dd_match = re.match(r"^(\d{1,2})-(\d{1,2})$", text)
    if mm_dd_match:
        month = int(mm_dd_match.group(1))
        day = int(mm_dd_match.group(2))
        try:
            target = datetime(now.year, month, day)
            if target < now:
                target = datetime(now.year + 1, month, day)
        except ValueError:
            return None
        days = max(1, (target - now).days)
        return days if days <= 120 else None

    if "tomorrow" in text or "next day" in text:
        return 1

    days_match = re.search(r"(\d+)[-\s]*(?:to[-\s]*)?(\d+)?\s*days?", text)
    if days_match:
        return int(days_match.group(2) or days_match.group(1))

    weeks_match = re.search(r"(\d+)\s*weeks?", text)
    if weeks_match:
        days = int(weeks_match.group(1)) * 7
        return days if days <= 120 else None

    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    date_match = re.search(
        r"(?:mon|tue|wed|thu|fri|sat|sun)?,?\s*([a-z]{3,9})\s+(\d{1,2})",
        text,
    )
    if date_match:
        month = months.get(date_match.group(1)[:3])
        if not month:
            return None
        day = int(date_match.group(2))
        try:
            target = datetime(now.year, month, day)
            if target < now:
                target = datetime(now.year + 1, month, day)
        except ValueError:
            return None
        days = max(1, (target - now).days)
        return days if days <= 120 else None

    return None


def get_link_domain(url: str) -> str:
    """Return the normalized domain for a URL or hostname."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        domain = parsed.netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except Exception:
        return ""


class WebSearchClient:
    """Search for product pricing through SerpAPI, Zyte, and Tavily."""

    def __init__(self, config: WebSearchConfig | None = None) -> None:
        self.config = config or WebSearchConfig.from_env()
        self.openai_client = OpenAI(api_key=self.config.openai_api_key) if self.config.openai_api_key else None
        self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key) if self.config.tavily_api_key else None

    def search(
        self,
        query: str | None = None,
        part_description: str | None = None,
        model: str | None = None,
        manufacturer: str | None = None,
        preferred_sites: list[str] | None = None,
        method: SearchMethod | None = None,
        limit: int = 3,
        preferred_only: bool = False,
        include_delivery_details: bool = True,
        enhanced_search: bool = False,
    ) -> list[dict[str, Any]]:
        """Search for products and return normalized result dictionaries."""
        search_query = query or self._build_product_query(part_description, model, manufacturer)
        if not search_query:
            return []

        method = method or self.config.default_method
        limit = max(1, min(limit, 10))

        if method == "google_shopping":
            return self.search_google_shopping(
                search_query,
                max_results=limit,
                preferred_sites=preferred_sites,
                preferred_only=preferred_only,
                include_delivery_details=include_delivery_details,
            )
        if method == "google_search":
            return self.search_google_search(
                search_query,
                preferred_sites=preferred_sites,
                enhanced_search=enhanced_search,
                limit=limit,
            )
        if method == "tavily":
            return self.search_tavily(search_query, max_results=limit)

        raise ValueError(f"Unsupported search method: {method}")

    def search_google_shopping(
        self,
        query: str,
        max_results: int = 3,
        preferred_sites: list[str] | None = None,
        preferred_only: bool = False,
        include_delivery_details: bool = True,
    ) -> list[dict[str, Any]]:
        """Search Google Shopping through SerpAPI and normalize product pricing."""
        if not self.config.serpapi_api_key:
            return [{"status": "error", "error": "SERPAPI_API_KEY is not set"}]

        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_shopping",
                "q": query,
                "location": "United States",
                "hl": "en",
                "gl": "us",
                "num": max_results,
                "api_key": self.config.serpapi_api_key,
            },
            timeout=30,
        )
        response.raise_for_status()

        raw_results = response.json().get("shopping_results", [])[:max_results]
        products = [self._shopping_item_to_result(item, query) for item in raw_results]

        if preferred_sites and preferred_only:
            products = [
                product
                for product in products
                if self._matches_preferred_site(product.get("source_url", ""), preferred_sites)
                or self._matches_preferred_site(product.get("preferred_site", ""), preferred_sites)
            ]

        if include_delivery_details:
            self._enrich_shopping_delivery(products, raw_results, preferred_sites, preferred_only)

        return products

    def search_google_search(
        self,
        query: str,
        preferred_sites: list[str] | None = None,
        enhanced_search: bool = False,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Search Google organic results and extract product data with Zyte."""
        sites = preferred_sites or self.config.search_sites
        results: list[dict[str, Any]] = []

        for site in sites:
            if len(results) >= limit:
                break
            candidates = self.search_google_for_product(
                query,
                site,
                num_results=5 if enhanced_search else min(limit * 2, 10),
            )
            if not candidates:
                continue
            if enhanced_search and isinstance(candidates, list):
                selected = self._select_best_url_with_llm(candidates, query)
                urls = [selected] if selected else []
            elif isinstance(candidates, str):
                urls = [candidates]
            else:
                urls = [item["url"] for item in candidates if item.get("url")]

            for url in urls:
                if len(results) >= limit:
                    break
                extracted = self.extract_price_with_zyte(url)
                if not extracted:
                    extracted = self.extract_price_with_zyte_robust(url)
                if not extracted:
                    continue
                results.append(
                    {
                        "status": "success",
                        "price": float(extracted["price"]),
                        "availability": extracted.get("availability"),
                        "lead_time_days": extracted.get("lead_time_days"),
                        "lead_time_raw": extracted.get("lead_time_raw"),
                        "source": "web",
                        "source_url": extracted.get("url") or url,
                        "search_terms": query,
                        "preferred_site": site,
                        "notes": f"Price extracted from {site}",
                        "confidence_score": 0.9,
                    }
                )

        return results

    def search_google_for_product(
        self,
        query: str,
        site: str,
        num_results: int = 1,
    ) -> str | list[dict[str, str]] | None:
        """Search Google through SerpAPI and return candidate product URLs."""
        if not self.config.serpapi_api_key:
            return None

        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": f"{query} site:{get_link_domain(site)}",
                "api_key": self.config.serpapi_api_key,
                "engine": "google",
                "num": num_results,
            },
            timeout=30,
        )
        response.raise_for_status()

        organic_results = response.json().get("organic_results", [])
        if not organic_results:
            return None
        if num_results == 1:
            return organic_results[0].get("link")

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in organic_results[:num_results]
        ]

    def extract_price_with_zyte(self, url: str) -> dict[str, Any] | None:
        """Extract product data from a product page using Zyte product extraction."""
        if not self.config.zyte_api_key:
            return None

        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(self.config.zyte_api_key, ""),
            json={
                "url": url,
                "product": True,
                "productOptions": {"extractFrom": "httpResponseBody", "ai": True},
                "customAttributes": {
                    "lead_time": {
                        "description": (
                            "Delivery date in MM-DD format. "
                            f"Current date is {datetime.now():%Y-%m-%d %H:%M}."
                        ),
                        "type": "string",
                    }
                },
                "followRedirect": True,
            },
            timeout=45,
        )
        if response.status_code == 520:
            return None
        response.raise_for_status()

        payload = response.json()
        product = payload.get("product") or {}
        price = product.get("price")
        if not price:
            return None

        lead_time_raw = (
            payload.get("customAttributes", {}).get("values", {}).get("lead_time")
        )
        return {
            "price": Decimal(str(price)),
            "availability": product.get("availability"),
            "lead_time_days": parse_lead_time(lead_time_raw),
            "lead_time_raw": lead_time_raw,
            "url": url,
        }

    def extract_price_with_zyte_robust(self, url: str) -> dict[str, Any] | None:
        """Extract product data from raw Zyte HTML using an OpenAI parser."""
        if not self.config.zyte_api_key or not self.openai_client:
            return None

        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(self.config.zyte_api_key, ""),
            json={"url": url, "httpResponseBody": True, "followRedirect": True},
            timeout=45,
        )
        if response.status_code == 520:
            return None
        response.raise_for_status()

        html_base64 = response.json().get("httpResponseBody")
        if not html_base64:
            return None
        html = base64.b64decode(html_base64).decode("utf-8", errors="ignore")
        filtered_html = self._filter_html_around_prices(html)

        completion = self.openai_client.beta.chat.completions.parse(
            model=self.config.llm_extraction_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract product price, availability, delivery lead time in MM-DD "
                        "format, and the best product URL from ecommerce HTML. "
                        "Use null for missing fields."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Source URL: {url}\n\nHTML:\n{filtered_html}",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "product_data",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "price": {"type": ["string", "null"]},
                            "availability": {"type": ["string", "null"]},
                            "lead_time": {"type": ["string", "null"]},
                            "url": {"type": ["string", "null"]},
                        },
                        "required": ["price", "availability", "lead_time", "url"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        parsed = json.loads(completion.choices[0].message.content)
        if not parsed.get("price"):
            return None

        return {
            "price": Decimal(parsed["price"]),
            "availability": parsed.get("availability"),
            "lead_time_days": parse_lead_time(parsed.get("lead_time")),
            "lead_time_raw": parsed.get("lead_time"),
            "url": parsed.get("url") or url,
        }

    def search_tavily(self, query: str, max_results: int = 3) -> list[dict[str, Any]]:
        """Search with Tavily and normalize the response."""
        if not self.tavily_client:
            return [{"status": "error", "error": "TAVILY_API_KEY is not set"}]

        response = self.tavily_client.search(query=query, max_results=max_results)
        return [
            {
                "status": "success",
                "title": item.get("title"),
                "source_url": item.get("url"),
                "content": item.get("content"),
                "confidence_score": item.get("score"),
                "search_terms": query,
                "source": "tavily",
            }
            for item in response.get("results", [])
        ]

    def _enrich_shopping_delivery(
        self,
        products: list[dict[str, Any]],
        raw_results: list[dict[str, Any]],
        preferred_sites: list[str] | None,
        preferred_only: bool,
    ) -> None:
        """Fetch immersive product data and use the LLM to pick vendor details."""
        if not self.openai_client:
            return

        batch: list[dict[str, Any]] = []
        for raw in raw_results:
            token = raw.get("immersive_product_page_token")
            if not token:
                continue
            try:
                response = requests.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google_immersive_product",
                        "page_token": token,
                        "api_key": self.config.serpapi_api_key,
                    },
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Could not fetch immersive product data: %s", exc)
                continue

            stores = response.json().get("product_results", {}).get("stores", [])[:6]
            if preferred_sites and preferred_only:
                stores = [
                    store
                    for store in stores
                    if self._matches_preferred_site(store.get("link", ""), preferred_sites)
                ]
            if stores:
                batch.append({"position": raw.get("position"), "stores_data": stores})

        if not batch:
            return

        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.config.llm_extraction_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "For each product position, choose the best in-stock vendor with "
                            "delivery information and lowest total price. Return one object per "
                            "position. Delivery time must be MM-DD or NOT FOUND."
                        ),
                    },
                    {"role": "user", "content": json.dumps(batch, indent=2)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "delivery_data",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "products": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "position": {"type": "integer"},
                                            "vendor_name": {"type": "string"},
                                            "price": {"type": "string"},
                                            "link": {"type": "string"},
                                            "delivery_cost": {"type": "string"},
                                            "delivery_time": {"type": "string"},
                                            "stock_status": {"type": "string"},
                                            "reason": {"type": "string"},
                                            "confidence_score": {
                                                "type": "number",
                                                "minimum": 0.0,
                                                "maximum": 1.0,
                                            },
                                        },
                                        "required": [
                                            "position",
                                            "vendor_name",
                                            "price",
                                            "link",
                                            "delivery_cost",
                                            "delivery_time",
                                            "stock_status",
                                            "reason",
                                            "confidence_score",
                                        ],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["products"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
        except Exception as exc:
            logger.warning("Could not extract delivery details with LLM: %s", exc)
            return

        vendors = {
            item["position"]: item
            for item in json.loads(completion.choices[0].message.content).get("products", [])
        }
        for product in products:
            vendor = vendors.get(product.get("position"))
            if not vendor:
                continue
            product["source_url"] = vendor.get("link") or product["source_url"]
            product["preferred_site"] = vendor.get("vendor_name") or product["preferred_site"]
            product["availability"] = vendor.get("stock_status")
            product["lead_time_raw"] = vendor.get("delivery_time")
            product["lead_time_days"] = parse_lead_time(vendor.get("delivery_time"))
            product["delivery_cost"] = self._parse_shipping_cost(vendor.get("delivery_cost"))
            product["confidence_score"] = vendor.get("confidence_score", 0.0)
            product["notes"] = vendor.get("reason", "")
            try:
                product["price"] = float(str(vendor.get("price", "")).replace("$", "").strip())
            except ValueError:
                pass

    def _select_best_url_with_llm(
        self,
        candidates: list[dict[str, str]],
        query: str,
    ) -> str | None:
        """Use OpenAI to select the best URL from search candidates."""
        if not self.openai_client:
            return candidates[0].get("url") if candidates else None

        completion = self.openai_client.beta.chat.completions.parse(
            model=self.config.llm_extraction_model,
            messages=[
                {
                    "role": "system",
                    "content": "Select the most likely ecommerce product page for the query.",
                },
                {"role": "user", "content": json.dumps({"query": query, "results": candidates})},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "selected_url",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_url": {"type": ["string", "null"]},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["selected_url", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        parsed = json.loads(completion.choices[0].message.content)
        return parsed.get("selected_url")

    @staticmethod
    def _build_product_query(
        part_description: str | None,
        model: str | None,
        manufacturer: str | None,
    ) -> str:
        """Build a product query from structured fields."""
        values = [
            "" if manufacturer == "generic" else manufacturer or "",
            "" if model == "generic" else model or "",
            part_description or "",
        ]
        return " ".join(value.strip() for value in values if value.strip())

    @staticmethod
    def _shopping_item_to_result(item: dict[str, Any], query: str) -> dict[str, Any]:
        """Convert a SerpAPI shopping item to the tool result schema."""
        delivery = item.get("delivery", "")
        delivery_cost: str | float = "NOT FOUND"
        if "free" in delivery.lower():
            delivery_cost = 0.0
        else:
            cost_match = re.search(r"\$(\d+(?:\.\d+)?)", delivery)
            if cost_match:
                delivery_cost = float(cost_match.group(1))

        return {
            "status": "success",
            "position": item.get("position"),
            "price": float(item.get("extracted_price") or 0.0),
            "availability": "NOT FOUND",
            "lead_time_days": None,
            "lead_time_raw": "NOT FOUND",
            "source": "web",
            "source_url": item.get("product_link") or item.get("link") or "NOT FOUND",
            "search_terms": query,
            "preferred_site": item.get("source", "NOT FOUND"),
            "notes": "Google Shopping result",
            "confidence_score": 0.0,
            "delivery_cost": delivery_cost,
            "product_title": item.get("title", "NOT FOUND"),
            "rating": item.get("rating"),
            "reviews": item.get("reviews"),
        }

    @staticmethod
    def _filter_html_around_prices(
        html: str,
        context_chars: int = 2000,
        max_chars: int = 100000,
    ) -> str:
        """Keep HTML chunks around prices to reduce LLM input size."""
        matches = list(re.finditer(r"\$[\s0-9,.]+", html))
        if not matches:
            return html[:max_chars]

        ranges = []
        for match in matches:
            midpoint = (match.start() + match.end()) // 2
            ranges.append((max(0, midpoint - context_chars), min(len(html), midpoint + context_chars)))

        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        chunks: list[str] = []
        remaining = max_chars
        for start, end in merged:
            if remaining <= 0:
                break
            chunk = html[start:end][:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)
        return "\n...\n".join(chunks)

    @staticmethod
    def _matches_preferred_site(value: str, preferred_sites: list[str]) -> bool:
        """Return whether a URL, domain, or vendor name matches preferred sites."""
        value_domain = get_link_domain(value)
        value_text = (value_domain or value).lower()
        return any(site.lower() in value_text or value_text in site.lower() for site in preferred_sites)

    @staticmethod
    def _parse_shipping_cost(value: str | None) -> str | float:
        """Convert shipping cost strings to floats where possible."""
        if not value or value == "NOT FOUND":
            return "NOT FOUND"
        if value.lower() == "free":
            return 0.0
        try:
            return float(value.replace("$", "").replace("+", "").strip())
        except ValueError:
            return "NOT FOUND"

