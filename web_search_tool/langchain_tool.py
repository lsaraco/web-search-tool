"""LangChain integration for the reusable web search client."""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from web_search_tool.web_search import WebSearchClient


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""

    query: str | None = Field(
        default=None,
        description="Full search query, such as 'Littelfuse FLQ01-5 fuse supplyhouse'.",
    )
    part_description: str | None = Field(
        default=None,
        description="Product or material description when using structured part fields.",
    )
    model: str | None = Field(default=None, description="Part number or model number.")
    manufacturer: str | None = Field(default=None, description="Manufacturer or brand.")
    preferred_sites: list[str] | None = Field(
        default=None,
        description="Preferred domains or vendor names, such as ['supplyhouse.com'].",
    )
    method: Literal["google_shopping", "google_search", "tavily"] = Field(
        default="google_shopping",
        description="Search backend to use.",
    )
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of results.")
    preferred_only: bool = Field(
        default=False,
        description="When supported, return only preferred vendor matches.",
    )
    include_delivery_details: bool = Field(
        default=True,
        description="Fetch and extract delivery, stock, and shipping details when possible.",
    )


def create_web_search_tool(client: WebSearchClient | None = None) -> StructuredTool:
    """Create a LangChain structured tool backed by WebSearchClient."""
    search_client = client or WebSearchClient()

    def run_search(
        query: str | None = None,
        part_description: str | None = None,
        model: str | None = None,
        manufacturer: str | None = None,
        preferred_sites: list[str] | None = None,
        method: str = "google_shopping",
        limit: int = 3,
        preferred_only: bool = False,
        include_delivery_details: bool = True,
    ) -> str:
        """Search the web for product pricing, availability, and delivery details."""
        results = search_client.search(
            query=query,
            part_description=part_description,
            model=model,
            manufacturer=manufacturer,
            preferred_sites=preferred_sites,
            method=method,
            limit=limit,
            preferred_only=preferred_only,
            include_delivery_details=include_delivery_details,
        )
        return json.dumps(results, indent=2)

    return StructuredTool.from_function(
        func=run_search,
        name="web_product_search",
        description=(
            "Search for product pricing, vendor source URLs, availability, delivery time, "
            "shipping cost, and ratings. Use for parts, materials, and ecommerce lookup."
        ),
        args_schema=WebSearchInput,
    )

