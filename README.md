# Web Search Tool

Reusable web search module for product lookup and agent tool use.

## Setup

Create a `.env` file in this folder. Values in this file take precedence over
existing system environment variables when the REPL starts.

Required for Google Shopping:

```env
SERPAPI_API_KEY=...
OPENAI_API_KEY=...
```

Optional:

```env
ZYTE_API_KEY=...
TAVILY_API_KEY=...
WEB_SEARCH_SITES=supplyhouse.com
WEB_SEARCH_DEFAULT_METHOD=google_shopping
WEB_SEARCH_LLM_EXTRACTION_MODEL=gpt-4.1-mini
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

## Search Methods

Set the default method with `WEB_SEARCH_DEFAULT_METHOD`, or pass `method` when
calling the LangChain tool.

| Method | What it does | Required env vars | Optional env vars |
|---|---|---|---|
| `google_shopping` | Uses SerpAPI Google Shopping to return product title, vendor, price, URL, rating, reviews, and optional delivery enrichment. | `SERPAPI_API_KEY` | `OPENAI_API_KEY` for delivery/vendor enrichment, `WEB_SEARCH_LLM_EXTRACTION_MODEL` |
| `google_search` | Uses SerpAPI Google Search to find product pages, then Zyte to extract product pricing from the page. | `SERPAPI_API_KEY`, `ZYTE_API_KEY` | `OPENAI_API_KEY` for robust HTML parsing and enhanced URL selection, `WEB_SEARCH_SITES`, `WEB_SEARCH_LLM_EXTRACTION_MODEL` |
| `tavily` | Uses Tavily general web search and returns normalized title, URL, content, and score results. | `TAVILY_API_KEY` | None |

## Run

```powershell
uv run python main.py
```

Then ask for product searches, for example:

```text
Find pricing for Littelfuse FLQ01-5 fuse on SupplyHouse.
```

## Screenshot

![Web search REPL showing a tool call and agent response](img/repl.png)

## Tests

Run all tests:

```powershell
uv run pytest -v -s
```

The test suite includes a real SerpAPI Google Shopping search for
`thermostat honeywell`. It requires `SERPAPI_API_KEY` in this folder's `.env`
or in the environment.
