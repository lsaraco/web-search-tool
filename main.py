"""LangGraph ReAct REPL for the reusable web search tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from rich.console import Console
from rich.markup import escape

from web_search_tool import create_web_search_tool


SYSTEM_PROMPT = """You are a concise product search agent.

Use the web_product_search tool when the user asks for product pricing,
availability, delivery time, shipping cost, vendors, or source links.
Summarize the best results with price, vendor, URL, and delivery details.
If credentials are missing or a provider returns no results, state that clearly.
"""

console = Console()


def load_environment() -> None:
    """Load this project's .env with precedence over system variables."""
    current_dir = Path(__file__).resolve().parent
    load_dotenv(current_dir / ".env", override=True)


def build_agent():
    """Build a LangGraph ReAct agent with the web search tool."""
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    model = ChatOpenAI(model=model_name, temperature=0)
    tool = create_web_search_tool()

    try:
        return create_react_agent(model=model, tools=[tool], prompt=SYSTEM_PROMPT)
    except TypeError:
        return create_react_agent(model=model, tools=[tool], state_modifier=SYSTEM_PROMPT)


def print_response(result: dict) -> None:
    """Print the last assistant response from a graph result."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
            console.print(f"\n[bold green]Agent:[/bold green] {escape(str(message.content))}\n")
            return
    console.print("\n[bold green]Agent:[/bold green] No response returned.\n")


def iter_update_messages(update: dict[str, Any]):
    """Yield messages from a LangGraph stream update."""
    for value in update.values():
        if isinstance(value, dict):
            yield from value.get("messages", [])


def describe_tool_call(tool_call: dict[str, Any]) -> str:
    """Format a tool call for REPL logging."""
    args = tool_call.get("args", {})
    method = args.get("method", "google_shopping")
    query = args.get("query") or " ".join(
        str(args.get(field) or "")
        for field in ("manufacturer", "model", "part_description")
    ).strip()
    preferred_sites = args.get("preferred_sites")
    preferred_text = f", preferred_sites={preferred_sites}" if preferred_sites else ""
    return f"Tool call: {tool_call.get('name')} using {method}, query='{query}'{preferred_text}"


def run_agent_turn(agent, user_input: str) -> None:
    """Run one REPL turn and print tool activity as it happens."""
    final_result: dict[str, Any] | None = None
    printed_tool_call_ids: set[str] = set()
    printed_tool_result_ids: set[str] = set()

    for update in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="updates",
    ):
        final_result = update
        for message in iter_update_messages(update):
            for tool_call in getattr(message, "tool_calls", []) or []:
                tool_call_id = tool_call.get("id") or repr(tool_call)
                if tool_call_id in printed_tool_call_ids:
                    continue
                printed_tool_call_ids.add(tool_call_id)
                console.print(f"\n[bold yellow]{escape(describe_tool_call(tool_call))}[/bold yellow]")

            if isinstance(message, ToolMessage) or getattr(message, "type", None) == "tool":
                tool_call_id = getattr(message, "tool_call_id", None) or repr(message)
                if tool_call_id in printed_tool_result_ids:
                    continue
                printed_tool_result_ids.add(tool_call_id)
                console.print("[dim yellow]Tool result received.[/dim yellow]")

    if final_result:
        print_response_from_stream_update(final_result)
    else:
        console.print("\n[bold green]Agent:[/bold green] No response returned.\n")


def print_response_from_stream_update(update: dict[str, Any]) -> None:
    """Print the final assistant response from a LangGraph stream update."""
    for message in reversed(list(iter_update_messages(update))):
        if isinstance(message, AIMessage) or getattr(message, "type", None) == "ai":
            console.print(f"\n[bold green]Agent:[/bold green] {escape(str(message.content))}\n")
            return
    console.print("\n[bold green]Agent:[/bold green] No response returned.\n")


def main() -> None:
    """Run the interactive REPL."""
    load_environment()
    agent = build_agent()

    console.print(
        "[bold]Web search ReAct agent.[/bold] "
        "Type [cyan]exit[/cyan], [cyan]quit[/cyan], or Ctrl+C to stop."
    )
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        run_agent_turn(agent, user_input)


if __name__ == "__main__":
    main()
