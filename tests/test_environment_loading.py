"""Tests for environment loading precedence."""

import os
from pathlib import Path

from dotenv import load_dotenv


def test_project_dotenv_overrides_existing_environment(tmp_path) -> None:
    """Project .env values should override already-set system variables."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from_project_dotenv\n", encoding="utf-8")

    original_value = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "from_system_environment"
    try:
        load_dotenv(Path(env_file), override=True)
        assert os.environ["OPENAI_API_KEY"] == "from_project_dotenv"
    finally:
        if original_value is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original_value

