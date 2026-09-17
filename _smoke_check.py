"""Offline smoke check for VidSynth AI.

Run it before starting Streamlit to confirm the environment is sane:

    python _smoke_check.py

Only local checks are performed (no AI provider calls, no network
downloads), so it is safe to run in CI. It verifies:

1. every third-party module that ``app.py`` / ``supporting_functions.py``
   rely on can be imported;
2. every helper ``app.py`` imports from ``supporting_functions`` exists;
3. the API keys documented in the README are present.

Exit code 0 means all checks passed.
"""

import importlib
import os

from dotenv import load_dotenv

APP_MODULES = (
    "av",
    "chromadb",
    "dotenv",
    "faster_whisper",
    "fpdf",
    "langchain_chroma",
    "langchain_community",
    "langchain_core",
    "langchain_google_genai",
    "langchain_groq",
    "langchain_openai",
    "langchain_text_splitters",
    "sentence_transformers",
    "streamlit",
    "youtube_transcript_api",
    "yt_dlp",
)

APP_HELPERS = (
    "LLMQuotaError",
    "create_chunks",
    "create_vector_store",
    "extract_video_id",
    "generate_notes_and_topics",
    "generate_notes_from_transcript",
    "get_transcript",
    "rag_answer",
    "transcribe_video_audio",
    "translate_transcript",
)

REQUIRED_ENV_VARS = ("GOOGLE_API_KEY",)
OPTIONAL_ENV_VARS = ("GROQ_API_KEY", "OPENROUTER_API_KEY")


def check_imports() -> list[str]:
    """Return one failure message per module that cannot be imported."""
    failures = []
    for module_name in APP_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            failures.append(f"{module_name} is not importable ({error})")
    return failures


def check_helpers() -> list[str]:
    """Return one failure message per helper missing from the module."""
    try:
        supporting_functions = importlib.import_module("supporting_functions")
    except ImportError as error:
        return [f"supporting_functions is not importable ({error})"]

    return [
        f"supporting_functions.{name} is missing"
        for name in APP_HELPERS
        if not hasattr(supporting_functions, name)
    ]


def check_env() -> tuple[list[str], list[str]]:
    """Return the missing required and missing optional API keys."""
    load_dotenv()
    missing_required = [
        f"{name} is not set" for name in REQUIRED_ENV_VARS if not os.getenv(name)
    ]
    missing_optional = [
        f"{name} is not set" for name in OPTIONAL_ENV_VARS if not os.getenv(name)
    ]
    return missing_required, missing_optional


def main() -> int:
    """Run every check and return the process exit code."""
    env_failures, env_warnings = check_env()
    failures = check_imports() + check_helpers() + env_failures

    for message in env_warnings:
        print(f"WARN: {message} - the matching fallback provider stays disabled.")

    if failures:
        print(f"Smoke check failed with {len(failures)} problem(s):")
        for message in failures:
            print(f"  - {message}")
        return 1

    print("Smoke check passed: imports, helpers, and required API keys are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
