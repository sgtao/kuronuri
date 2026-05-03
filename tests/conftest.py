"""Pytest configuration and shared fixtures for kuronuri."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-model",
        action="store_true",
        default=False,
        help="Run tests that load actual Hugging Face NER models (slow, requires network on first run).",  # noqa: E501
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model: mark test as requiring a real NER model (skipped unless --run-model is given).",  # noqa: E501
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-model"):
        return
    skip = pytest.mark.skip(reason="Skipped by default. Pass --run-model to run.")
    for item in items:
        if item.get_closest_marker("model"):
            item.add_marker(skip)
