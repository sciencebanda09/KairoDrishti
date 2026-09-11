"""
tests/conftest.py
==================
Shared pytest configuration for all KairoDrishti test suites.
Registers the --hil flag used by hardware-in-loop tests.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hil",
        action="store_true",
        default=False,
        help="Run HIL tests (requires real hardware)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "hil: mark test as requiring real hardware"
    )
