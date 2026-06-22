import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "datasets: marks tests that download dataset files from Zenodo "
        "(deselect with '-m \"not datasets\"')",
    )
