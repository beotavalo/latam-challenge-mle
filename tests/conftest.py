"""Shared pytest configuration for the test suite.

The provided model tests load the dataset with the relative path
``../data/data.csv``. That path only resolves when the current working
directory is a direct sub-directory of the repository root, whereas the
``make model-test`` / ``make api-test`` targets invoke pytest from the
repository root itself.

To keep the provided test files untouched while still allowing the suite to run
from the repository root, this autouse fixture switches the working directory to
the ``tests`` folder for the duration of each test (so ``../data`` points at
``<repo>/data``) and restores it afterwards — leaving coverage/report paths,
which are resolved at the end of the session, untouched.
"""

import os

import pytest

_TESTS_DIR = os.path.dirname(__file__)


@pytest.fixture(autouse=True)
def _chdir_to_tests_dir():
    previous_cwd = os.getcwd()
    os.chdir(_TESTS_DIR)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
