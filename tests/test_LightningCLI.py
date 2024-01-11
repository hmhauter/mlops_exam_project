from pathlib import Path
from src.main import cli_main
import pytest
from unittest.mock import patch
from io import StringIO
import sys

@patch("sys.argv", ["main.py", "-c", "config/test.yaml", "fit"])
def test_cli_main():
    print("!!!!!!!!!!!!!!!")
    captured_output = StringIO()
    sys.stdout = captured_output
    cli_main()
    sys.stdout = sys.__stdout__
    print(captured_output)
    assert Path("config.yaml").exists()


