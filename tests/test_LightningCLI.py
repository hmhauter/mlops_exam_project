from src.main import cli_main
import pytest
from unittest.mock import patch
from io import StringIO
import sys

@patch("sys.argv", ["main.py", "-c", "config/main.yaml", "fit"])
def test_cli_main(self):
    captured_output = StringIO()
    sys.stdout = captured_output
    cli_main()
    sys.stdout = sys.__stdout__
    self.assertIn("Expected Output", captured_output.getvalue())


