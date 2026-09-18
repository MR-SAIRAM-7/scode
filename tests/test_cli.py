import os
import unittest
from unittest import mock

from scode.cli import main


class CLITests(unittest.TestCase):
    def test_missing_provider_key_returns_clean_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                main(["--provider", "nvidia", "hello"])

        self.assertIn("NVIDIA_API_KEY", str(ctx.exception))
