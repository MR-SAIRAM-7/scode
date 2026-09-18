import os
import unittest
from unittest import mock

from scode.config import AppConfig, DEFAULT_MAX_OUTPUT_TOKENS


class ConfigTests(unittest.TestCase):
    def test_defaults_include_60k_output_tokens(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()
        self.assertEqual(config.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(config.max_output_tokens, 60000)

    def test_rejects_non_positive_max_tokens(self) -> None:
        with mock.patch.dict(os.environ, {"SCODE_MAX_OUTPUT_TOKENS": "0"}, clear=True):
            with self.assertRaises(ValueError):
                AppConfig.from_env()
