import os
import unittest
from unittest import mock

from scode.config import AppConfig, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_REASONING_EFFORT, DEFAULT_SEED, DEFAULT_TEMPERATURE


class ConfigTests(unittest.TestCase):
    def test_defaults_include_60k_output_tokens(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_env()
        self.assertEqual(config.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(config.max_output_tokens, 60000)
        self.assertEqual(config.temperature, DEFAULT_TEMPERATURE)
        self.assertEqual(config.seed, DEFAULT_SEED)
        self.assertEqual(config.reasoning_effort, DEFAULT_REASONING_EFFORT)

    def test_rejects_non_positive_max_tokens(self) -> None:
        with mock.patch.dict(os.environ, {"SCODE_MAX_OUTPUT_TOKENS": "0"}, clear=True):
            with self.assertRaises(ValueError):
                AppConfig.from_env()

    def test_rejects_negative_temperature(self) -> None:
        with mock.patch.dict(os.environ, {"SCODE_TEMPERATURE": "-1"}, clear=True):
            with self.assertRaises(ValueError):
                AppConfig.from_env()
