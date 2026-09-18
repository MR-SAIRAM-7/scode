import os
import unittest
from unittest import mock

from scode.cli import _build_direct_messages, main


class CLITests(unittest.TestCase):
    def test_missing_provider_key_returns_clean_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                main(["--provider", "nvidia", "hello"])

        self.assertIn("NVIDIA_API_KEY", str(ctx.exception))

    def test_build_direct_messages_with_image(self) -> None:
        messages = _build_direct_messages("What is in this image?", "https://example.com/image.jpg")
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
