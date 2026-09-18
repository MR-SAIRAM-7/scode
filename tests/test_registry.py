import unittest

from scode.config import AppConfig
from scode.providers.registry import get_provider


class ProviderRegistryTests(unittest.TestCase):
    def test_nvidia_provider(self) -> None:
        config = AppConfig(provider="nvidia", nvidia_api_key="token")
        provider = get_provider(config)
        self.assertEqual(provider.name, "nvidia")
        self.assertIn("nvidia", provider.base_url)

    def test_kimi_k3_provider(self) -> None:
        config = AppConfig(provider="kimi-k3", kimi_api_key="token")
        provider = get_provider(config)
        self.assertEqual(provider.name, "kimi-k3")
        self.assertIn("moonshot", provider.base_url)
