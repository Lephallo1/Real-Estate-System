from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lesotho_property_ai.data.live_scrapers import CreativePropertiesScraper, PropMarketScraper, UnsafeScraperURL


class _FakeResponse:
    def __init__(self, *, headers: dict[str, str] | None = None, content: bytes = b"", location: str = "") -> None:
        self.headers = headers or {}
        self.content = content
        self.text = content.decode("utf-8", "ignore")
        self.is_redirect = bool(location)
        self.is_permanent_redirect = False
        if location:
            self.headers["Location"] = location
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 65536):
        yield self.content

    def close(self) -> None:
        self.closed = True


class LiveScraperSecurityTests(unittest.TestCase):
    def _scraper(self) -> CreativePropertiesScraper:
        return CreativePropertiesScraper(image_root=Path(tempfile.mkdtemp()), max_items=1)

    def test_rejects_unknown_image_host(self) -> None:
        scraper = self._scraper()

        with self.assertRaises(UnsafeScraperURL):
            scraper._safe_absolute_url("https://evil.example/image.jpg", purpose="image")

    def test_rejects_private_network_resolution(self) -> None:
        scraper = self._scraper()
        private_address = (None, None, None, None, ("10.0.0.5", 443))

        with patch("lesotho_property_ai.data.live_scrapers.socket.getaddrinfo", return_value=[private_address]):
            with self.assertRaises(UnsafeScraperURL):
                scraper._safe_absolute_url("https://creativeproperties.co.ls/image.jpg", purpose="image")

    def test_rejects_unsafe_redirect_target(self) -> None:
        scraper = self._scraper()
        scraper.session.get = lambda *args, **kwargs: _FakeResponse(location="http://127.0.0.1/private.jpg")

        with patch("lesotho_property_ai.data.live_scrapers._public_ip_only", return_value=True):
            with self.assertRaises(UnsafeScraperURL):
                scraper._safe_get("https://creativeproperties.co.ls/image.jpg", purpose="image")

    def test_downloads_valid_allowlisted_image_without_relying_on_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = CreativePropertiesScraper(image_root=Path(tmpdir), max_items=1)
            scraper.session.get = lambda *args, **kwargs: _FakeResponse(
                headers={"Content-Type": "image/png"},
                content=b"fake-png-bytes",
            )

            with patch("lesotho_property_ai.data.live_scrapers._public_ip_only", return_value=True):
                paths = scraper.download_images(["https://creativeproperties.co.ls/uploads/property"], "safe-house")

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].endswith(".png"))
            self.assertTrue(Path(paths[0]).exists())

    def test_propmarket_allows_r2_images_only_for_image_fetches(self) -> None:
        scraper = PropMarketScraper(image_root=Path(tempfile.mkdtemp()), max_items=1)

        with patch("lesotho_property_ai.data.live_scrapers._public_ip_only", return_value=True):
            image_url = scraper._safe_absolute_url("https://cdn.example.r2.dev/property.webp", purpose="image")
            self.assertEqual(image_url, "https://cdn.example.r2.dev/property.webp")
            with self.assertRaises(UnsafeScraperURL):
                scraper._safe_absolute_url("https://cdn.example.r2.dev/sitemap.xml", purpose="page")


if __name__ == "__main__":
    unittest.main()
