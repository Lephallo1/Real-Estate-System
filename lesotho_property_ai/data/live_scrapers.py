from __future__ import annotations

import json
import ipaddress
import re
import socket
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .cleaning import clean_property_dataframe, normalize_district
from .scraper_adapter import ScraperAdapter
from ..text_utils import strip_html_text


AVAILABLE_LIVE_SOURCES = (
    "creativeproperties",
    "propmarket",
    "sotholand",
    "lesothohousing",
    "mestech",
    "mosoholdings",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
)

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REDIRECTS = 3

SOURCE_HOST_ALLOWLIST = {
    "creativeproperties": {
        "page": ("creativeproperties.co.ls", "www.creativeproperties.co.ls"),
        "image": ("creativeproperties.co.ls", "www.creativeproperties.co.ls"),
    },
    "propmarket": {
        "page": ("propmarket.co.ls", "www.propmarket.co.ls"),
        "image": ("propmarket.co.ls", "www.propmarket.co.ls", ".r2.dev"),
    },
    "sotholand": {
        "page": ("sotholandproperties.co.ls", "www.sotholandproperties.co.ls"),
        "image": ("sotholandproperties.co.ls", "www.sotholandproperties.co.ls"),
    },
    "lesothohousing": {
        "page": ("lesothohousing.org.ls", "www.lesothohousing.org.ls"),
        "image": ("lesothohousing.org.ls", "www.lesothohousing.org.ls"),
    },
    "mestech": {
        "page": ("mestech.co.ls", "www.mestech.co.ls"),
        "image": ("mestech.co.ls", "www.mestech.co.ls"),
    },
    "mosoholdings": {
        "page": ("mosoholdings.co.ls", "www.mosoholdings.co.ls"),
        "image": ("mosoholdings.co.ls", "www.mosoholdings.co.ls"),
    },
}

IMAGE_CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

AMENITY_KEYWORDS = {
    "water": "water connection",
    "electricity": "electricity",
    "parking": "parking",
    "garage": "garage",
    "garden": "garden",
    "yard": "yard",
    "school": "near schools",
    "schools": "near schools",
    "road": "road access",
    "tar": "tar road",
    "office": "office-ready",
    "security": "security",
    "furnished": "furnished",
    "bathtub": "bathtub",
    "shower": "shower",
    "wall": "walled property",
}

PROPERTY_TYPE_RULES = (
    ("apartment", "Apartment"),
    ("bachelor", "Apartment"),
    ("flat", "Apartment"),
    ("townhouse", "Townhouse"),
    ("duplex", "Townhouse"),
    ("house", "House"),
    ("home", "House"),
    ("cottage", "Cottage"),
    ("site", "Site"),
    ("land", "Site"),
    ("plot", "Site"),
    ("shop", "Commercial"),
    ("office", "Commercial"),
    ("commercial", "Commercial"),
)


class UnsafeScraperURL(ValueError):
    """Raised when a scraped URL points outside the safe fetch policy."""


def _normalize_host(hostname: str | None) -> str:
    if not hostname:
        raise UnsafeScraperURL("URL is missing a hostname")
    cleaned = hostname.strip().rstrip(".").lower()
    try:
        return cleaned.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeScraperURL("URL hostname is not valid") from exc


def _host_allowed(hostname: str, allowed_hosts: tuple[str, ...]) -> bool:
    for allowed in allowed_hosts:
        if allowed.startswith("."):
            suffix = allowed.lower()
            if hostname.endswith(suffix) and hostname != suffix.lstrip("."):
                return True
        elif hostname == allowed.lower():
            return True
    return False


def _public_ip_only(hostname: str) -> bool:
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            resolved = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise UnsafeScraperURL(f"Could not resolve host {hostname}") from exc
        addresses = []
        for item in resolved:
            address = item[4][0]
            try:
                addresses.append(ipaddress.ip_address(address))
            except ValueError:
                continue

    if not addresses:
        raise UnsafeScraperURL(f"Could not resolve host {hostname}")

    for address in addresses:
        if not address.is_global:
            raise UnsafeScraperURL(f"Blocked non-public network address for {hostname}")
    return True


def _safe_image_suffix(url: str, content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in IMAGE_CONTENT_TYPE_SUFFIXES:
        return IMAGE_CONTENT_TYPE_SUFFIXES[media_type]
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"} else ".jpg"


def scrape_live_properties(
    image_root: Path,
    sources: Iterable[str] | None = None,
    per_source_limit: int = 2,
    include_rentals: bool = False,
    max_images_per_property: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_records, report = collect_live_property_records(
        image_root=image_root,
        sources=sources,
        per_source_limit=per_source_limit,
        include_rentals=include_rentals,
        max_images_per_property=max_images_per_property,
    )
    combined = clean_property_dataframe(raw_records)
    report["total_records"] = int(len(combined))
    return combined, report


def collect_live_property_records(
    image_root: Path,
    sources: Iterable[str] | None = None,
    per_source_limit: int = 2,
    include_rentals: bool = False,
    max_images_per_property: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected = list(sources or AVAILABLE_LIVE_SOURCES)
    invalid = [source for source in selected if source not in AVAILABLE_LIVE_SOURCES]
    if invalid:
        raise ValueError(f"Unsupported live sources: {', '.join(invalid)}")

    source_map = {
        "creativeproperties": CreativePropertiesScraper,
        "propmarket": PropMarketScraper,
        "sotholand": SotholandScraper,
        "lesothohousing": LesothoHousingScraper,
        "mestech": MestechScraper,
        "mosoholdings": MosoHoldingsScraper,
    }
    frames: list[pd.DataFrame] = []
    report: dict[str, object] = {"sources": {}, "total_records": 0}

    for source in selected:
        scraper = source_map[source](
            image_root=image_root,
            max_items=per_source_limit,
            include_rentals=include_rentals,
            max_images_per_property=max_images_per_property,
        )
        try:
            dataframe = pd.DataFrame(scraper.fetch_raw_records())
            frames.append(dataframe)
            report["sources"][source] = {
                "records": int(len(dataframe)),
                "listing_page": scraper.listing_reference,
                "unsafe_url_skips": scraper.unsafe_url_skips,
            }
        except Exception as exc:
            report["sources"][source] = {
                "records": 0,
                "listing_page": scraper.listing_reference,
                "unsafe_url_skips": scraper.unsafe_url_skips,
                "error": str(exc),
            }

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    report["total_records"] = int(len(combined))
    return combined, report


class WebScraperAdapter(ScraperAdapter):
    listing_reference: str = ""

    def __init__(
        self,
        image_root: Path,
        max_items: int = 2,
        include_rentals: bool = False,
        max_images_per_property: int = 2,
    ) -> None:
        self.image_root = Path(image_root)
        self.max_items = max_items if max_items and max_items > 0 else 1_000_000
        self.include_rentals = include_rentals
        self.max_images_per_property = max_images_per_property
        self.unsafe_url_skips = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _safe_absolute_url(self, url: str, *, purpose: str, base_url: str | None = None) -> str:
        absolute_url = urljoin(base_url or self.listing_reference, str(url or "").strip())
        parsed = urlparse(absolute_url)
        if parsed.scheme not in {"http", "https"}:
            raise UnsafeScraperURL("Only HTTP and HTTPS URLs are supported")
        if parsed.username or parsed.password:
            raise UnsafeScraperURL("URL userinfo is not allowed")
        if parsed.port and parsed.port not in {80, 443}:
            raise UnsafeScraperURL("Only standard HTTP/HTTPS ports are allowed")

        hostname = _normalize_host(parsed.hostname)
        allowed = SOURCE_HOST_ALLOWLIST.get(self.source_name, {}).get(purpose, ())
        if not allowed or not _host_allowed(hostname, allowed):
            raise UnsafeScraperURL(f"Host {hostname} is not allowed for {self.source_name} {purpose} fetches")
        _public_ip_only(hostname)
        return absolute_url

    def _safe_get(self, url: str, *, purpose: str, stream: bool = False) -> requests.Response:
        next_url = self._safe_absolute_url(url, purpose=purpose)
        for _ in range(MAX_REDIRECTS + 1):
            response = self.session.get(next_url, timeout=30, allow_redirects=False, stream=stream)
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location", "")
                response.close()
                if not location:
                    raise UnsafeScraperURL("Redirect response did not include a Location header")
                next_url = self._safe_absolute_url(location, purpose=purpose, base_url=next_url)
                continue
            response.raise_for_status()
            return response
        raise UnsafeScraperURL("Too many redirects while fetching scraped URL")

    def _mark_unsafe_url(self) -> None:
        self.unsafe_url_skips += 1

    def get_soup(self, url: str) -> BeautifulSoup:
        response = self._safe_get(url, purpose="page")
        time.sleep(0.05)
        return BeautifulSoup(response.text, "lxml")

    def get_text(self, url: str) -> str:
        response = self._safe_get(url, purpose="page")
        time.sleep(0.05)
        return response.text

    def download_images(self, image_urls: list[str], property_id: str) -> list[str]:
        local_paths: list[str] = []
        property_dir = self.image_root / "live" / self.source_name / property_id
        property_dir.mkdir(parents=True, exist_ok=True)

        filtered = self._filter_image_urls(image_urls)[: self.max_images_per_property]
        for index, url in enumerate(filtered, start=1):
            try:
                safe_url = self._safe_absolute_url(url, purpose="image")
            except UnsafeScraperURL:
                self._mark_unsafe_url()
                continue

            try:
                response = self._safe_get(safe_url, purpose="image", stream=True)
                content_type = response.headers.get("Content-Type", "")
                if not content_type.lower().split(";", 1)[0].startswith("image/"):
                    response.close()
                    continue

                suffix = _safe_image_suffix(safe_url, content_type)
                destination = property_dir / f"image_{index}{suffix}"
                if not destination.exists():
                    total = 0
                    chunks: list[bytes] = []
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise UnsafeScraperURL("Image response exceeded the allowed size")
                        chunks.append(chunk)
                    destination.write_bytes(b"".join(chunks))
                    time.sleep(0.05)
                response.close()
            except UnsafeScraperURL:
                self._mark_unsafe_url()
                continue
            except requests.RequestException:
                continue

            if destination.exists():
                local_paths.append(str(destination))
        return local_paths

    @staticmethod
    def parse_price(value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"([0-9][0-9,]*\.?[0-9]*)", value.replace(" ", ""))
        if not match:
            return None
        numeric = match.group(1).replace(",", "")
        try:
            return int(float(numeric))
        except ValueError:
            return None

    @staticmethod
    def parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        match = re.search(r"\d+", str(value))
        return int(match.group()) if match else None

    @staticmethod
    def clean_text(value: str | None) -> str:
        return strip_html_text(value)

    @staticmethod
    def decode_embedded_string(raw_value: str | None) -> str:
        if raw_value is None:
            return ""
        try:
            return json.loads(f"\"{raw_value}\"")
        except json.JSONDecodeError:
            return raw_value.encode("utf-8", "ignore").decode("unicode_escape", "ignore")

    @staticmethod
    def _extract_regex(pattern: str, text: str, flags: int = 0) -> str | None:
        match = re.search(pattern, text, flags)
        return match.group(1) if match else None

    @staticmethod
    def infer_property_type(*values: str) -> str:
        haystack = " ".join(values).lower()
        for needle, normalized in PROPERTY_TYPE_RULES:
            if needle in haystack:
                return normalized
        return "House"

    @staticmethod
    def infer_condition(*values: str) -> str:
        haystack = " ".join(values).lower()
        if any(keyword in haystack for keyword in ("renovation", "fixer", "unfinished", "needs work")):
            return "Renovation Needed"
        if any(keyword in haystack for keyword in ("new", "modern", "brand new", "newly renovated", "highly finished")):
            return "New"
        return "Good"

    @staticmethod
    def infer_style(*values: str) -> str:
        haystack = " ".join(values).lower()
        if any(keyword in haystack for keyword in ("modern", "new units", "highly finished")):
            return "Modern"
        if any(keyword in haystack for keyword in ("family", "yard", "garden", "spacious")):
            return "Family"
        if any(keyword in haystack for keyword in ("office", "commercial", "shop")):
            return "Contemporary"
        return "Traditional"

    @staticmethod
    def infer_environment(*values: str) -> str:
        haystack = " ".join(values).lower()
        if any(keyword in haystack for keyword in ("garden", "yard", "trees")):
            return "Garden"
        if any(keyword in haystack for keyword in ("hill", "view", "mountain")):
            return "Hillside"
        if any(keyword in haystack for keyword in ("town", "city", "central", "tar road", "school", "shop")):
            return "Urban"
        return "Suburban"

    @staticmethod
    def infer_amenities(*values: str) -> list[str]:
        haystack = " ".join(values).lower()
        amenities = []
        for needle, label in AMENITY_KEYWORDS.items():
            if needle in haystack and label not in amenities:
                amenities.append(label)
        return amenities

    @staticmethod
    def infer_district(*values: str) -> str:
        haystack = " ".join(values)
        for district in (
            "Maseru",
            "Leribe",
            "Berea",
            "Mafeteng",
            "Mohale's Hoek",
            "Quthing",
            "Butha-Buthe",
            "Mokhotlong",
            "Qacha's Nek",
            "Thaba-Tseka",
        ):
            if district.lower() in haystack.lower():
                return normalize_district(district)
        return normalize_district(haystack.split(",")[-1] if haystack else "Maseru")

    @staticmethod
    def parse_labeled_text(text: str, labels: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        normalized = WebScraperAdapter.clean_text(text)
        if not normalized:
            return result
        pattern = re.compile("(" + "|".join(re.escape(label) for label in labels) + r")\s*:")
        matches = list(pattern.finditer(normalized))
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            result[match.group(1)] = normalized[start:end].strip(" :|")
        return result

    @staticmethod
    def build_sesotho_summary(
        title: str,
        property_type: str,
        location_text: str,
        bedrooms: int,
        bathrooms: int,
        price: int | None,
    ) -> str:
        bedroom_text = f"dikamore tse {bedrooms}" if bedrooms else "sebaka sena"
        bathroom_text = f"le dibate tse {bathrooms}" if bathrooms else ""
        price_text = f"ka theko ya M {price:,}" if price else "ka theko e sa hlaloswang"
        return (
            f"Thepa ena ya mofuta wa {property_type.lower()} e fumaneha {location_text}. "
            f"E na le {bedroom_text} {bathroom_text} mme e fumaneha {price_text}. "
            f"Tlotla ena e tshwanetse ho shejwa bakeng sa bareki ba batlang menyetla ya nnete Lesotho."
        ).replace("  ", " ").strip()

    def _filter_image_urls(self, urls: list[str]) -> list[str]:
        cleaned: list[str] = []
        for url in urls:
            lower = url.lower()
            try:
                safe_url = self._safe_absolute_url(url, purpose="image")
            except UnsafeScraperURL:
                self._mark_unsafe_url()
                continue
            if any(token in lower for token in ("logo", "icon", "favicon", "favi", "avatar")):
                continue
            if safe_url not in cleaned:
                cleaned.append(safe_url)
        preferred = [
            url for url in cleaned if not re.search(r"-\d+x\d+\.(jpg|jpeg|png|webp)(?:\?|$)", url, re.I)
        ]
        return preferred or cleaned


class CreativePropertiesScraper(WebScraperAdapter):
    source_name = "creativeproperties"
    listing_reference = "https://creativeproperties.co.ls/property-list-lesotho/"

    def fetch_raw_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        seen_urls: set[str] = set()
        page_urls = [self.listing_reference]
        first_page = self.get_soup(self.listing_reference)
        page_urls.extend(
            link["href"]
            for link in first_page.select('a[href*="/property-list-lesotho/page/"]')
            if link.get("href")
        )

        for page_url in dict.fromkeys(page_urls):
            soup = first_page if page_url == self.listing_reference else self.get_soup(page_url)
            for item in soup.select("div.property_listing"):
                try:
                    record = self._parse_card(item)
                except Exception:
                    continue
                if not record:
                    continue
                if record["listing_url"] in seen_urls:
                    continue
                if not self.include_rentals and record["listing_intent"] != "sale":
                    continue
                records.append(record)
                seen_urls.add(str(record["listing_url"]))
                if len(records) >= self.max_items:
                    return records
        return records

    def _parse_card(self, item) -> dict[str, object] | None:
        title_node = item.select_one("h3.listing_title")
        link_node = item.select_one('a[href*="/properties/"]')
        price_node = item.select_one("span.property_price")
        if not title_node or not link_node:
            return None

        card_text = self.clean_text(item.get_text(" ", strip=True))
        meta_text = self.clean_text(
            " ".join(node.get_text(" ", strip=True) for node in item.select(".article_property_type, .property_listing_details"))
        )
        bed_bath = re.search(r"(\d+)\s+(\d+)", meta_text)
        segments = [segment.strip(" ,") for segment in meta_text.split("|")]
        location_hint = segments[1] if len(segments) > 1 else ""
        raw_type = segments[2] if len(segments) > 2 else ""
        raw_type = re.sub(r"\s+in\s+.*$", "", raw_type, flags=re.I).strip()

        detail = self._parse_detail(link_node["href"])
        title = self.clean_text(title_node.get_text(" ", strip=True))
        price_text = detail.get("Price") or self.clean_text(price_node.get_text(" ", strip=True) if price_node else "")
        location_text = self.clean_text(
            ", ".join(part for part in (detail.get("Address"), detail.get("City"), detail.get("Area")) if part)
        ) or location_hint or "Maseru"
        district = self.infer_district(detail.get("City", ""), location_text, location_hint)
        property_type = self.infer_property_type(raw_type, title, detail.get("Property Type", ""))
        description_en = detail.get("description_en") or title
        price = self.parse_price(price_text)
        bedrooms = self.parse_int(detail.get("Bedrooms")) or self.parse_int(bed_bath.group(1) if bed_bath else None) or 0
        bathrooms = self.parse_int(detail.get("Bathrooms")) or self.parse_int(bed_bath.group(2) if bed_bath else None) or 0
        images = self.download_images(detail.get("image_urls", []), detail["property_id"])
        combined_text = " ".join([title, description_en, location_text, raw_type, card_text])
        listing_intent = (
            "rent"
            if any(token in f"{title} {price_text} {meta_text}".lower() for token in ("per month", "rental", "to let"))
            else "sale"
        )

        return {
            "property_id": detail["property_id"],
            "source": self.source_name,
            "title": title,
            "description_en": description_en,
            "description_st": self.build_sesotho_summary(title, property_type, location_text, bedrooms, bathrooms, price),
            "price": price,
            "currency": "LSL",
            "district": district,
            "location_text": location_text,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "image_paths": images,
            "listing_url": link_node["href"],
            "condition": self.infer_condition(title, description_en, price_text),
            "style": self.infer_style(title, description_en, raw_type),
            "environment": self.infer_environment(location_text, description_en),
            "amenities": self.infer_amenities(description_en, " ".join(detail.get("feature_text", []))),
            "listing_intent": listing_intent,
        }

    def _parse_detail(self, url: str) -> dict[str, object]:
        soup = self.get_soup(url)
        title = self.clean_text((soup.find("h1") or soup.title).get_text(" ", strip=True))
        description_block = soup.select_one(".property_description")
        description_text = self.clean_text(description_block.get_text(" ", strip=True) if description_block else title)
        description_text = re.sub(r"^Property Description\s*", "", description_text, flags=re.I)

        label_map: dict[str, str] = {}
        feature_text: list[str] = []
        labels = [
            "Price",
            "Address",
            "City",
            "Area",
            "County",
            "State",
            "Zip",
            "Country",
            "Bedrooms",
            "Bathrooms",
            "Lounge",
            "Kitchen",
        ]
        for block in soup.select("div.prop_details, div.prop_details_custom"):
            block_text = self.clean_text(block.get_text(" ", strip=True))
            if ":" in block_text:
                label_map.update(self.parse_labeled_text(block_text, labels))
            else:
                feature_text.append(block_text)

        candidate_urls = []
        for tag in soup.find_all(["img", "a"]):
            candidate = tag.get("src") or tag.get("href")
            if candidate and "wp-content/uploads" in candidate:
                candidate_urls.append(candidate)

        slug = url.rstrip("/").split("/")[-1]
        return {
            "property_id": f"creative-{slug}",
            "description_en": description_text,
            "image_urls": candidate_urls,
            "feature_text": feature_text,
            **label_map,
        }


class LesothoHousingScraper(WebScraperAdapter):
    source_name = "lesothohousing"
    listing_reference = "https://lesothohousing.org.ls/wp-sitemap-posts-property-1.xml"

    def fetch_raw_records(self) -> list[dict[str, object]]:
        urls = self._property_urls_from_sitemap(self.listing_reference)
        records: list[dict[str, object]] = []
        for url in urls:
            try:
                record = self._parse_detail(url)
            except Exception:
                continue
            if not record:
                continue
            if not self.include_rentals and record["listing_intent"] not in {"sale", "available"}:
                continue
            records.append(record)
            if len(records) >= self.max_items:
                break
        return records

    def _property_urls_from_sitemap(self, sitemap_url: str) -> list[str]:
        text = self.get_text(sitemap_url)
        root = ET.fromstring(text)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        return [node.text.strip() for node in root.findall(".//sm:loc", namespace) if node.text]

    def _parse_detail(self, url: str) -> dict[str, object] | None:
        soup = self.get_soup(url)
        title_node = soup.select_one(".page-title") or soup.find("h1")
        title = self.clean_text(title_node.get_text(" ", strip=True) if title_node else url.rstrip("/").split("/")[-1])
        status = self.clean_text((soup.select_one(".label-status") or soup.find("a", class_=re.compile("label-status"))).get_text(" ", strip=True) if soup.select_one(".label-status") else "")
        status_key = "sale" if "sell" in status.lower() else "available" if "available" in status.lower() else "other"

        description_node = soup.select_one(".property-description-content")
        description_en = self.clean_text(description_node.get_text(" ", strip=True) if description_node else title)

        detail_node = soup.select_one(".detail-wrap")
        detail_map = self.parse_labeled_text(
            self.clean_text(detail_node.get_text(" ", strip=True) if detail_node else ""),
            ["Property Type", "Property Status", "Area Size"],
        )
        address_block = soup.select_one(".property-address-wrap")
        address_map = self.parse_labeled_text(
            self.clean_text(address_block.get_text(" ", strip=True) if address_block else ""),
            ["Address", "City", "Zip/Postal Code", "Area", "Country"],
        )

        json_ld = {}
        script = soup.find("script", attrs={"type": "application/ld+json"})
        if script:
            try:
                json_ld = json.loads(script.get_text())
            except json.JSONDecodeError:
                json_ld = {}

        image_urls = []
        if isinstance(json_ld.get("image"), list):
            image_urls.extend(str(item) for item in json_ld["image"])
        location_text = self.clean_text(
            ", ".join(
                part for part in (
                    address_map.get("Address"),
                    address_map.get("City"),
                    address_map.get("Area"),
                    json_ld.get("address", {}).get("addressLocality") if isinstance(json_ld.get("address"), dict) else "",
                ) if part
            )
        )
        district = self.infer_district(location_text, json.dumps(json_ld.get("address", {})))
        property_type = self.infer_property_type(detail_map.get("Property Type", "Sites"))
        price = self.parse_price(description_en)
        bedrooms = 0 if property_type == "Site" else 2
        bathrooms = 0 if property_type == "Site" else 1
        images = self.download_images(image_urls, f"lhldc-{url.rstrip('/').split('/')[-1]}")

        return {
            "property_id": f"lhldc-{url.rstrip('/').split('/')[-1]}",
            "source": self.source_name,
            "title": title,
            "description_en": description_en,
            "description_st": self.build_sesotho_summary(title, property_type, location_text, bedrooms, bathrooms, price),
            "price": price,
            "currency": "LSL",
            "district": district,
            "location_text": location_text or district,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "image_paths": images,
            "listing_url": url,
            "condition": self.infer_condition(title, description_en),
            "style": self.infer_style(title, description_en, property_type),
            "environment": self.infer_environment(location_text, description_en),
            "amenities": self.infer_amenities(description_en, location_text),
            "listing_intent": status_key,
        }


class PropMarketScraper(WebScraperAdapter):
    source_name = "propmarket"
    listing_reference = "https://www.propmarket.co.ls/sitemap.xml"

    def fetch_raw_records(self) -> list[dict[str, object]]:
        urls = self._property_urls_from_sitemap(self.listing_reference)
        records: list[dict[str, object]] = []
        for url in urls:
            try:
                record = self._parse_detail(url)
            except Exception:
                continue
            if not record:
                continue
            if not self.include_rentals and record["listing_intent"] != "sale":
                continue
            records.append(record)
            if len(records) >= self.max_items:
                break
        return records

    def _property_urls_from_sitemap(self, sitemap_url: str) -> list[str]:
        text = self.get_text(sitemap_url)
        root = ET.fromstring(text)
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [node.text.strip() for node in root.findall(".//sm:loc", namespace) if node.text]
        return [url for url in urls if "/properties/" in url]

    def _parse_detail(self, url: str) -> dict[str, object] | None:
        html = self.get_text(url)
        soup = BeautifulSoup(html, "lxml")

        title = self.clean_text(
            (soup.find("meta", attrs={"property": "og:title"}) or {}).get("content")
            if soup.find("meta", attrs={"property": "og:title"})
            else soup.title.get_text(" ", strip=True)
        )
        meta_description = self.clean_text(
            (soup.find("meta", attrs={"name": "description"}) or {}).get("content", "")
        )
        listing_type = self._extract_regex(r'\\"listingType\\":\\"([^\\"]+)\\"', html) or (
            "rent" if "per month" in meta_description.lower() else "sale"
        )
        property_type_raw = self._extract_regex(r'\\"propertyType\\":\{.*?\\"name\\":\\"([^\\"]+)\\"', html) or title
        description_raw = self._extract_regex(r'\\"description\\":\\"(.*?)\\"', html, re.S)
        description_en = self.clean_text(self.decode_embedded_string(description_raw) or meta_description or title)
        display_name = self.decode_embedded_string(self._extract_regex(r'\\"displayName\\":\\"(.*?)\\"', html))
        address = self.decode_embedded_string(self._extract_regex(r'\\"address\\":\\"(.*?)\\"', html))
        price_value = self._extract_regex(r'\\"price\\":([0-9]+)', html)
        bedrooms = self.parse_int(self._extract_regex(r'\\"bedrooms\\":(null|[0-9]+)', html))
        bathrooms = self.parse_int(self._extract_regex(r'\\"bathrooms\\":(null|[0-9]+)', html))

        image_urls = []
        for image in soup.find_all("img", src=True):
            if "r2.dev" in image["src"]:
                image_urls.append(urljoin(url, image["src"]))

        location_text = display_name or address or meta_description
        district = self.infer_district(location_text)
        property_type = self.infer_property_type(property_type_raw, title, description_en)
        price = int(price_value) if price_value else self.parse_price(meta_description)
        bedrooms = bedrooms if bedrooms is not None else (0 if property_type == "Site" else 2)
        bathrooms = bathrooms if bathrooms is not None else (0 if property_type == "Site" else 1)
        property_id = f"propmarket-{url.rstrip('/').split('/')[-1]}"
        images = self.download_images(image_urls, property_id)

        return {
            "property_id": property_id,
            "source": self.source_name,
            "title": title,
            "description_en": description_en,
            "description_st": self.build_sesotho_summary(title, property_type, location_text, bedrooms, bathrooms, price),
            "price": price,
            "currency": "LSL",
            "district": district,
            "location_text": location_text,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "image_paths": images,
            "listing_url": url,
            "condition": self.infer_condition(title, description_en),
            "style": self.infer_style(title, description_en, property_type_raw),
            "environment": self.infer_environment(location_text, description_en),
            "amenities": self.infer_amenities(description_en, address, meta_description),
            "listing_intent": listing_type,
        }

    @staticmethod
    def _extract_regex(pattern: str, text: str, flags: int = 0) -> str | None:
        match = re.search(pattern, text, flags)
        return match.group(1) if match else None


class SotholandScraper(WebScraperAdapter):
    source_name = "sotholand"
    listing_reference = "https://sotholandproperties.co.ls/properties"

    def fetch_raw_records(self) -> list[dict[str, object]]:
        soup = self.get_soup(self.listing_reference)
        card_map: dict[str, dict[str, object]] = {}
        for card in soup.select("div.property-item"):
            link = card.select_one('a[href*="property-details?id="]')
            title_node = card.select_one("a.d-block.h5")
            if not link or not title_node:
                continue
            detail_url = urljoin(self.listing_reference, link["href"])
            status_node = card.select_one(".bg-info")
            price_node = card.select_one(".bg-white")
            location_node = card.select_one("p")
            smalls = card.select("small")
            size_text = self.clean_text(smalls[0].get_text(" ", strip=True)) if len(smalls) > 0 else ""
            bedrooms_text = self.clean_text(smalls[1].get_text(" ", strip=True)) if len(smalls) > 1 else ""
            bathrooms_text = self.clean_text(smalls[2].get_text(" ", strip=True)) if len(smalls) > 2 else ""
            card_map[detail_url] = {
                "title": self.clean_text(title_node.get_text(" ", strip=True)),
                "status": self.clean_text(status_node.get_text(" ", strip=True) if status_node else ""),
                "price_text": self.clean_text(price_node.get_text(" ", strip=True) if price_node else ""),
                "location_text": self.clean_text(location_node.get_text(" ", strip=True) if location_node else ""),
                "size_text": size_text,
                "bedrooms_text": bedrooms_text,
                "bathrooms_text": bathrooms_text,
                "image_url": urljoin(self.listing_reference, card.select_one("img")["src"]) if card.select_one("img[src]") else "",
            }

        records: list[dict[str, object]] = []
        for detail_url, card_data in card_map.items():
            try:
                record = self._parse_detail(detail_url, card_data)
            except Exception:
                continue
            if not record:
                continue
            if not self.include_rentals and record["listing_intent"] != "sale":
                continue
            records.append(record)
            if len(records) >= self.max_items:
                break
        return records

    def _parse_detail(self, url: str, card_data: dict[str, object]) -> dict[str, object] | None:
        soup = self.get_soup(url)
        property_block = soup.select_one(".property-details")
        if not property_block:
            return None

        description_node = property_block.find("p")
        description_en = self.clean_text(description_node.get_text(" ", strip=True) if description_node else card_data["title"])
        title = self.clean_text(str(card_data.get("title", ""))) or self.clean_text(
            (soup.select_one("a.d-block.h5") or soup.find("h1")).get_text(" ", strip=True)
            if (soup.select_one("a.d-block.h5") or soup.find("h1"))
            else "Sotholand Property"
        )

        image_urls = []
        for img in property_block.find_all("img", src=True):
            full = urljoin(url, img["src"])
            if full not in image_urls:
                image_urls.append(full)
        if card_data.get("image_url"):
            image_urls.insert(0, str(card_data["image_url"]))

        price_text = str(card_data.get("price_text", "")).strip()
        if "sale:" in description_en.lower() and (self.include_rentals is False):
            sale_match = re.search(r"Sale:\s*([^|]+)", description_en, re.I)
            if sale_match:
                price_text = sale_match.group(1).strip()
        location_text = str(card_data.get("location_text", "")).strip()
        district = self.infer_district(location_text, description_en, title)
        property_type = self.infer_property_type(title, description_en)
        bedrooms = self.parse_int(str(card_data.get("bedrooms_text", ""))) or self.parse_int(description_en) or 0
        bathrooms = self.parse_int(str(card_data.get("bathrooms_text", ""))) or 0
        price = self.parse_price(price_text or description_en)
        listing_intent = "rent"
        if "sale" in str(card_data.get("status", "")).lower():
            listing_intent = "sale"
        elif "sale:" in description_en.lower():
            listing_intent = "sale" if not self.include_rentals else "rent"

        property_id = f"sotholand-{url.split('=')[-1]}"
        images = self.download_images(image_urls, property_id)

        return {
            "property_id": property_id,
            "source": self.source_name,
            "title": title,
            "description_en": description_en,
            "description_st": self.build_sesotho_summary(title, property_type, location_text, bedrooms, bathrooms, price),
            "price": price,
            "currency": "LSL",
            "district": district,
            "location_text": location_text or district,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "image_paths": images,
            "listing_url": url,
            "condition": self.infer_condition(title, description_en),
            "style": self.infer_style(title, description_en),
            "environment": self.infer_environment(location_text, description_en),
            "amenities": self.infer_amenities(description_en),
            "listing_intent": listing_intent,
        }


class MestechScraper(WebScraperAdapter):
    source_name = "mestech"
    listing_reference = "https://www.mestech.co.ls/properties/"

    def fetch_raw_records(self) -> list[dict[str, object]]:
        soup = self.get_soup(self.listing_reference)
        cards = soup.select("div.realesate-sale")
        records: list[dict[str, object]] = []
        for index, card in enumerate(cards, start=1):
            try:
                record = self._parse_card(card, index)
            except Exception:
                continue
            if not record:
                continue
            records.append(record)
            if len(records) >= self.max_items:
                break
        return records

    def _parse_card(self, card, index: int) -> dict[str, object] | None:
        headings = [self.clean_text(node.get_text(" ", strip=True)) for node in card.select("h2")]
        if len(headings) < 3:
            return None
        size_text, price_text, label_text = headings[:3]
        address_node = card.select_one(".elementor-icon-list-text")
        image_node = card.select_one("img[src]")
        address_text = self.clean_text(address_node.get_text(" ", strip=True) if address_node else "Maseru, Lesotho")
        image_url = urljoin(self.listing_reference, image_node["src"]) if image_node else ""
        property_id = f"mestech-{re.sub(r'[^A-Za-z0-9]+', '-', label_text).strip('-').lower() or index}"
        description_en = (
            f"Commercial office space {label_text} at {address_text}. Size {size_text}. "
            f"Rental price {price_text}. Suitable for office or business use."
        )
        images = self.download_images([image_url] if image_url else [], property_id)
        return {
            "property_id": property_id,
            "source": self.source_name,
            "title": f"Office Space {label_text}",
            "description_en": description_en,
            "description_st": self.build_sesotho_summary(
                f"Office Space {label_text}",
                "Commercial",
                address_text,
                0,
                0,
                self.parse_price(price_text),
            ),
            "price": self.parse_price(price_text),
            "currency": "LSL",
            "district": self.infer_district(address_text),
            "location_text": address_text,
            "property_type": "Commercial",
            "bedrooms": 0,
            "bathrooms": 0,
            "image_paths": images,
            "listing_url": self.listing_reference,
            "condition": "Good",
            "style": "Contemporary",
            "environment": self.infer_environment(address_text, description_en),
            "amenities": self.infer_amenities(description_en, "electricity maintenance security water toilets"),
            "listing_intent": "rent",
        }


class MosoHoldingsScraper(WebScraperAdapter):
    source_name = "mosoholdings"
    listing_reference = "https://mosoholdings.co.ls/listings-search/"

    def fetch_raw_records(self) -> list[dict[str, object]]:
        soup = self.get_soup(self.listing_reference)
        records: list[dict[str, object]] = []
        for index, card in enumerate(self._listing_cards(soup), start=1):
            try:
                record = self._parse_card(card, index)
            except Exception:
                continue
            if not record:
                continue
            if not self.include_rentals and record["listing_intent"] != "sale":
                continue
            records.append(record)
            if len(records) >= self.max_items:
                break
        return records

    def _listing_cards(self, soup: BeautifulSoup) -> list[BeautifulSoup]:
        cards = []
        for column in soup.select("div.elementor-inner-column"):
            if column.select_one("h3") and column.select_one("h5") and column.select_one("img[src]"):
                cards.append(column)
        return cards

    def _parse_card(self, card, index: int) -> dict[str, object] | None:
        title_node = card.select_one("h3")
        price_node = card.select_one("h5")
        image_node = card.select_one("img[src]")
        link_node = title_node.find("a", href=True) if title_node else None
        category_node = card.select_one("ul.elementor-icon-list-items")
        details_node = card.select_one("div.elementor-widget-text-editor")
        if not title_node or not price_node or not image_node:
            return None

        title = self.clean_text(title_node.get_text(" ", strip=True))
        price_text = self.clean_text(price_node.get_text(" ", strip=True))
        location_text = self._extract_location_text(card)
        category_text = self.clean_text(category_node.get_text(" ", strip=True) if category_node else "")
        details_text = self.clean_text(details_node.get_text(" ", strip=True) if details_node else "")
        full_text = " ".join(part for part in (title, price_text, location_text, category_text, details_text) if part)

        bedrooms = self.parse_int(self._extract_regex(r"Bedrooms:\s*([0-9]+)", details_text)) or 0
        bathrooms = self.parse_int(self._extract_regex(r"Baths:\s*([0-9]+(?:\.[0-9]+)?)", details_text)) or 0
        listing_intent = "rent" if any(token in price_text.lower() for token in ("/mo", "/pm")) else "sale"
        property_type = self.infer_property_type(title, category_text, details_text)
        if "land" in title.lower() or "vacant" in category_text.lower():
            property_type = "Site"
        if "commercial" in category_text.lower():
            property_type = "Commercial"

        property_slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{title}-{location_text}-{index}").strip("-").lower()
        property_id = f"moso-{property_slug or index}"
        image_url = urljoin(self.listing_reference, image_node["src"])
        images = self.download_images([image_url], property_id)
        details_link = details_node.find("a", href=True) if details_node else None
        listing_url = (
            link_node["href"]
            if link_node and link_node.get("href") and link_node["href"] != "#"
            else details_link["href"]
            if details_link and details_link.get("href")
            else self.listing_reference
        )
        district = self.infer_district(location_text)
        price = self.parse_price(price_text)

        return {
            "property_id": property_id,
            "source": self.source_name,
            "title": title,
            "description_en": full_text,
            "description_st": self.build_sesotho_summary(title, property_type, location_text, bedrooms, bathrooms, price),
            "price": price,
            "currency": "LSL",
            "district": district,
            "location_text": location_text,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "image_paths": images,
            "listing_url": listing_url,
            "condition": self.infer_condition(title, details_text, price_text),
            "style": self.infer_style(title, details_text, category_text),
            "environment": self.infer_environment(location_text, details_text),
            "amenities": self.infer_amenities(details_text, category_text),
            "listing_intent": listing_intent,
        }

    def _extract_location_text(self, card) -> str:
        headings = [
            self.clean_text(node.get_text(" ", strip=True))
            for node in card.select("h6")
            if self.clean_text(node.get_text(" ", strip=True))
        ]
        for heading in reversed(headings):
            lowered = heading.lower()
            if lowered not in {"for rent", "for sale", "residential", "commercial"}:
                return heading
        return "Maseru, Lesotho"
