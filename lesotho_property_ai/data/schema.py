from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PropertyRecord:
    property_id: str
    source: str
    title: str
    description_en: str
    description_st: str
    price: int
    currency: str
    district: str
    location_text: str
    property_type: str
    bedrooms: int
    bathrooms: int
    image_paths: list[str]
    listing_url: str
    condition: str
    style: str
    environment: str
    amenities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClientProfile:
    client_id: str
    name: str
    budget_min: int
    budget_max: int
    preferred_districts: list[str]
    preferred_property_types: list[str]
    preferred_bedrooms: int
    free_text_preference_en: str
    free_text_preference_st: str
    preferred_language: str
    preferred_channels: list[str] = field(default_factory=lambda: ["email"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
