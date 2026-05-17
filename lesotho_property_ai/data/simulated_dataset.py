from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from .schema import ClientProfile, PropertyRecord


DISTRICTS = ["Maseru", "Leribe", "Berea", "Mafeteng", "Mohale's Hoek", "Quthing"]
PROPERTY_TYPES = ["House", "Apartment", "Townhouse", "Cottage"]
CONDITIONS = ["New", "Good", "Renovation Needed"]
STYLES = ["Modern", "Family", "Contemporary", "Traditional"]
ENVIRONMENTS = ["Urban", "Suburban", "Hillside", "Garden"]
AMENITIES = [
    "garage",
    "garden",
    "solar geyser",
    "water tank",
    "security wall",
    "balcony",
    "near schools",
    "paved access",
]

DISTRICT_NEIGHBORHOODS = {
    "Maseru": ["Mabote", "Thetsane", "Masianokeng", "Ha Abia"],
    "Leribe": ["Hlotse", "Maputsoe", "Peka", "Tsikoane"],
    "Berea": ["Teyateyaneng", "Mapoteng", "Kueneng", "Ha Matala"],
    "Mafeteng": ["Mafeteng Central", "Thabana Morena", "Makoae", "Sekamaneng"],
    "Mohale's Hoek": ["Qoaling", "Mpharane", "Khubetsoana", "Kolo"],
    "Quthing": ["Quthing Town", "Moyeni", "Liphakoe", "Mashai"],
}

ENGLISH_CLIENT_PREFERENCES = [
    ("Naledi", "I want a modern 3 bedroom home in Maseru near schools and shops."),
    ("Thabo", "Looking for a quiet family house with a garden and secure yard."),
    ("Mpho", "Need an affordable apartment close to transport and town."),
    ("Kamohelo", "I want a neat townhouse with parking and low maintenance."),
    ("Lineo", "Searching for a spacious hillside home with good views."),
    ("Lerato", "I need a warm traditional home for a growing family."),
]

SESOTHO_CLIENT_PREFERENCES = [
    "Ke batla ntlo ya kajeno ya dikamore tse tharo Maseru e haufi le mabenkele le dikolo.",
    "Ke batla ntlo e khutsitseng ya lelapa e nang le serapa le lebala le sireletsehileng.",
    "Ke hloka apartment e theko e tlase e haufi le dipalangwang le toropo.",
    "Ke batla townhouse e hlwekileng e nang le parking le tlhokomelo e nyane.",
    "Ke batla ntlo e batsi e maralleng e nang le pono e ntle.",
    "Ke hloka ntlo ya setso e mofuthu bakeng sa lelapa le ntseng le hola.",
]

COLOR_BY_CONDITION = {
    "New": (102, 197, 122),
    "Good": (249, 215, 97),
    "Renovation Needed": (220, 126, 98),
}

ACCENT_BY_STYLE = {
    "Modern": (38, 70, 83),
    "Family": (68, 110, 82),
    "Contemporary": (51, 88, 140),
    "Traditional": (141, 96, 71),
}

SKY_BY_ENVIRONMENT = {
    "Urban": (185, 220, 245),
    "Suburban": (176, 228, 210),
    "Hillside": (166, 203, 235),
    "Garden": (171, 235, 179),
}


class SimulatedDatasetGenerator:
    def __init__(self, image_dir: Path) -> None:
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        property_count: int = 18,
        client_count: int = 6,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = random.Random(seed)
        properties = [self._build_property_record(rng, index) for index in range(property_count)]
        clients = [self._build_client_profile(rng, index) for index in range(client_count)]
        return (
            pd.DataFrame([record.to_dict() for record in properties]),
            pd.DataFrame([record.to_dict() for record in clients]),
        )

    def _build_property_record(self, rng: random.Random, index: int) -> PropertyRecord:
        district = DISTRICTS[index % len(DISTRICTS)]
        neighborhood = rng.choice(DISTRICT_NEIGHBORHOODS[district])
        property_type = rng.choice(PROPERTY_TYPES)
        bedrooms = rng.randint(1, 5)
        bathrooms = max(1, min(3, bedrooms - 1 if bedrooms > 1 else 1))
        condition = rng.choice(CONDITIONS)
        style = rng.choice(STYLES)
        environment = rng.choice(ENVIRONMENTS)
        amenities = rng.sample(AMENITIES, k=3)
        price = self._estimate_price(
            district=district,
            property_type=property_type,
            bedrooms=bedrooms,
            condition=condition,
            environment=environment,
            rng=rng,
        )
        property_id = f"LS-PROP-{index + 1:03d}"
        title = f"{style} {bedrooms}-bedroom {property_type} in {neighborhood}"
        description_en = (
            f"This {condition.lower()} {property_type.lower()} in {neighborhood}, {district}, offers "
            f"{bedrooms} bedrooms, {bathrooms} bathrooms, a {environment.lower()} setting, and "
            f"features such as {', '.join(amenities)}."
        )
        description_st = (
            f"Ntlo ena ya {property_type.lower()} e {condition.lower()} e fumaneha {neighborhood}, "
            f"{district}, e na le dikamore tse {bedrooms}, dibate tse {bathrooms}, tikoloho ya "
            f"{environment.lower()}, mme e fana ka {', '.join(amenities)}."
        )
        listing_url = f"https://example.ls/listings/{property_id.lower()}"
        image_paths = self._create_property_images(
            property_id=property_id,
            district=district,
            neighborhood=neighborhood,
            property_type=property_type,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            condition=condition,
            style=style,
            environment=environment,
            price=price,
        )
        return PropertyRecord(
            property_id=property_id,
            source="simulated_dataset",
            title=title,
            description_en=description_en,
            description_st=description_st,
            price=price,
            currency="LSL",
            district=district,
            location_text=f"{neighborhood}, {district}",
            property_type=property_type,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            image_paths=image_paths,
            listing_url=listing_url,
            condition=condition,
            style=style,
            environment=environment,
            amenities=amenities,
        )

    def _build_client_profile(self, rng: random.Random, index: int) -> ClientProfile:
        name, english_preference = ENGLISH_CLIENT_PREFERENCES[index % len(ENGLISH_CLIENT_PREFERENCES)]
        sesotho_preference = SESOTHO_CLIENT_PREFERENCES[index % len(SESOTHO_CLIENT_PREFERENCES)]
        preferred_district = DISTRICTS[index % len(DISTRICTS)]
        preferred_type = PROPERTY_TYPES[index % len(PROPERTY_TYPES)]
        preferred_bedrooms = 2 + (index % 3)
        budget_min = 180000 + index * 90000
        budget_max = budget_min + 420000 + rng.randint(0, 120000)
        preferred_language = "st" if index % 2 else "en"
        channels = ["email", "social"]
        return ClientProfile(
            client_id=f"LS-CLIENT-{index + 1:03d}",
            name=name,
            budget_min=budget_min,
            budget_max=budget_max,
            preferred_districts=[preferred_district],
            preferred_property_types=[preferred_type],
            preferred_bedrooms=preferred_bedrooms,
            free_text_preference_en=english_preference,
            free_text_preference_st=sesotho_preference,
            preferred_language=preferred_language,
            preferred_channels=channels,
        )

    def _estimate_price(
        self,
        district: str,
        property_type: str,
        bedrooms: int,
        condition: str,
        environment: str,
        rng: random.Random,
    ) -> int:
        district_factor = {
            "Maseru": 1.30,
            "Leribe": 1.12,
            "Berea": 1.07,
            "Mafeteng": 0.98,
            "Mohale's Hoek": 0.92,
            "Quthing": 0.88,
        }[district]
        type_factor = {
            "House": 1.15,
            "Apartment": 0.92,
            "Townhouse": 1.00,
            "Cottage": 0.84,
        }[property_type]
        condition_factor = {
            "New": 1.12,
            "Good": 1.00,
            "Renovation Needed": 0.81,
        }[condition]
        environment_factor = {
            "Urban": 1.08,
            "Suburban": 1.02,
            "Hillside": 1.03,
            "Garden": 1.05,
        }[environment]
        base_price = 150000 + bedrooms * 85000
        jitter = rng.randint(-25000, 45000)
        return int(base_price * district_factor * type_factor * condition_factor * environment_factor + jitter)

    def _create_property_images(
        self,
        property_id: str,
        district: str,
        neighborhood: str,
        property_type: str,
        bedrooms: int,
        bathrooms: int,
        condition: str,
        style: str,
        environment: str,
        price: int,
    ) -> list[str]:
        property_dir = self.image_dir / property_id
        property_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for view in ("exterior", "interior"):
            image = self._draw_scene(
                label=view,
                district=district,
                neighborhood=neighborhood,
                property_type=property_type,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                condition=condition,
                style=style,
                environment=environment,
                price=price,
            )
            path = property_dir / f"{view}.png"
            image.save(path)
            paths.append(str(path))
        return paths

    def _draw_scene(
        self,
        label: str,
        district: str,
        neighborhood: str,
        property_type: str,
        bedrooms: int,
        bathrooms: int,
        condition: str,
        style: str,
        environment: str,
        price: int,
    ) -> Image.Image:
        image = Image.new("RGB", (720, 420), SKY_BY_ENVIRONMENT[environment])
        draw = ImageDraw.Draw(image)
        accent = ACCENT_BY_STYLE[style]
        base = COLOR_BY_CONDITION[condition]

        draw.rectangle((0, 300, 720, 420), fill=(109, 156, 91))
        if label == "exterior":
            draw.rectangle((180, 155, 540, 300), fill=base, outline=accent, width=4)
            roof = [(160, 165), (360, 75), (560, 165)]
            draw.polygon(roof, fill=accent)
            draw.rectangle((330, 225, 390, 300), fill=(114, 77, 60))
            for index in range(max(2, bedrooms)):
                x0 = 210 + (index % 3) * 95
                y0 = 185 + (index // 3) * 50
                draw.rectangle((x0, y0, x0 + 52, y0 + 36), fill=(229, 240, 249), outline=accent, width=3)
            if environment in {"Garden", "Suburban"}:
                draw.ellipse((90, 190, 145, 280), fill=(56, 118, 64))
                draw.rectangle((112, 250, 122, 300), fill=(91, 62, 46))
            if environment == "Urban":
                draw.rectangle((560, 145, 610, 300), fill=(93, 107, 127))
                draw.rectangle((615, 120, 665, 300), fill=(75, 92, 111))
            if environment == "Hillside":
                draw.polygon([(0, 300), (100, 225), (220, 300)], fill=(123, 155, 106))
                draw.polygon([(560, 300), (650, 230), (720, 300)], fill=(123, 155, 106))
        else:
            draw.rectangle((60, 90, 660, 330), fill=(240, 236, 229), outline=accent, width=4)
            for index in range(bedrooms):
                x = 95 + (index % 3) * 180
                y = 125 + (index // 3) * 95
                draw.rectangle((x, y, x + 100, y + 50), fill=base, outline=accent, width=3)
                draw.rectangle((x + 12, y + 10, x + 88, y + 40), fill=(247, 247, 247), outline=accent)
            draw.rectangle((520, 120, 620, 220), fill=(200, 213, 220), outline=accent, width=3)
            draw.text((530, 155), f"Baths {bathrooms}", fill=(30, 30, 30))

        draw.rectangle((0, 0, 720, 58), fill=accent)
        draw.text((20, 18), f"{property_type} | {bedrooms} bed | {condition}", fill=(255, 255, 255))
        draw.text((20, 70), f"{neighborhood}, {district}", fill=(30, 30, 30))
        draw.text((20, 96), f"{style} style | {environment} setting", fill=(30, 30, 30))
        draw.text((20, 122), f"Price LSL {price:,}", fill=(30, 30, 30))
        draw.text((540, 388), label.upper(), fill=(255, 255, 255))
        return image
