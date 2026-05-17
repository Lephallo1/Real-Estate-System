
# # Real Data Collection Notebook Script


# %%
from pathlib import Path
import pandas as pd

from lesotho_property_ai.artifacts import artifact_dir
from lesotho_property_ai.data.cleaning import clean_property_dataframe
from lesotho_property_ai.data.live_scrapers import (
    AVAILABLE_LIVE_SOURCES,
    collect_live_property_records,
)

BASE_DIR = Path.cwd()
IMAGE_ROOT = BASE_DIR / "generated" / "images"
OUTPUT_DIR = artifact_dir(BASE_DIR / "generated" / "artifacts", "scraping")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    "creativeproperties",
    "propmarket",
    "sotholand",
    "lesothohousing",
    "mestech",
    "mosoholdings",
]
LIVE_LIMIT = 500
INCLUDE_RENTALS = True
MAX_IMAGES_PER_PROPERTY = 5

print("Available sources:", AVAILABLE_LIVE_SOURCES)
print("Selected sources:", SOURCES)

# %%
raw_df, raw_report = collect_live_property_records(
    image_root=IMAGE_ROOT,
    sources=SOURCES,
    per_source_limit=LIVE_LIMIT,
    include_rentals=INCLUDE_RENTALS,
    max_images_per_property=MAX_IMAGES_PER_PROPERTY,
)

print("Raw scrape report")
print(raw_report)
print("Raw rows:", len(raw_df))
raw_df.head(10)

# %%
properties_df = clean_property_dataframe(raw_df)
report = raw_report

print("Scrape report")
print(report)
print()
print("Rows:", len(properties_df))
print("Source counts:")
print(properties_df["source"].value_counts())

# %%
properties_df[["property_id", "source", "title", "district", "price", "property_type"]].head(20)

# %%
raw_df.to_csv(OUTPUT_DIR / "real_only_properties_raw.csv", index=False)
properties_df.to_csv(OUTPUT_DIR / "real_only_properties_cleaned.csv", index=False)
print("Saved raw:", OUTPUT_DIR / "real_only_properties_raw.csv")
print("Saved cleaned:", OUTPUT_DIR / "real_only_properties_cleaned.csv")

# %%
image_counts = properties_df["image_paths"].map(len)
print("Total images:", int(image_counts.sum()))
print("Average images per property:", round(float(image_counts.mean()), 2) if len(image_counts) else 0.0)
print("Missing price rows:", int(properties_df["price"].isna().sum()) if "price" in properties_df else 0)

# %%
properties_df[["title", "image_paths"]].head(10)
