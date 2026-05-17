# Project Progress Notes

## Project
Multimodal AI Real Estate Property Marketing System for Lesotho

## Current Date
May 3, 2026

## Main Goal
Build a real-estate AI prototype that can:
- collect real property listings and images from Lesotho-related sources
- analyze property images with a CNN-style vision module
- analyze listing text and client preferences with NLP
- match properties to clients
- generate personalized marketing messages
- display results in a dashboard

## What We Have Done So Far

### 1. Built the project structure
We created a modular Python project with:
- `app.py` for the Flask dashboard
- `main.py` for command-line runs
- `lesotho_property_ai/` for the core logic
- `tests/` for integration tests
- `notebooks/real_data_collection.py` for VS Code notebook-style scraping and inspection

### 2. Built an end-to-end prototype pipeline
The pipeline already supports:
- property data ingestion
- image feature analysis
- bilingual text processing
- property-client matching
- marketing message generation
- artifact export to CSV and JSON

### 3. Added real web scraping
We implemented real scrapers for:
- `propmarket`
- `creativeproperties`
- `sotholand`
- `lesothohousing`
- `mestech`
- `mosoholdings`

These now feed into the same project pipeline instead of using only simulated data.

### 4. Collected a real-only dataset
Latest saved dataset:
- `369` raw real records
- `346` cleaned real records
- `1162` raw downloaded images
- `1081` cleaned downloaded images

Cleaned source counts:
- `211` from `propmarket`
- `68` from `mosoholdings`
- `35` from `lesothohousing`
- `14` from `creativeproperties`
- `12` from `sotholand`
- `6` from `mestech`

Cleaned property-type counts:
- `197` `House`
- `89` `Site`
- `23` `Townhouse`
- `19` `Commercial`
- `17` `Apartment`
- `1` `Cottage`

### 5. Identified the usable residential subset
For the house-focused CNN work, the strongest current subset is the residential set:
- `238` residential listings
- `788` residential images

This is the subset most suitable for the first real vision-training phase.

### 6. Added data-quality checks
We created quality inspection outputs and flagged suspicious rows.

Current quality-flag summary:
- `90` flagged rows needing review
- most issues are district/location normalization problems
- some issues are suspicious prices
- a few are bedroom outliers

## Where We Are In The Assignment

### Module 1: Web Scraping and Data Collection
Status: `Strong progress`

Covered:
- real source discovery
- scraper implementation
- image downloading
- raw and cleaned dataset generation
- multi-source collection

Still needed:
- more cleaning and normalization
- final dataset freeze for modeling

### Module 2: Computer Vision (CNN)
Status: `Started but not complete`

Covered:
- vision module exists in the prototype
- image handling is integrated into the pipeline
- real images are now collected

Not yet complete:
- no proper labeled real training set yet
- no real fine-tuned CNN results on the cleaned residential dataset yet
- no formal evaluation on held-out labeled real data yet

This is the biggest unfinished technical area.

### Module 3: NLP
Status: `Prototype implemented`

Covered:
- bilingual text-processing workflow
- English/Sesotho support in the prototype
- text similarity and preference handling

Still needed:
- stronger explanation of model choice and evaluation for the report
- optional improvement of real-text cleaning on scraped descriptions

### Module 4: Matching and Marketing Automation
Status: `Prototype implemented`

Covered:
- property-client matching
- score-based ranking
- campaign message generation
- simulated sending

Still needed:
- upgrade from rules/weighted fusion toward a clearer MLP-based fusion story if required by the final technical writeup

### Module 5: Dashboard
Status: `Prototype implemented`

Covered:
- Flask app works
- pipeline can be run from the dashboard
- results can be inspected in the UI

Still needed:
- final polish for demo quality
- cleaner presentation of real-only data views

## Honest Current Position
Right now, the project is no longer just an idea or a toy prototype.

We have already covered:
- the project structure
- the end-to-end workflow
- the real-data scraping backbone
- the prototype NLP/matching/dashboard flow

The project is currently in this stage:

`Transition from data collection into serious cleaning and model preparation`

That means:
- scraping is now good enough to pause
- cleaning is now the immediate bottleneck
- CNN training on real cleaned data is the next major milestone

## Important Files
- [main.py](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\main.py)
- [app.py](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\app.py)
- [live_scrapers.py](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\lesotho_property_ai\data\live_scrapers.py)
- [real_data_collection.py](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\notebooks\real_data_collection.py)
- [real_only_properties_raw.csv](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\generated\artifacts\real_only_properties_raw.csv)
- [real_only_properties_cleaned.csv](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\generated\artifacts\real_only_properties_cleaned.csv)
- [real_only_scrape_summary.json](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\generated\artifacts\real_only_scrape_summary.json)
- [real_only_quality_flags.csv](C:\Users\lepha\Documents\Codex\2026-04-20-files-mentioned-by-the-user-group\generated\artifacts\real_only_quality_flags.csv)

## Next Step Forward
The next correct step is:

### Build the real modeling dataset
We should now:
- clean and normalize districts, prices, and noisy locations
- split the data into `residential`, `site/land`, and `commercial`
- create the residential-only modeling subset
- prepare labels for the first CNN training targets such as:
  - bedrooms
  - property type
  - condition
  - style

### After that
Then we should:
- train or fine-tune the CNN on the residential subset
- evaluate it properly
- feed those outputs into the matching system
- prepare the report methodology around the real dataset instead of simulated data

## Short Version
What we have finished:
- real scraping backbone
- dataset collection
- working prototype pipeline
- working dashboard

What is partly done:
- NLP and matching prototype

What is not finished:
- proper real-data cleaning
- labeled real training data
- real CNN training and evaluation

## Conclusion
We are in a good position now, but the project is not finished.

The scraping phase is good enough for now.
The next phase is the real machine-learning phase:
clean the real dataset, prepare training labels, and train the CNN on the residential subset.
