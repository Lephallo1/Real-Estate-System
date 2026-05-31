# LesothoHomeAI Final Project Report

## Executive Summary
LesothoHomeAI is a multimodal real-estate intelligence system built for the Lesotho housing market. It combines real listing collection, data cleaning, CNN-based visual analysis, bilingual NLP, structured recommendation scoring, and marketing automation inside a role-based Flask dashboard. The project is designed for a live presentation where the dashboard proves the system behavior while the source code proves how each module was implemented.

The final system focuses on house recommendations rather than broad property browsing. It prioritizes strict budget, district, and bedroom matching for customers, uses protected MySQL-backed authentication, and exposes admin modules for scraping, curation, vision, NLP, fusion, Smart Matching, campaigns, and analytics.

## Current Evidence Snapshot
| Area | Evidence |
| --- | --- |
| Raw scraped records | 369 |
| Cleaned usable records | 346 |
| Raw image links/files | 1162 |
| Clean image references | 1081 |
| Residential curated rows | 213 |
| CNN candidate properties | 213 |
| CNN image rows | 697 |
| NLP vocabulary size | 924 |
| NLP query success rate | 100.0% |
| Recommendation properties considered | 93 |
| Clients profiled in demo artifacts | 6 |
| Matches generated in demo artifacts | 18 |
| Campaigns generated in demo artifacts | 6 |
| Mean top-match score | 0.724 |
| Mean fusion reliability | 0.939 |
| Mean marketing engagement estimate | 0.847 |

Clean source distribution: creativeproperties: 14, lesothohousing: 35, mestech: 6, mosoholdings: 68, propmarket: 211, sotholand: 12

Clean property-type distribution: Apartment: 17, Commercial: 19, Cottage: 1, House: 197, Site: 89, Townhouse: 23

## Problem Statement
The Lesotho property market is fragmented across several listing sources. Buyers often search with incomplete or bilingual preferences, while agents need fast ways to match customers to houses and generate persuasive outreach. The project solves this by converting scattered real-estate listings into a structured, searchable, explainable recommendation platform.

## Objectives
- Collect real property data and images from reachable Lesotho real-estate sources.
- Clean scraped HTML, normalize fields, and curate house-focused datasets.
- Train and evaluate CNN models for visual property signals.
- Process English and Sesotho preference text with explainable NLP scoring.
- Fuse structured, text, and vision signals into ranked recommendations.
- Generate bilingual marketing messages from recommendation outputs.
- Provide customer and admin dashboards that are practical for demonstration.

## System Architecture
The pipeline starts with live scraping and ends with role-specific dashboards:

1. Live scrapers collect listing pages and image URLs from allowlisted sources.
2. Data cleaning converts raw rows into consistent prices, districts, listing intent, property type, amenities, and plain-text descriptions.
3. Curation separates residential, commercial, and site/land rows, then prepares CNN candidate datasets.
4. Vision training uses image-level data to learn property condition, style, bedroom class, environment, and residential property type.
5. NLP converts buyer preferences and listing descriptions into explainable text-similarity signals.
6. Fusion scoring combines structured matching, NLP similarity, and vision evidence.
7. Marketing generation turns top matches into English or Sesotho campaign messages.
8. Flask dashboards expose the workflow to customers and administrators.

## Web Scraping Module
The scraping module lives mainly in `lesotho_property_ai/data/live_scrapers.py`. It uses `requests`, `BeautifulSoup`, URL parsing, and source-specific extraction logic to collect listing titles, prices, locations, descriptions, property attributes, and image links.

A recent security hardening pass added SSRF protection. The scraper now validates URLs before requests, allows only known real-estate hosts, rejects private or local IP targets, blocks unsafe redirects, verifies image content types, streams image downloads, and counts skipped unsafe URLs in the scrape report instead of crashing. This matters because a compromised listing page should not be able to make the server fetch internal Railway, localhost, or private-network resources.

## Data Cleaning And Curation
Cleaning is handled in `lesotho_property_ai/data/cleaning.py` and curation in `lesotho_property_ai/data/curation.py`. Important tasks include:

- Stripping HTML tags and encoded entities from property descriptions.
- Normalizing prices from text into numeric LSL values.
- Standardizing district, locality, bedroom, bathroom, property type, and listing intent fields.
- Removing duplicates and low-quality rows.
- Splitting data into residential, commercial, and site/land categories.
- Creating image-level rows for CNN training.

The curation summary proves the dataset was not only scraped but filtered into model-ready artifacts.

## Computer Vision Module
Vision training is implemented in `lesotho_property_ai/vision/training.py`, with upload-time analysis in `lesotho_property_ai/vision/analyzer.py`. The training code uses PyTorch and torchvision, especially a ResNet-style transfer-learning setup. Early layers are frozen first, later layers are fine-tuned, and the model uses class weighting, weighted sampling, dropout, validation tracking, and early stopping to reduce overfitting.

Current saved model evidence:

- House condition accuracy: 83.3%
- House style accuracy: 83.3%
- Exact bedroom accuracy: 41.7%
- Grouped bedroom accuracy: 33.3%
- Environment accuracy: 41.7%
- Residential property-type accuracy: 58.8%

The honest interpretation is that the model is stronger on visual categories like condition/style than on exact bedroom counting. This is expected because exterior images do not always reveal bedroom count. For the dashboard demo, the uploaded-image analyzer can use a Gemini-backed vision description when configured, while the lower evaluation tables remain the actual CNN training evidence.

## NLP Module
The NLP pipeline lives in `lesotho_property_ai/nlp/processor.py`. It is intentionally explainable instead of being a black-box language model. It tokenizes English and Sesotho preference text, normalizes Sesotho spelling variants, extracts property signals, and computes similarity between customer preferences and listing descriptions.

The NLP score combines:

`0.55 * cosine_similarity + 0.25 * keyword_overlap + 0.20 * signal_alignment`

This design lets the group explain why a listing matched: not just "the AI said so", but because keywords, amenities, intent, location, and extracted buyer signals aligned.

## Fusion And Recommendation Engine
The fusion logic is in `lesotho_property_ai/matching/engine.py`. It combines:

- Structured score: budget, district, bedroom count, property type, and amenities.
- Text score: similarity between buyer language and listing descriptions.
- Vision score: property condition/style/environment evidence from CNN artifacts.

The final recommendation is not a single model guess. It is a controlled weighted fusion. This helps the system stay explainable and safe for presentation. The latest stabilization makes customer-facing search strict: main results must obey maximum budget, district, and exact bedroom count. Near-bedroom matches are separated from exact results so the lecturer cannot catch the system recommending a wrong-bedroom house as if it were exact.

Current fusion evidence:

- Mean structured component: 0.962
- Mean text component: 0.231
- Mean vision component: 0.877
- Mean top-match score: 0.724
- Mean fusion reliability: 0.939

## Marketing Automation
Marketing generation is implemented in `lesotho_property_ai/marketing/generator.py`. It creates subject lines, preview text, and full messages from match evidence. The system supports English and Sesotho, keeps language output monolingual, and now generates stronger English hooks for customer-facing "Why this match" and admin campaign previews.

## Flask Dashboard And Authentication
The user interface is a Flask application with role-based routes. Customers can register, sign in, enter house preferences, and receive recommendations. Admins can inspect scraping, data preparation, vision, NLP, fusion, Smart Matching, campaigns, and analytics.

Authentication uses MySQL and bcrypt password hashing. Railway uses environment variables, while local development can use `.flask/secrets.toml`. Secrets are intentionally excluded from the repository.

## Reliability, Multi-User Safety, And Deployment
The final stabilization work focused on presentation reliability:

- Customer searches are isolated by run identifiers so simultaneous users do not overwrite one another.
- Smart Matching and Campaigns read live customer activity instead of fixed demo clients when available.
- Gunicorn can be run with workers and threads for presentation traffic.
- Database failures are handled with friendly in-app messages instead of raw Railway crash pages.
- Scraper URL validation reduces SSRF-style risk.

## Testing Strategy
The project includes unit tests under `tests/`. The important verification areas are:

- Authentication and role handling.
- Data curation and label-review logic.
- NLP processing and marketing generation.
- Recommendation pipeline behavior.
- Vision training utilities.
- Flask route behavior.
- Scraper safety for private/unsafe URLs.

## Reproducible Terminal Commands
```powershell
py -m pip install -r requirements.txt
py scripts/init_mysql_auth.py
py scripts/seed_demo_users.py
py scripts/run_scraper.py --real-only
py scripts/prepare_modeling_dataset.py
py scripts/train_house_vision_model.py
py scripts/train_house_bedroom_model.py
py scripts/train_residential_property_type_model.py
py scripts/evaluate_nlp_module.py
py scripts/run_house_recommendation_demo.py
py flask_app.py
```

## Limitations And Future Work
- Bedroom prediction from exterior images is naturally difficult; the system now treats user-entered bedroom count as a strict structured filter instead of relying on vision alone.
- More labeled house images would improve CNN generalization.
- More Sesotho listing text would improve bilingual NLP quality.
- A production version should add stronger monitoring, database migrations, and rate limiting.
- A larger deployment should use managed object storage for images and background workers for long jobs.

## Conclusion
LesothoHomeAI demonstrates an end-to-end machine-learning system rather than a single isolated model. It collects real data, cleans it, trains visual classifiers, processes bilingual preference text, fuses multiple evidence streams, generates marketing messages, and presents everything through a deployed Flask dashboard. The strongest project argument is that every dashboard result can be traced back to data artifacts and source code.
