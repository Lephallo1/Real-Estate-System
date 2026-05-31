# LesothoHomeAI Presentation Script And Demo Runbook

## Three Possible Introductions
1. Good morning. Our project is LesothoHomeAI, a multimodal AI real-estate system designed for the Lesotho housing market. Instead of only showing a model metric, we built the full workflow from scraping real listings to recommending houses and generating marketing messages.

2. Real-estate data in Lesotho is scattered across different websites, formats, and languages. Our system solves that by collecting listings, cleaning them, analyzing images, understanding English and Sesotho preferences, and matching customers to houses through a Flask dashboard.

3. Today we will show both the dashboard and the code. The dashboard proves what the system does, while the code proves how each module works: scraping, cleaning, CNN vision, NLP, fusion matching, campaigns, and deployment.

## Presenter Strategy
The strongest presentation style is to move between three surfaces: dashboard, code, and artifacts. Each speaker should first explain the concept, then show the dashboard evidence, then briefly open the exact code file that implements it.

## Speaker 1 - Project Overview
Time: 1 to 1.5 minutes.

Dashboard: Admin overview.

Code: No code needed for this speaker.

Main points:
- Introduce the project problem: scattered Lesotho property data and hard manual matching.
- Explain the full pipeline: scrape -> clean -> vision -> NLP -> fusion -> campaigns -> dashboard.
- Mention the scale: 369 raw records, 346 cleaned records, 1081 cleaned image references.
- Say that each following speaker will prove one module through dashboard and source code.

## Speaker 2 - Web Scraping
Time: 1.5 minutes.

Dashboard: Admin Web Scraping tab.

Code to open: `lesotho_property_ai/data/live_scrapers.py`.

Artifacts to mention:
- `generated/artifacts/scraping/real_only_properties_raw.csv`
- `generated/artifacts/scraping/real_only_properties_cleaned.csv`
- `generated/artifacts/scraping/real_only_scrape_summary.json`

Script:
- Define web scraping as automated collection of public listing data from property sources.
- Show the dashboard summary and say the scraper collected 369 raw records and 1162 raw image references.
- Open `live_scrapers.py` and point to the source-specific parsing functions and safe fetch helpers.
- Explain that SSRF protection was added: only allowed hosts are fetched, private IPs are rejected, redirects are checked, and images must return an image content type.

## Speaker 3 - Data Cleaning And Curation
Time: 1.5 minutes.

Dashboard: Data Preparation tab.

Code to open:
- `lesotho_property_ai/data/cleaning.py`
- `lesotho_property_ai/data/curation.py`

Artifacts to mention:
- `generated/artifacts/curation/curation_summary.json`
- `generated/artifacts/curation/properties_residential_curated.csv`
- `generated/artifacts/curation/properties_residential_cnn_images.csv`

Script:
- Explain that scraped data contains messy prices, HTML descriptions, missing values, mixed property types, and repeated listings.
- Show how cleaning converts it into reliable columns for modeling.
- Mention that 213 residential rows and 697 CNN image rows were prepared.
- In code, point to functions that strip HTML, normalize prices, infer listing intent, and prepare modeling splits.

## Speaker 4 - CNN Vision
Time: 1.5 minutes.

Dashboard: Vision tab.

Code to open:
- `lesotho_property_ai/vision/training.py`
- `lesotho_property_ai/vision/analyzer.py`

Artifacts to mention:
- `generated/artifacts/vision/house_vision_metrics.json`
- `generated/artifacts/vision/house_bedroom_metrics.json`
- `generated/artifacts/vision/residential_property_type_metrics.json`

Script:
- Explain that the CNN extracts visual evidence such as condition, style, environment, and property type.
- Be honest: exact bedrooms are difficult from exterior photos, so bedrooms are handled strictly from structured user input during recommendation.
- Metrics to mention: condition 83.3%, style 83.3%, property type 58.8%.
- Show the upload demo only as a presentation aid. Clarify that lower metric tables show the actual saved CNN evaluation.

## Speaker 5 - NLP And Marketing
Time: 1.5 minutes.

Dashboard: NLP Studio and Campaigns tabs.

Code to open:
- `lesotho_property_ai/nlp/processor.py`
- `lesotho_property_ai/marketing/generator.py`

Artifacts to mention:
- `generated/artifacts/nlp/house_nlp_metrics.json`
- `generated/artifacts/recommendation/house_recommendation_campaigns.csv`

Script:
- Explain that buyers can write preferences in English or Sesotho.
- Show the NLP formula: cosine similarity, keyword overlap, and signal alignment.
- Mention vocabulary size 924 and query success rate 100.0%.
- Show marketing messages and explain that they are generated from match evidence, not random text.

## Speaker 6 - Fusion, Smart Matching, Customer Demo, And Deployment
Time: 2 minutes.

Dashboard: Fusion Engine, Smart Matching, Customer Search.

Code to open:
- `lesotho_property_ai/matching/engine.py`
- `lesotho_property_ai/web/customer.py`
- `lesotho_property_ai/web/admin.py`

Artifacts to mention:
- `generated/artifacts/recommendation/house_recommendation_metrics.json`
- `generated/artifacts/recommendation/house_recommendation_matches.csv`

Script:
- Explain that fusion combines structured, NLP, and vision signals.
- Mention current mean top-match score 0.724 and fusion reliability 0.939.
- Demonstrate strict customer search: budget first, then district, then exact bedroom count.
- Explain near-bedroom matches are separated so the main results stay lecturer-safe.
- End by showing Railway/local deployment readiness and MySQL-backed login.

## Common Lecturer Questions
Q: Why not only use CNN for recommendations?
A: House recommendations need budget, district, bedrooms, and buyer text. CNN provides visual evidence, but structured constraints must be obeyed first.

Q: Why is exact bedroom accuracy lower?
A: Many listing images show exteriors, kitchens, or yards, so bedrooms are not visually observable. We solve this by using structured bedroom data as a strict filter.

Q: What makes this machine learning?
A: The project includes trained CNN models, explainable NLP scoring, and a fusion model that combines learned and structured signals.

Q: Why Flask?
A: Flask is lightweight, easy to deploy, and gave us full control over routes, sessions, role-based dashboards, and Railway deployment.

Q: How do you prevent unsafe scraping?
A: The scraper validates schemes, hosts, IPs, redirects, and image content types. Unsafe URLs are skipped and counted.

## Closing
In conclusion, LesothoHomeAI is not just a static website. It is a complete AI pipeline: real data collection, cleaning, CNN vision, bilingual NLP, fusion recommendations, marketing automation, and deployment. The dashboard is the user-facing proof, and the code/artifacts are the technical proof.
