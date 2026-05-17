# Final Demo Runbook

Run these commands from:

`C:\Users\lepha\Documents\Codex\Real Estate System`

## Quick start

```powershell
python scripts/run_house_recommendation_demo.py --listing-intent sale --top-n 3 --clients 6
python app.py
```

## Recommended page order

1. `Overview`
2. `Properties`
3. `Web Scraping`
4. `Data Preparation`
5. `Vision (CNN)`
6. `NLP Studio`
7. `Fusion Engine`
8. `Smart Matching`

## What to say on each page

### 1. Overview
- Summarizes the whole implemented system.
- Shows that the project is a working Flask application backed by real AI modules.

### 2. Properties
- Show the visual listing gallery, filters, price bands, and sale/rent categories.
- Mention that the cards come from the prepared reviewed inventory.

### 3. Web Scraping
- Explain that the system uses real scraped property data from Lesotho-related sources.
- Show the record counts, image counts, and source coverage.

### 4. Data Preparation
- Explain that raw data was cleaned using rule-based preprocessing.
- Explain that the final modeling path focuses on houses only.

### 5. Vision (CNN)
- Explain that Module 2 uses:
  - one main house model
  - one grouped bedroom support model
  - one residential property-type support model
- Mention that grouped bedroom prediction was added because exact bedroom classes were too unstable.
- Show the upload-based analysis demo when useful.

### 6. NLP Studio
- Explain that the NLP module handles English and Sesotho preference text.
- Show direct message generation and bilingual marketing flow.

### 7. Fusion Engine
- Explain that recommendations combine:
  - structured data
  - NLP text scoring
  - vision outputs
- Show the fusion summary and weight transparency.

### 8. Smart Matching
- Show the best property match, reasons, and generated campaign preview.
- Mention that campaign outputs now include:
  - subject line
  - preview text
  - body message
  - call to action
  - estimated engagement

## Live customer-input demo

On the customer `Search` page:
- enter a buyer name
- choose `sale` or `rent`
- set a budget range
- choose preferred districts
- choose preferred bedrooms
- enter English and/or Sesotho preference text
- click `Find Matching Homes`

## Key honest points

- Bedroom prediction improved after grouping the classes into `1-2`, `3`, and `4+`.
- Style and condition are currently stronger than the original exact-bedroom target.
- The recommender now explains why a house was chosen instead of only showing a score.
- Campaign sending is simulated, not sent to real users.

## Final check command

```powershell
python -m unittest discover -s tests -v
```
