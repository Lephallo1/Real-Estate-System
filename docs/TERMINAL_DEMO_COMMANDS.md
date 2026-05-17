# Terminal Demo Commands

Run these commands from:

`C:\Users\lepha\Documents\Codex\Real Estate System`

## 1. Prepare the cleaned modeling dataset
This shows the cleaning and curation step.

```powershell
python scripts/prepare_modeling_dataset.py
```

## 2. Train the house vision model

```powershell
python scripts/train_house_vision_model.py
```

## 3. Prepare the house label review sheet

```powershell
python scripts/prepare_house_label_review.py
```

## 4. Apply reviewed labels back into the training dataset

```powershell
python scripts/apply_house_label_review.py
```

Optional retrain on the reviewed dataset:

```powershell
python scripts/train_house_vision_model.py --input-csv generated/artifacts/review/properties_house_reviewed_images.csv
```

## 5. Small terminal scraping demo

```powershell
python scripts/run_scraper.py --sources creativeproperties,propmarket,sotholand --live-limit 1 --include-rentals --max-images 1 --output-dir generated\artifacts\scraping\demo_terminal_scrape --image-root generated\images
```

## 6. Run the house recommendation demo

```powershell
python scripts/run_house_recommendation_demo.py --listing-intent sale --top-n 3 --clients 6
```

## 7. Open the Flask demo dashboard
If MySQL login is not initialized yet, run these first:

```powershell
python scripts/init_mysql_auth.py
python scripts/seed_demo_users.py
```

Then launch the dashboard:

```powershell
python app.py
```

Demo login accounts:
- Admin: `admin@lesothohome.ai` / `admin123`
- Customer: `user@lesothohome.ai` / `user123`

Admin portal sections:
- Overview
- Properties
- Web Scraping
- Data Preparation
- Vision (CNN)
- NLP Studio
- Fusion Engine
- Smart Matching
- Campaigns
- Analytics
- Settings

Customer portal sections:
- Search
- Available Stock
- Recommended Homes
- Why This Match
- Settings

Inside the customer portal:
- enter budget, district, bedrooms, language, and preference text
- click `Find Matching Homes`
- review the ranked homes, images, score breakdown, and generated marketing text

## 8. Optional test command

```powershell
python -m unittest discover -s tests -v
```

## Artifact folders

- `generated\artifacts\scraping`
- `generated\artifacts\curation`
- `generated\artifacts\review`
- `generated\artifacts\vision`
- `generated\artifacts\nlp`
- `generated\artifacts\recommendation`
