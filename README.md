# Multimodal Real Estate AI System

An end-to-end Lesotho housing-market system built around:

- live property scraping from reachable Lesotho-related sources
- rule-based cleaning and house-focused curation
- computer vision for bedrooms, condition, style, environment, and property type
- bilingual English/Sesotho NLP
- multimodal property-client matching
- personalized campaign generation
- a Flask HTML/CSS frontend with **admin** and **customer** dashboards
- MySQL-backed authentication and lightweight usage logging

## Frontend Experience

The live frontend is now Flask-based.

- `Admin Dashboard`
  Internal control center for properties, scraping, data preparation, vision, NLP, fusion, smart matching, campaigns, analytics, and settings.
- `Customer Dashboard`
  Home-search experience where a buyer enters preferences, browses stock, receives recommendations, and reviews match explanations.

The login layer uses **MySQL + bcrypt**:

- MySQL stores users and lightweight activity logs
- bcrypt stores password hashes securely
- the AI pipeline remains artifact-driven under `generated/artifacts/`

## Project Layout

- `app.py`
  Main Flask dashboard entrypoint.
- `flask_app.py`
  Explicit Flask app runner used by tests and local startup.
- `main.py`
  Original end-to-end CLI pipeline entrypoint.
- `lesotho_property_ai/`
  Main package with scraping, ML, matching, marketing, auth, and Flask web code.
- `lesotho_property_ai/web/`
  Active Flask routes, templates, static assets, and view helpers.
- `lesotho_property_ai/db.py`
  MySQL connection/configuration helpers.
- `lesotho_property_ai/auth_service.py`
  Authentication, password hashing, and activity logging helpers.
- `scripts/`
  CLI helpers such as:
  - `scripts/init_mysql_auth.py`
  - `scripts/seed_demo_users.py`
  - `scripts/run_scraper.py`
  - `scripts/prepare_modeling_dataset.py`
  - `scripts/prepare_house_label_review.py`
  - `scripts/apply_house_label_review.py`
  - `scripts/train_house_vision_model.py`
  - `scripts/train_house_bedroom_model.py`
  - `scripts/train_residential_property_type_model.py`
  - `scripts/evaluate_nlp_module.py`
  - `scripts/evaluate_bedroom_improvement.py`
  - `scripts/run_house_recommendation_demo.py`
- `docs/`
  Setup guides, runbooks, and demo notes.
- `sql/mysql_auth_schema.sql`
  Database schema for MySQL authentication and usage logs.
- `tests/`
  `unittest` coverage for the backend and Flask routes.
- `generated/artifacts/`
  Organized outputs grouped by module:
  - `scraping/`
  - `curation/`
  - `review/`
  - `vision/`
  - `nlp/`
  - `recommendation/`
  - `pipeline/`

## MySQL Authentication Setup

1. Copy the example secrets file:

```powershell
Copy-Item .flask\secrets.example.toml .flask\secrets.toml
```

2. Edit `.flask/secrets.toml` and provide:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

3. Create the MySQL schema:

```powershell
python scripts/init_mysql_auth.py
```

4. Seed the demo accounts:

```powershell
python scripts/seed_demo_users.py
```

5. Launch the dashboard:

```powershell
python app.py
```

Demo logins:

- Admin: `admin@lesothohome.ai` / `admin123`
- Customer: `user@lesothohome.ai` / `user123`

For full schema details and setup notes, see [docs/MYSQL_SETUP.md](docs/MYSQL_SETUP.md).
For hosting, see [docs/RAILWAY_DEPLOYMENT.md](docs/RAILWAY_DEPLOYMENT.md).

## Quick Start

Run the main dashboard:

```powershell
python app.py
```

Run the main training/evaluation helpers:

```powershell
python scripts/train_house_vision_model.py
python scripts/train_house_bedroom_model.py
python scripts/train_residential_property_type_model.py
python scripts/evaluate_nlp_module.py
python scripts/evaluate_bedroom_improvement.py
python scripts/run_house_recommendation_demo.py --listing-intent sale --top-n 3 --clients 6
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Notes

- The real-data workflow remains the primary path.
- Generated model and dashboard artifacts live under `generated/artifacts/`, grouped by module.
- Real scraped images live under `generated/images/live/`.
- Temporary test folders and disposable logs can be removed safely.
- MySQL is used for login and lightweight usage logs only; model and recommendation outputs remain file-based for v1.



