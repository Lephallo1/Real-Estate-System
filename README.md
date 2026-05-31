# LesothoHomeAI

LesothoHomeAI is a multimodal AI real-estate recommendation and marketing system for the Lesotho housing market. It collects real property listings, cleans and curates the data, trains computer-vision models, processes English and Sesotho buyer preferences, fuses structured/text/vision signals, and presents recommendations through a Flask dashboard.

The project is designed for a live presentation: every dashboard claim maps back to source code, generated CSV/JSON artifacts, or a runnable terminal script.

## Final Submission Pack

- Full report: [docs/FINAL_PROJECT_REPORT.docx](docs/FINAL_PROJECT_REPORT.docx)
- Report source: [docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md)
- Presentation script/runbook: [docs/PRESENTATION_SCRIPT_AND_DEMO_RUNBOOK.docx](docs/PRESENTATION_SCRIPT_AND_DEMO_RUNBOOK.docx)
- Runbook source: [docs/PRESENTATION_SCRIPT_AND_DEMO_RUNBOOK.md](docs/PRESENTATION_SCRIPT_AND_DEMO_RUNBOOK.md)
- Slides: [presentation/LesothoHomeAI_Final_Presentation.pptx](presentation/LesothoHomeAI_Final_Presentation.pptx)

## Current Evidence Snapshot

| Area | Current artifact evidence |
| --- | --- |
| Raw scraped records | 369 |
| Cleaned records | 346 |
| Raw image references | 1162 |
| Clean image references | 1081 |
| Residential curated rows | 213 |
| CNN image rows | 697 |
| NLP vocabulary size | 924 |
| NLP query success rate | 100% |
| Recommendation properties considered | 93 |
| Demo matches generated | 18 |
| Demo campaigns generated | 6 |
| Mean top-match score | 0.724 |
| Mean fusion reliability | 0.939 |

## Main Features

- Live property scraping from reachable Lesotho real-estate sources.
- SSRF-hardened scraping with allowlisted hosts, private-network blocking, redirect checks, content-type validation, and image-size limits.
- Data cleaning for prices, districts, property types, listing intent, bedrooms, bathrooms, amenities, and HTML-free descriptions.
- House-focused curation for residential recommendation and CNN training.
- PyTorch vision training for condition, style, environment, bedroom class, and property type.
- Gemini-assisted uploaded-image demo when `GEMINI_API_KEY` is configured, while saved CNN metrics remain the real training evidence.
- Explainable bilingual NLP for English and Sesotho preferences.
- Fusion recommendation engine combining structured, text, and vision signals.
- Strict customer search behavior: budget, district, and exact bedroom count are enforced for main results.
- Campaign generation in English and Sesotho.
- Role-based Flask dashboards for admin and customer users.
- MySQL-backed authentication with bcrypt password hashes.

## Project Layout

| Path | Purpose |
| --- | --- |
| [flask_app.py](flask_app.py) | Flask app entrypoint used locally and by Railway/Gunicorn. |
| [app.py](app.py) | Convenience app runner. |
| [lesotho_property_ai/](lesotho_property_ai) | Main package for data, vision, NLP, matching, marketing, auth, and web code. |
| [lesotho_property_ai/data/live_scrapers.py](lesotho_property_ai/data/live_scrapers.py) | Live scraping plus URL safety controls. |
| [lesotho_property_ai/vision/training.py](lesotho_property_ai/vision/training.py) | CNN training logic. |
| [lesotho_property_ai/vision/analyzer.py](lesotho_property_ai/vision/analyzer.py) | Uploaded-image analysis and fallback logic. |
| [lesotho_property_ai/nlp/processor.py](lesotho_property_ai/nlp/processor.py) | Bilingual NLP scoring. |
| [lesotho_property_ai/matching/engine.py](lesotho_property_ai/matching/engine.py) | Fusion scoring and recommendation ranking. |
| [lesotho_property_ai/marketing/generator.py](lesotho_property_ai/marketing/generator.py) | Marketing subject lines, previews, and messages. |
| [lesotho_property_ai/web/](lesotho_property_ai/web) | Flask routes, templates, static CSS, and dashboard helpers. |
| [scripts/](scripts) | Terminal commands for setup, scraping, training, evaluation, and demos. |
| [generated/artifacts/](generated/artifacts) | CSV/JSON outputs used by the dashboard and report. |
| [tests/](tests) | Unit tests for backend and Flask behavior. |

## Local Setup

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Create a local secrets file from the example:

```powershell
Copy-Item .flask\secrets.example.toml .flask\secrets.toml
```

Edit `.flask/secrets.toml` with your own local database values. Do not commit real secrets.

```toml
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "lesotho_property_ai_app"
DB_USER = "root"
DB_PASSWORD = "your-local-password"
GEMINI_API_KEY = "optional-google-ai-studio-key"
```

Initialize auth tables and demo users:

```powershell
py scripts/init_mysql_auth.py
py scripts/seed_demo_users.py
```

Run the Flask dashboard:

```powershell
py flask_app.py
```

Open the local app:

```text
http://127.0.0.1:5000/login
```

Demo accounts:

```text
Admin: admin@lesothohome.ai / admin123
Customer: user@lesothohome.ai / user123
```

## Railway Setup

Railway should use service variables, not committed secrets.

Required `web` service variables:

```text
FLASK_SECRET_KEY
DB_HOST
DB_PORT
DB_NAME
DB_USER
DB_PASSWORD
GEMINI_API_KEY
```

For the best Railway-to-Railway database connection, use the private MySQL host and port when the web service and MySQL service are in the same project/environment.

Recommended Gunicorn start command for presentation traffic:

```text
gunicorn flask_app:app --workers 2 --threads 8 --timeout 120
```

Database health check:

```text
/health/database
```

## Pipeline Commands

Run the main real-data and modeling workflow:

```powershell
py scripts/run_scraper.py --real-only
py scripts/prepare_modeling_dataset.py
py scripts/train_house_vision_model.py
py scripts/train_house_bedroom_model.py
py scripts/train_residential_property_type_model.py
py scripts/evaluate_nlp_module.py
py scripts/run_house_recommendation_demo.py
```

Build the final submission pack:

```powershell
py scripts/build_final_submission_pack.py
```

## Testing

Run all tests:

```powershell
py -m unittest discover -s tests -v
```

Useful quick checks:

```powershell
py -m compileall lesotho_property_ai scripts
py -m pip check
```

## Presentation Strategy

The recommended live presentation flow is:

1. Explain the project overview and modules.
2. Show the dashboard tab for the current module.
3. Open the matching code file in VS Code.
4. Point to generated artifacts that prove the output.
5. Move to the next group member.

Use [docs/PRESENTATION_SCRIPT_AND_DEMO_RUNBOOK.docx](docs/PRESENTATION_SCRIPT_AND_DEMO_RUNBOOK.docx) for exact speaker cues, code files, dashboard tabs, and common lecturer questions.

## Security Notes

- Real secrets must stay in `.flask/secrets.toml` locally or Railway Variables in production.
- `.flask/secrets.example.toml` is safe to commit; `.flask/secrets.toml` is not.
- Passwords are hashed with bcrypt.
- Live scraping is restricted to known hosts and blocks unsafe/private-network targets.
- The Gemini key is optional and only supports the uploaded-image demo path.

## Known Limitations

- Exact bedroom prediction from exterior images is naturally difficult; customer-entered bedroom count is therefore treated as a strict structured filter.
- More labeled house images would improve CNN generalization.
- More Sesotho listing text would improve bilingual NLP.
- A production-grade version should add database migrations, object storage, background queues, monitoring, and rate limiting.
