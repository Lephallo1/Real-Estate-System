# Flask Frontend Notes

The project now uses Flask as the main frontend.

## What is included

- MySQL-backed login using the existing auth helpers
- customer registration with:
  - full name
  - email
  - address
  - password
- role-based HTML/CSS dashboards for:
  - admin
  - customer
- visual stock browsing with filters such as:
  - sale
  - rent
  - district
  - price bands
  - 3+ bedrooms
  - pool
  - new build
- recommendation and match-detail pages backed by the current AI pipeline

## Start command

```powershell
python app.py
```

Then open:

- [http://127.0.0.1:5000/login](http://127.0.0.1:5000/login)

## Current approach

- Flask is the active product-style frontend.
- The AI/data pipeline still reads from the same generated artifacts and MySQL-backed auth layer.
