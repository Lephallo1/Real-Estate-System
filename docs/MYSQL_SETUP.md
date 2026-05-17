# MySQL Setup For Admin/Customer Login

This project uses **MySQL + bcrypt** for:

- user authentication
- login audit logs
- customer search submissions
- recommendation run summaries

The machine learning pipeline remains file-based under `generated/artifacts/`.

## 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

Required packages for the auth layer:

- `mysql-connector-python`
- `bcrypt`

## 2. Configure Secrets

Copy the example file:

```powershell
Copy-Item .flask\secrets.example.toml .flask\secrets.toml
```

Edit `.flask/secrets.toml`:

```toml
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "lesotho_property_ai_app"
DB_USER = "root"
DB_PASSWORD = "your_mysql_password"
```

The app reads Flask-local secrets first and falls back to environment variables with the same names.

## 3. Database Schema

The SQL schema lives in:

- [mysql_auth_schema.sql](C:\Users\lepha\Documents\Codex\Real%20Estate%20System\sql\mysql_auth_schema.sql)

It creates these tables:

- `users`
- `login_audit`
- `customer_search_requests`
- `recommendation_runs`

## 4. Initialize The Database

```powershell
python scripts/init_mysql_auth.py
```

## 5. Seed Demo Users

```powershell
python scripts/seed_demo_users.py
```

This creates or updates:

- `admin@lesothohome.ai` / `admin123`
- `user@lesothohome.ai` / `user123`

Passwords are stored as **bcrypt hashes**, never plain text.

## 6. Start The Dashboard

```powershell
python app.py
```

## 7. What Uses MySQL vs Files

### Stored in MySQL

- users and roles
- login history
- customer search requests
- recommendation run summaries

### Still Stored In Files

- scraped property datasets
- cleaned/curated datasets
- CNN model files (`.pt`)
- evaluation JSON files
- recommendation CSV/JSON artifacts
- generated campaign outputs

## 8. Useful Commands

```powershell
python scripts/init_mysql_auth.py
python scripts/seed_demo_users.py
python app.py
python -m unittest discover -s tests -v
```
