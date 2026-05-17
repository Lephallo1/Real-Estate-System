# Railway Deployment Guide

This project is ready to be deployed as **one Flask app** with **one MySQL database**.

You do **not** need two separate deployments.
After deployment, share two direct entry links:

- customer entry: `https://YOUR-DOMAIN/customer/search`
- admin entry: `https://YOUR-DOMAIN/admin/overview`

Both routes use the same backend and database.

## Why Railway

Railway is a good fit because it supports both:

- Flask deployment: [Railway Flask guide](https://docs.railway.com/guides/flask)
- MySQL provisioning: [Railway MySQL guide](https://docs.railway.com/guides/mysql)

Railway also supports public HTTPS domains:

- [Public networking](https://docs.railway.com/reference/public-networking)
- [Domains](https://docs.railway.com/networking/domains)

## What is already prepared

This repo now includes:

- `Procfile` with:
  - `web: gunicorn flask_app:app`
- `gunicorn` in `requirements.txt`
- Railway-compatible MySQL environment fallback:
  - `MYSQLHOST`
  - `MYSQLPORT`
  - `MYSQLUSER`
  - `MYSQLPASSWORD`
  - `MYSQLDATABASE`
- a health route:
  - `/health`

## Deployment Steps

### 1. Push the project to GitHub

Railway deployment is easiest from a GitHub repository.

### 2. Create a Railway project

In Railway:

1. create a new project
2. add your GitHub repo as the web service
3. add a MySQL service to the same project

Official references:

- Flask deploy: [Railway Flask guide](https://docs.railway.com/guides/flask)
- MySQL service: [Railway MySQL guide](https://docs.railway.com/guides/mysql)

### 3. Set the start command

If Railway does not detect it automatically, set the start command to:

```text
gunicorn flask_app:app
```

Railway start-command reference:

- [Set a Start Command](https://docs.railway.com/deployments/start-command)

### 4. Set the secret key

Add this variable in Railway:

```text
FLASK_SECRET_KEY=your-long-random-secret
```

### 5. Use Railway MySQL variables

If the Flask service can reference the Railway MySQL service variables directly, the app will now understand:

- `MYSQLHOST`
- `MYSQLPORT`
- `MYSQLUSER`
- `MYSQLPASSWORD`
- `MYSQLDATABASE`

You can also manually define:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`

### 6. Initialize the database schema

After the MySQL service is ready, run:

```powershell
python scripts/init_mysql_auth.py
python scripts/seed_demo_users.py
```

You can run these locally against the hosted MySQL credentials, or from a Railway shell if you use one.

### 7. Generate a public domain

In Railway service settings:

1. open `Networking`
2. use `Public Networking`
3. click `Generate Domain`

Official reference:

- [Working with Domains](https://docs.railway.com/networking/domains/working-with-domains)

## Links to share

Once deployed, you can share:

- customer route:
  - `https://YOUR-RAILWAY-DOMAIN/customer/search`
- admin route:
  - `https://YOUR-RAILWAY-DOMAIN/admin/overview`

## Health Check

This route should return `200 OK`:

- `https://YOUR-RAILWAY-DOMAIN/health`

## Demo Accounts

Seeded demo logins:

- Admin: `admin@lesothohome.ai` / `admin123`
- Customer: `user@lesothohome.ai` / `user123`

## Local Network Option

If you want to let groupmates on the same Wi-Fi test your machine without Railway, run:

```powershell
$env:HOST='0.0.0.0'
python app.py
```

Then share:

- `http://YOUR-PC-IP:5000/customer/search`
- `http://YOUR-PC-IP:5000/admin/overview`

This only works while your machine is on and reachable on the same network.
