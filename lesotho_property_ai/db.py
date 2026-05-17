"""MySQL connection helpers for the Flask login and activity layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import tomllib


class DatabaseConfigError(RuntimeError):
    """Raised when required database credentials are missing."""


class DatabaseConnectionError(RuntimeError):
    """Raised when a MySQL connection cannot be created."""


@dataclass(slots=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str


def _mapping_value(mapping: Mapping[str, Any] | None, key: str) -> str | None:
    if not mapping:
        return None
    value = mapping.get(key)
    if value is None:
        return None
    return str(value)


def _load_local_secrets_file() -> Mapping[str, Any]:
    """Load `.flask/secrets.toml` for local scripts when available."""

    search_roots = [
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
    ]
    for root in search_roots:
        candidate = root / ".flask" / "secrets.toml"
        if candidate.exists():
            try:
                with candidate.open("rb") as handle:
                    loaded = tomllib.load(handle)
                if isinstance(loaded, Mapping):
                    return loaded
            except Exception:
                return {}
    return {}


def _split_sql_statements(script: str) -> list[str]:
    """Split a simple SQL setup script into executable statements."""

    statements: list[str] = []
    current: list[str] = []
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        current.append(raw_line)
        if line.endswith(";"):
            statement = "\n".join(current).strip()
            if statement:
                statements.append(statement[:-1].strip())
            current = []
    trailing = "\n".join(current).strip()
    if trailing:
        statements.append(trailing)
    return statements


def resolve_database_settings(
    app_secrets: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> DatabaseSettings:
    """Resolve DB credentials from local app secrets first, then environment."""

    env = environ or os.environ
    secrets = app_secrets or _load_local_secrets_file() or {}
    nested = secrets.get("database") if isinstance(secrets, Mapping) else None
    if not isinstance(nested, Mapping):
        nested = {}

    def pick(*names: str, default: str | None = None) -> str | None:
        for name in names:
            value = _mapping_value(secrets, name) or _mapping_value(nested, name) or env.get(name)
            if value:
                return value
        return default

    host = pick("DB_HOST", "MYSQLHOST")
    port_text = pick("DB_PORT", "MYSQLPORT", default="3306")
    name = pick("DB_NAME", "MYSQLDATABASE")
    user = pick("DB_USER", "MYSQLUSER")
    password = pick("DB_PASSWORD", "MYSQLPASSWORD")

    missing = [key for key, value in {
        "DB_HOST": host,
        "DB_NAME": name,
        "DB_USER": user,
        "DB_PASSWORD": password,
    }.items() if not value]
    if missing:
        raise DatabaseConfigError(
            "Missing database settings: "
            + ", ".join(missing)
            + ". Configure .flask/secrets.toml or environment variables."
        )

    try:
        port = int(port_text or "3306")
    except ValueError as exc:
        raise DatabaseConfigError("DB_PORT must be a valid integer.") from exc

    return DatabaseSettings(
        host=str(host),
        port=port,
        name=str(name),
        user=str(user),
        password=str(password),
    )


def get_connection(app_secrets: Mapping[str, Any] | None = None):
    """Create a MySQL connection using configured settings."""

    try:
        import mysql.connector
    except ModuleNotFoundError as exc:
        raise DatabaseConnectionError(
            "mysql-connector-python is not installed. Install requirements first."
        ) from exc

    settings = resolve_database_settings(app_secrets=app_secrets)
    return _open_connection(settings=settings, include_database=True, mysql_connector=mysql.connector)


def _open_connection(*, settings: DatabaseSettings, include_database: bool, mysql_connector):
    connection_kwargs = {
        "host": settings.host,
        "port": settings.port,
        "user": settings.user,
        "password": settings.password,
        "autocommit": False,
    }
    if include_database:
        connection_kwargs["database"] = settings.name
    try:
        return mysql_connector.connect(**connection_kwargs)
    except Exception as exc:  # pragma: no cover - exact connector exception varies
        raise DatabaseConnectionError(f"Could not connect to MySQL: {exc}") from exc


def execute_sql_script(
    script_path: str | Path,
    *,
    app_secrets: Mapping[str, Any] | None = None,
    include_database: bool = True,
) -> None:
    """Run a schema/setup SQL file against MySQL."""

    try:
        import mysql.connector
    except ModuleNotFoundError as exc:
        raise DatabaseConnectionError(
            "mysql-connector-python is not installed. Install requirements first."
        ) from exc

    script = Path(script_path).read_text(encoding="utf-8")
    settings = resolve_database_settings(app_secrets=app_secrets)
    connection = _open_connection(
        settings=settings,
        include_database=include_database,
        mysql_connector=mysql.connector,
    )
    try:
        cursor = connection.cursor()
        for statement in _split_sql_statements(script):
            cursor.execute(statement)
        connection.commit()
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        connection.close()


def ensure_auth_schema_updates(
    *,
    app_secrets: Mapping[str, Any] | None = None,
) -> None:
    """Apply lightweight auth-table migrations needed by newer dashboard builds."""

    try:
        import mysql.connector
    except ModuleNotFoundError as exc:
        raise DatabaseConnectionError(
            "mysql-connector-python is not installed. Install requirements first."
        ) from exc

    settings = resolve_database_settings(app_secrets=app_secrets)
    connection = _open_connection(
        settings=settings,
        include_database=True,
        mysql_connector=mysql.connector,
    )
    try:
        cursor = connection.cursor()
        cursor.execute("SHOW COLUMNS FROM users")
        columns = {str(row[0]) for row in (cursor.fetchall() or [])}

        if "email" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN email VARCHAR(160) NULL AFTER username")
        if "address" not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN address VARCHAR(255) NULL AFTER role")

        cursor.execute(
            "UPDATE users SET email = %s WHERE username = 'admin_demo' AND (email IS NULL OR email = '')",
            ("admin@lesothohome.ai",),
        )
        cursor.execute(
            "UPDATE users SET email = %s WHERE username = 'customer_demo' AND (email IS NULL OR email = '')",
            ("user@lesothohome.ai",),
        )
        cursor.execute(
            "UPDATE users SET email = username WHERE username LIKE %s AND (email IS NULL OR email = '')",
            ("%@%",),
        )
        cursor.execute(
            "UPDATE users SET email = CONCAT('account', id, '@local.invalid') WHERE email IS NULL OR email = ''"
        )

        cursor.execute("SHOW INDEX FROM users")
        indexes = {str(row[2]) for row in (cursor.fetchall() or [])}
        if "uq_users_email" not in indexes:
            cursor.execute("ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE (email)")

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        connection.close()

