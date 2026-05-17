"""Authentication and activity logging helpers backed by MySQL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import bcrypt

from .db import get_connection


@dataclass(slots=True)
class AuthenticatedUser:
    user_id: int
    username: str
    email: str
    full_name: str
    role: str


@dataclass(slots=True)
class AuthenticationResult:
    success: bool
    user: AuthenticatedUser | None
    message: str


@dataclass(slots=True)
class RegistrationResult:
    success: bool
    user_id: int | None
    message: str


ConnectionFactory = Callable[[], Any]


def hash_password(password: str) -> str:
    """Return a bcrypt hash suitable for database storage."""

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Validate a plaintext password against a stored bcrypt hash."""

    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _default_connection_factory(app_secrets: Mapping[str, Any] | None) -> ConnectionFactory:
    return lambda: get_connection(app_secrets=app_secrets)


def authenticate_user(
    username: str,
    password: str,
    *,
    app_secrets: Mapping[str, Any] | None = None,
    connection_factory: ConnectionFactory | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> AuthenticationResult:
    """Authenticate one user by email or username and record the login attempt."""

    factory = connection_factory or _default_connection_factory(app_secrets)
    connection = factory()
    cursor = connection.cursor(dictionary=True)
    login_identifier = username.strip()
    login_status = "failure"
    role_at_login: str | None = None
    user_id: int | None = None
    result = AuthenticationResult(
        success=False,
        user=None,
        message="Invalid username or password.",
    )
    try:
        cursor.execute(
            """
            SELECT id, username, email, full_name, password_hash, role
            FROM users
            WHERE (email = %s OR username = %s) AND is_active = TRUE
            LIMIT 1
            """,
            (login_identifier, login_identifier),
        )
        row = cursor.fetchone()
        if row and verify_password(password, str(row.get("password_hash", ""))):
            login_status = "success"
            role_at_login = str(row["role"])
            user_id = int(row["id"])
            user = AuthenticatedUser(
                user_id=user_id,
                username=str(row["username"]),
                email=str(row.get("email") or row["username"]),
                full_name=str(row["full_name"]),
                role=role_at_login,
            )
            result = AuthenticationResult(
                success=True,
                user=user,
                message="Login successful.",
            )

        cursor.execute(
            """
            INSERT INTO login_audit (user_id, username_attempt, login_status, role_at_login, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user_id, login_identifier, login_status, role_at_login, ip_address, user_agent),
        )
        connection.commit()
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def insert_or_update_demo_user(
    username: str,
    email: str,
    full_name: str,
    password: str,
    role: str,
    address: str | None = None,
    *,
    app_secrets: Mapping[str, Any] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> None:
    """Create or update one demo user with a fresh bcrypt hash."""

    factory = connection_factory or _default_connection_factory(app_secrets)
    connection = factory()
    cursor = connection.cursor()
    try:
        password_hash = hash_password(password)
        cursor.execute(
            """
            INSERT INTO users (username, email, full_name, password_hash, role, address, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            ON DUPLICATE KEY UPDATE
                email = VALUES(email),
                full_name = VALUES(full_name),
                password_hash = VALUES(password_hash),
                role = VALUES(role),
                address = VALUES(address),
                is_active = TRUE
            """,
            (username, email, full_name, password_hash, role, address),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def register_customer_user(
    *,
    full_name: str,
    email: str,
    address: str,
    password: str,
    app_secrets: Mapping[str, Any] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> RegistrationResult:
    """Create a new customer account if the email is available."""

    normalized_full_name = full_name.strip()
    normalized_email = email.strip().lower()
    normalized_address = address.strip()
    if not normalized_full_name:
        return RegistrationResult(False, None, "Full name is required.")
    if not normalized_email or "@" not in normalized_email:
        return RegistrationResult(False, None, "A valid email address is required.")
    if len(password.strip()) < 6:
        return RegistrationResult(False, None, "Password must be at least 6 characters long.")

    factory = connection_factory or _default_connection_factory(app_secrets)
    connection = factory()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = %s
            LIMIT 1
            """,
            (normalized_email,),
        )
        existing = cursor.fetchone()
        if existing:
            connection.rollback()
            return RegistrationResult(False, None, "That email address is already registered.")

        password_hash = hash_password(password)
        cursor.execute(
            """
            INSERT INTO users (username, email, full_name, password_hash, role, address, is_active)
            VALUES (%s, %s, %s, %s, 'customer', %s, TRUE)
            """,
            (normalized_email, normalized_email, normalized_full_name, password_hash, normalized_address or None),
        )
        user_id = int(cursor.lastrowid)
        connection.commit()
        return RegistrationResult(True, user_id, "Customer account created successfully.")
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def record_customer_search(
    user_id: int,
    *,
    listing_intent: str,
    budget_min: float | int | None,
    budget_max: float | int | None,
    preferred_districts: list[str],
    preferred_bedrooms: int | None,
    preferred_language: str,
    free_text_preference_en: str,
    free_text_preference_st: str,
    app_secrets: Mapping[str, Any] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> int:
    """Persist one customer search request and return its id."""

    import json

    factory = connection_factory or _default_connection_factory(app_secrets)
    connection = factory()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO customer_search_requests (
                user_id, listing_intent, budget_min, budget_max, preferred_districts,
                preferred_bedrooms, preferred_language, free_text_preference_en, free_text_preference_st
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                listing_intent,
                budget_min,
                budget_max,
                json.dumps(preferred_districts),
                preferred_bedrooms,
                preferred_language,
                free_text_preference_en,
                free_text_preference_st,
            ),
        )
        search_id = int(cursor.lastrowid)
        connection.commit()
        return search_id
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def record_recommendation_run(
    user_id: int,
    *,
    search_request_id: int | None,
    top_n: int,
    listing_intent: str,
    properties_considered: int,
    matches_generated: int,
    mean_top_match_score: float | None,
    artifact_prefix: str,
    app_secrets: Mapping[str, Any] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> int:
    """Persist one recommendation execution summary."""

    factory = connection_factory or _default_connection_factory(app_secrets)
    connection = factory()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO recommendation_runs (
                user_id, search_request_id, top_n, listing_intent,
                properties_considered, matches_generated, mean_top_match_score, artifact_prefix
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                search_request_id,
                top_n,
                listing_intent,
                properties_considered,
                matches_generated,
                mean_top_match_score,
                artifact_prefix,
            ),
        )
        run_id = int(cursor.lastrowid)
        connection.commit()
        return run_id
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()


def fetch_recent_activity(
    *,
    limit: int = 12,
    app_secrets: Mapping[str, Any] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent login/search/recommendation events for the admin timeline."""

    factory = connection_factory or _default_connection_factory(app_secrets)
    connection = factory()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT event_type, title, detail, created_at
            FROM (
                SELECT
                    'login' AS event_type,
                    CONCAT('Login ', login_status) AS title,
                    CONCAT(username_attempt, COALESCE(CONCAT(' (', role_at_login, ')'), '')) AS detail,
                    created_at
                FROM login_audit
                UNION ALL
                SELECT
                    'search' AS event_type,
                    'Customer search' AS title,
                    CONCAT('user_id=', user_id, ', intent=', listing_intent) AS detail,
                    created_at
                FROM customer_search_requests
                UNION ALL
                SELECT
                    'recommendation' AS event_type,
                    'Recommendation run' AS title,
                    CONCAT('user_id=', user_id, ', matches=', matches_generated) AS detail,
                    created_at
                FROM recommendation_runs
            ) recent_events
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall() or []
        return [dict(row) for row in rows]
    finally:
        cursor.close()
        connection.close()
