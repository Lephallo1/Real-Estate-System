from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from lesotho_property_ai import auth_service


class _FakeCursor:
    def __init__(self, *, fetchone_result=None, lastrowid: int = 1) -> None:
        self.fetchone_result = fetchone_result
        self.lastrowid = lastrowid
        self.executed: list[tuple[str, tuple | None]] = []
        self.closed = False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.fetchone_result

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self.cursor_instance = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self, dictionary=False):
        return self.cursor_instance

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class AuthServiceTests(unittest.TestCase):
    def test_bcrypt_hash_round_trip(self) -> None:
        password_hash = auth_service.hash_password("admin123")
        self.assertNotEqual(password_hash, "admin123")
        self.assertTrue(auth_service.verify_password("admin123", password_hash))
        self.assertFalse(auth_service.verify_password("wrong-password", password_hash))

    def test_authenticate_user_success_logs_audit(self) -> None:
        password_hash = auth_service.hash_password("admin123")
        cursor = _FakeCursor(
            fetchone_result={
                "id": 1,
                "username": "admin_demo",
                "email": "admin@lesothohome.ai",
                "full_name": "Admin Demo",
                "password_hash": password_hash,
                "role": "admin",
            }
        )
        connection = _FakeConnection(cursor)

        result = auth_service.authenticate_user(
            "admin_demo",
            "admin123",
            connection_factory=lambda: connection,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.user.role, "admin")
        self.assertEqual(result.user.email, "admin@lesothohome.ai")
        self.assertTrue(connection.committed)
        self.assertEqual(len(cursor.executed), 2)
        self.assertIn("SELECT id, username, email, full_name, password_hash, role", cursor.executed[0][0])
        self.assertIn("INSERT INTO login_audit", cursor.executed[1][0])

    def test_authenticate_user_failure_logs_generic_result(self) -> None:
        cursor = _FakeCursor(fetchone_result=None)
        connection = _FakeConnection(cursor)

        result = auth_service.authenticate_user(
            "missing_user",
            "wrong-password",
            connection_factory=lambda: connection,
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.user)
        self.assertEqual(result.message, "Invalid username or password.")
        self.assertTrue(connection.committed)
        self.assertEqual(len(cursor.executed), 2)

    def test_record_customer_search_returns_insert_id(self) -> None:
        cursor = _FakeCursor(lastrowid=42)
        connection = _FakeConnection(cursor)

        search_id = auth_service.record_customer_search(
            7,
            listing_intent="sale",
            budget_min=300000,
            budget_max=800000,
            preferred_districts=["Maseru"],
            preferred_bedrooms=3,
            preferred_language="en",
            free_text_preference_en="Looking for a family home.",
            free_text_preference_st="Ke batla ntlo ya lelapa.",
            connection_factory=lambda: connection,
        )

        self.assertEqual(search_id, 42)
        self.assertTrue(connection.committed)
        self.assertEqual(len(cursor.executed), 1)
        self.assertIn("INSERT INTO customer_search_requests", cursor.executed[0][0])

    def test_record_recommendation_run_returns_insert_id(self) -> None:
        cursor = _FakeCursor(lastrowid=99)
        connection = _FakeConnection(cursor)

        run_id = auth_service.record_recommendation_run(
            7,
            search_request_id=42,
            top_n=3,
            listing_intent="sale",
            properties_considered=93,
            matches_generated=3,
            mean_top_match_score=0.73,
            artifact_prefix="house_user_input",
            connection_factory=lambda: connection,
        )

        self.assertEqual(run_id, 99)
        self.assertTrue(connection.committed)
        self.assertEqual(len(cursor.executed), 1)
        self.assertIn("INSERT INTO recommendation_runs", cursor.executed[0][0])

    def test_register_customer_user_creates_customer_account(self) -> None:
        cursor = _FakeCursor(fetchone_result=None, lastrowid=14)
        connection = _FakeConnection(cursor)

        result = auth_service.register_customer_user(
            full_name="Naleli Customer",
            email="naleli@example.com",
            address="Maseru West",
            password="customer123",
            connection_factory=lambda: connection,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.user_id, 14)
        self.assertTrue(connection.committed)
        self.assertEqual(len(cursor.executed), 2)
        self.assertIn("SELECT id", cursor.executed[0][0])
        self.assertIn("INSERT INTO users", cursor.executed[1][0])

    def test_register_customer_user_rejects_duplicate_username(self) -> None:
        cursor = _FakeCursor(fetchone_result={"id": 2})
        connection = _FakeConnection(cursor)

        result = auth_service.register_customer_user(
            full_name="Duplicate User",
            email="user@lesothohome.ai",
            address="Berea",
            password="customer123",
            connection_factory=lambda: connection,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.message, "That email address is already registered.")
        self.assertFalse(connection.committed)
        self.assertTrue(connection.rolled_back)
        self.assertEqual(len(cursor.executed), 1)


if __name__ == "__main__":
    unittest.main()
