"""Regression coverage for database configuration helpers."""

from __future__ import annotations

import unittest

from lesotho_property_ai.db import DatabaseConfigError, DatabaseSettings, _open_connection, resolve_database_settings


class _FakeConnector:
    def __init__(self) -> None:
        self.kwargs = None

    def connect(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


class DatabaseHelperTests(unittest.TestCase):
    def test_resolve_database_settings_uses_short_connect_timeout_by_default(self) -> None:
        settings = resolve_database_settings(
            app_secrets={
                "DB_HOST": "db.internal",
                "DB_PORT": "3306",
                "DB_NAME": "railway",
                "DB_USER": "root",
                "DB_PASSWORD": "secret",
            },
            environ={},
        )

        self.assertEqual(settings.connect_timeout_seconds, 5)

    def test_resolve_database_settings_accepts_explicit_connect_timeout(self) -> None:
        settings = resolve_database_settings(
            app_secrets={
                "DB_HOST": "db.internal",
                "DB_PORT": "3306",
                "DB_NAME": "railway",
                "DB_USER": "root",
                "DB_PASSWORD": "secret",
                "DB_CONNECT_TIMEOUT": "7",
            },
            environ={},
        )

        self.assertEqual(settings.connect_timeout_seconds, 7)

    def test_resolve_database_settings_rejects_non_positive_timeout(self) -> None:
        with self.assertRaises(DatabaseConfigError):
            resolve_database_settings(
                app_secrets={
                    "DB_HOST": "db.internal",
                    "DB_PORT": "3306",
                    "DB_NAME": "railway",
                    "DB_USER": "root",
                    "DB_PASSWORD": "secret",
                    "DB_CONNECT_TIMEOUT": "0",
                },
                environ={},
            )

    def test_open_connection_passes_timeout_to_connector(self) -> None:
        connector = _FakeConnector()
        settings = DatabaseSettings(
            host="db.internal",
            port=3306,
            name="railway",
            user="root",
            password="secret",
            connect_timeout_seconds=4,
        )

        result = _open_connection(settings=settings, include_database=True, mysql_connector=connector)

        self.assertEqual(result["connection_timeout"], 4)
        self.assertEqual(connector.kwargs["database"], "railway")


if __name__ == "__main__":
    unittest.main()
