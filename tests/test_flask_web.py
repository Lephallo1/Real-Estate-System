"""Regression coverage for the Flask frontend routes."""

from __future__ import annotations

import importlib.util
import unittest


HAS_FLASK = importlib.util.find_spec("flask") is not None


@unittest.skipUnless(HAS_FLASK, "Flask is not installed in this environment.")
class FlaskWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from flask_app import app

        cls.app = app
        cls.app.config.update(TESTING=True)

    def setUp(self) -> None:
        self.client = self.app.test_client()

    def _login_customer_session(self) -> None:
        with self.client.session_transaction() as session:
            session["authenticated"] = True
            session["role"] = "customer"
            session["user_id"] = 1
            session["full_name"] = "Test Customer"
            session["email"] = "test@example.com"

    def _login_admin_session(self) -> None:
        with self.client.session_transaction() as session:
            session["authenticated"] = True
            session["role"] = "admin"
            session["user_id"] = 99
            session["full_name"] = "Admin Tester"
            session["email"] = "admin@example.com"

    def test_login_page_loads(self) -> None:
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)

    def test_health_route_loads(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_database_health_route_loads(self) -> None:
        response = self.client.get("/health/database")
        self.assertIn(response.status_code, {200, 503})

    def test_admin_requires_authentication(self) -> None:
        response = self.client.get("/admin/overview")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))

    def test_customer_requires_authentication(self) -> None:
        response = self.client.get("/customer/search")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))
        self.assertIn("next=/customer/search", response.headers.get("Location", ""))

    def test_customer_access_route_redirects_to_login_first(self) -> None:
        response = self.client.get("/customer/access", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login?next=/customer/search", response.headers.get("Location", ""))

    def test_admin_access_route_redirects_to_login_first(self) -> None:
        response = self.client.get("/admin/access", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login?next=/admin/overview", response.headers.get("Location", ""))

    def test_customer_stock_route_with_session(self) -> None:
        self._login_customer_session()
        response = self.client.get("/customer/stock")
        self.assertEqual(response.status_code, 200)

    def test_customer_settings_route_with_session(self) -> None:
        self._login_customer_session()
        response = self.client.get("/customer/settings")
        self.assertEqual(response.status_code, 200)

    def test_customer_search_post_redirects_to_recommendations(self) -> None:
        self._login_customer_session()
        response = self.client.post(
            "/customer/search",
            data={
                "listing_intent": "sale",
                "preferred_language": "en",
                "preferred_districts": ["Maseru"],
                "budget_min": "300000",
                "budget_max": "2200000",
                "preferred_bedrooms": "3",
                "top_n": "3",
                "preference_en": "Looking for a modern house with parking.",
                "preference_st": "",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/customer/recommendations", response.headers.get("Location", ""))

    def test_admin_new_pages_load_with_session(self) -> None:
        self._login_admin_session()
        for route in (
            "/admin/properties",
            "/admin/web-scraping",
            "/admin/data-preparation",
            "/admin/vision-cnn",
            "/admin/nlp-studio",
            "/admin/fusion-engine",
            "/admin/smart-matching",
            "/admin/campaigns",
            "/admin/analytics",
            "/admin/settings",
        ):
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)

    def test_old_admin_routes_redirect_to_new_pages(self) -> None:
        self._login_admin_session()
        redirects = {
            "/admin/stock": "/admin/properties",
            "/admin/data-collection": "/admin/web-scraping",
            "/admin/vision-nlp": "/admin/vision-cnn",
            "/admin/recommendations": "/admin/smart-matching",
        }
        for route, expected in redirects.items():
            with self.subTest(route=route):
                response = self.client.get(route, follow_redirects=False)
                self.assertEqual(response.status_code, 302)
                self.assertIn(expected, response.headers.get("Location", ""))

    def test_root_redirects_to_login_when_signed_out(self) -> None:
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))

