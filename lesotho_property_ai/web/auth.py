"""Authentication routes and role guards for the Flask frontend."""

from __future__ import annotations

from functools import wraps
import logging
from typing import Callable
from urllib.parse import urlsplit

from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for

from lesotho_property_ai.auth_service import authenticate_user, register_customer_user
from lesotho_property_ai.db import DatabaseConfigError, DatabaseConnectionError, resolve_database_settings

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)


def _safe_next_target(raw_target: str | None) -> str | None:
    if not raw_target:
        return None
    target = raw_target.strip()
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None
    return target


def _login_redirect_target() -> str:
    next_target = request.full_path.rstrip("?") if request.query_string else request.path
    return url_for("auth.login", next=next_target)


def login_required(view: Callable):
    @wraps(view)
    def wrapped_view(**kwargs):
        if not session.get("authenticated"):
            flash("Please sign in first.", "warning")
            return redirect(_login_redirect_target())
        g.current_role = session.get("role")
        return view(**kwargs)

    return wrapped_view


def role_required(role: str):
    def decorator(view: Callable):
        @wraps(view)
        def wrapped_view(**kwargs):
            if not session.get("authenticated"):
                flash("Please sign in first.", "warning")
                return redirect(_login_redirect_target())
            if session.get("role") != role:
                flash("You do not have access to that page.", "danger")
                if session.get("role") == "admin":
                    return redirect(url_for("admin.overview"))
                return redirect(url_for("customer.search"))
            g.current_role = role
            return view(**kwargs)

        return wrapped_view

    return decorator


def _setup_status() -> tuple[bool, str]:
    try:
        resolve_database_settings()
        return True, ""
    except Exception as exc:  # pragma: no cover - defensive branch for missing config
        return False, str(exc)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    setup_ok, setup_message = _setup_status()
    next_target = _safe_next_target(request.values.get("next"))
    if request.method == "POST" and setup_ok:
        identifier = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        try:
            result = authenticate_user(
                username=identifier,
                password=password,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
        except (DatabaseConfigError, DatabaseConnectionError) as exc:
            logger.warning("Login failed because the database layer is unavailable.", exc_info=True)
            flash(f"Sign-in is temporarily unavailable: {exc}", "danger")
            result = None
        except Exception:
            logger.exception("Unexpected error while processing login for '%s'.", identifier)
            flash("Sign-in could not complete right now. Please try again shortly.", "danger")
            result = None
        if result is None:
            return render_template(
                "auth/login.html",
                page_title="Secure Access",
                setup_ok=setup_ok,
                setup_message=setup_message,
                next_target=next_target,
            )
        if result.success and result.user:
            session.clear()
            session.update(
                {
                    "authenticated": True,
                    "user_id": result.user.user_id,
                    "username": result.user.username,
                    "email": result.user.email,
                    "full_name": result.user.full_name,
                    "role": result.user.role,
                    "customer_has_results": False,
                    "last_recommendation_prefix": "",
                }
            )
            flash("Signed in successfully.", "success")
            if next_target:
                return redirect(next_target)
            if result.user.role == "admin":
                return redirect(url_for("admin.overview"))
            return redirect(url_for("customer.search"))
        flash(result.message, "danger")

    return render_template(
        "auth/login.html",
        page_title="Secure Access",
        setup_ok=setup_ok,
        setup_message=setup_message,
        next_target=next_target,
    )


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    setup_ok, setup_message = _setup_status()
    if request.method == "POST" and setup_ok:
        full_name = request.form.get("full_name", "")
        email = request.form.get("email", "")
        address = request.form.get("address", "")
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
        else:
            try:
                result = register_customer_user(
                    full_name=full_name,
                    email=email,
                    address=address,
                    password=password,
                )
            except (DatabaseConfigError, DatabaseConnectionError) as exc:
                logger.warning("Customer registration failed because the database layer is unavailable.", exc_info=True)
                flash(f"Registration is temporarily unavailable: {exc}", "danger")
                result = None
            except Exception:
                logger.exception("Unexpected error while processing registration for '%s'.", email)
                flash("Registration could not complete right now. Please try again shortly.", "danger")
                result = None
            if result is None:
                return render_template(
                    "auth/register.html",
                    page_title="Create Customer Account",
                    setup_ok=setup_ok,
                    setup_message=setup_message,
                )
            if result.success:
                flash("Account created successfully. Please sign in.", "success")
                return redirect(url_for("auth.login", email=email.strip().lower()))
            flash(result.message, "danger")

    return render_template(
        "auth/register.html",
        page_title="Create Customer Account",
        setup_ok=setup_ok,
        setup_message=setup_message,
    )


@auth_bp.post("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
