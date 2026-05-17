"""Flask application factory for the new dashboard frontend."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, send_file, session, url_for

from .admin import admin_bp
from .auth import auth_bp
from .customer import customer_bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    base_dir = Path(__file__).resolve().parents[2]
    app.config.update(
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "lesothohomeai-dev-secret"),
        BASE_DIR=base_dir,
        ARTIFACT_DIR=base_dir / "generated" / "artifacts",
        IMAGE_ROOT=base_dir / "generated" / "images",
        TEMPLATES_AUTO_RELOAD=True,
    )

    @app.context_processor
    def inject_shell_state():
        return {
            "current_user": {
                "user_id": session.get("user_id"),
                "role": session.get("role"),
                "full_name": session.get("full_name"),
                "email": session.get("email"),
                "authenticated": bool(session.get("authenticated")),
            }
        }

    @app.get("/")
    def index():
        if session.get("authenticated"):
            if session.get("role") == "admin":
                return redirect(url_for("admin.overview"))
            return redirect(url_for("customer.search"))
        return redirect(url_for("auth.login"))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.get("/media/generated-images/<path:relpath>")
    def media_image(relpath: str):
        image_root = Path(app.config["IMAGE_ROOT"]).resolve()
        target = (image_root / relpath).resolve()
        if image_root not in target.parents and target != image_root:
            abort(404)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_file(target)

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(customer_bp)
    return app

