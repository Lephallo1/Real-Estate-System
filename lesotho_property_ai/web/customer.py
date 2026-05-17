"""Customer-facing routes for the Flask dashboard."""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

from lesotho_property_ai.auth_service import record_customer_search, record_recommendation_run
from lesotho_property_ai.pipeline import run_house_recommendation_for_clients

from .auth import role_required
from .helpers import (
    apply_stock_filters,
    build_stock_chips,
    load_recommendation_bundle,
    load_stock_frame,
    recommendation_cards,
    stock_card_rows,
)

customer_bp = Blueprint("customer", __name__, url_prefix="/customer")

CUSTOMER_NAV = [
    {"label": "Search", "endpoint": "customer.search", "icon": "🔎"},
    {"label": "Available Stock", "endpoint": "customer.stock", "icon": "🏘️"},
    {"label": "Recommended Homes", "endpoint": "customer.recommendations", "icon": "✨"},
    {"label": "Why This Match", "endpoint": "customer.property_detail_placeholder", "icon": "🎯"},
    {"label": "Settings", "endpoint": "customer.settings", "icon": "⚙️"},
]


@customer_bp.app_context_processor
def inject_customer_nav():
    return {"customer_nav": CUSTOMER_NAV}


@customer_bp.get("/access")
def access():
    return redirect(url_for("auth.login", next=url_for("customer.search")))


def _stock_summary(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {"total": 0, "sale": 0, "rent": 0, "districts": 0}
    district_column = "district" if "district" in frame.columns else "district_canonical"
    return {
        "total": int(len(frame)),
        "sale": int(frame["listing_intent"].fillna("").astype(str).str.lower().eq("sale").sum())
        if "listing_intent" in frame.columns
        else 0,
        "rent": int(frame["listing_intent"].fillna("").astype(str).str.lower().eq("rent").sum())
        if "listing_intent" in frame.columns
        else 0,
        "districts": int(frame[district_column].fillna("").astype(str).nunique())
        if district_column in frame.columns
        else 0,
    }


def _customer_sidebar_groups() -> list[dict[str, object]]:
    items = [dict(item) for item in CUSTOMER_NAV]
    try:
        stock = load_stock_frame()
        stock_counts = _stock_summary(stock)
        items[1]["badge"] = stock_counts["total"]
    except Exception:
        stock_counts = {"total": 0}
    if session.get("customer_has_results"):
        try:
            bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", "house_user_input"))
            items[2]["badge"] = int(bundle.get("metrics", {}).get("recommendation", {}).get("matches_generated", 0))
        except Exception:
            pass
    return [
        {"label": "Home", "items": items[:4]},
        {"label": "Account", "items": items[4:]},
    ]


def _render_customer(template_name: str, **context):
    context.setdefault("sidebar_groups", _customer_sidebar_groups())
    return render_template(template_name, **context)


@customer_bp.get("/")
@role_required("customer")
def root():
    return redirect(url_for("customer.search"))


@customer_bp.route("/search", methods=["GET", "POST"])
@role_required("customer")
def search():
    stock = load_stock_frame()
    district_column = "district" if "district" in stock.columns else "district_canonical"
    district_options = sorted(
        {
            str(value)
            for value in stock.get(district_column, pd.Series(dtype=str)).dropna().tolist()
            if str(value).strip()
        }
    ) or ["Maseru", "Leribe", "Berea"]
    stock_counts = _stock_summary(stock)

    defaults = session.get(
        "last_search_summary",
        {
            "listing_intent": "sale",
            "preferred_language": "en",
            "budget_min": 300000,
            "budget_max": 2500000,
            "preferred_bedrooms": 3,
            "preferred_districts": ["Maseru"],
            "top_n": 3,
            "preference_en": "Looking for a family house with secure parking, good condition, and access to schools.",
            "preference_st": "Ke batla ntlo ya lelapa e nang le parking e sireletsehileng le boemo bo botle.",
        },
    )

    if request.method == "POST":
        listing_intent = request.form.get("listing_intent", "sale")
        preferred_language = request.form.get("preferred_language", "en")
        preferred_districts = request.form.getlist("preferred_districts")
        budget_min = int(request.form.get("budget_min", 0) or 0)
        budget_max = int(request.form.get("budget_max", 0) or 0)
        preferred_bedrooms = int(request.form.get("preferred_bedrooms", 3) or 3)
        top_n = int(request.form.get("top_n", 3) or 3)
        preference_en = request.form.get("preference_en", "").strip()
        preference_st = request.form.get("preference_st", "").strip()

        defaults = {
            "listing_intent": listing_intent,
            "preferred_language": preferred_language,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "preferred_bedrooms": preferred_bedrooms,
            "preferred_districts": preferred_districts,
            "top_n": top_n,
            "preference_en": preference_en,
            "preference_st": preference_st,
        }
        session["last_search_summary"] = defaults

        if budget_max < budget_min:
            flash("Maximum budget must be greater than or equal to minimum budget.", "danger")
        elif not preferred_districts:
            flash("Please choose at least one district.", "danger")
        else:
            custom_client = pd.DataFrame(
                [
                    {
                        "client_id": f"CUSTOMER-{session['user_id']}",
                        "name": session["full_name"],
                        "budget_min": budget_min,
                        "budget_max": budget_max,
                        "preferred_districts": preferred_districts,
                        "preferred_property_types": ["House"],
                        "preferred_bedrooms": preferred_bedrooms,
                        "free_text_preference_en": preference_en,
                        "free_text_preference_st": preference_st,
                        "preferred_language": preferred_language,
                        "preferred_channels": ["dashboard"],
                    }
                ]
            )

            search_request_id = None
            try:
                search_request_id = record_customer_search(
                    int(session["user_id"]),
                    listing_intent=listing_intent,
                    budget_min=budget_min,
                    budget_max=budget_max,
                    preferred_districts=preferred_districts,
                    preferred_bedrooms=preferred_bedrooms,
                    preferred_language=preferred_language,
                    free_text_preference_en=preference_en,
                    free_text_preference_st=preference_st,
                )
            except Exception as exc:
                flash(f"MySQL search logging did not complete: {exc}", "warning")

            try:
                result = run_house_recommendation_for_clients(
                    base_dir=current_app.config["BASE_DIR"],
                    clients=custom_client,
                    top_n=top_n,
                    listing_intent=listing_intent,
                    strict_house_only=True,
                    artifact_prefix="house_user_input",
                )
            except Exception as exc:
                session["customer_has_results"] = False
                flash(f"The matching engine could not complete the request: {exc}", "danger")
            else:
                try:
                    record_recommendation_run(
                        int(session["user_id"]),
                        search_request_id=search_request_id,
                        top_n=top_n,
                        listing_intent=listing_intent,
                        properties_considered=int(result.metrics["recommendation"].get("properties_considered", 0)),
                        matches_generated=int(result.metrics["recommendation"].get("matches_generated", 0)),
                        mean_top_match_score=float(result.metrics["recommendation"].get("mean_top_match_score", 0.0)),
                        artifact_prefix="house_user_input",
                    )
                except Exception as exc:
                    flash(f"MySQL recommendation logging did not complete: {exc}", "warning")

                session["customer_has_results"] = True
                session["last_recommendation_prefix"] = "house_user_input"
                flash("Your recommendations are ready.", "success")
                return redirect(url_for("customer.recommendations"))

    return _render_customer(
        "customer/search.html",
        page_title="Find a Home",
        sidebar_title="Search",
        district_options=district_options,
        search_defaults=defaults,
        stock_counts=stock_counts,
    )


@customer_bp.get("/stock")
@role_required("customer")
def stock():
    frame = load_stock_frame()
    filtered, state = apply_stock_filters(frame, request.args.to_dict(flat=True))
    stock_counts = _stock_summary(frame)
    return _render_customer(
        "customer/stock.html",
        page_title="Available Stock",
        sidebar_title="Available Stock",
        cards=stock_card_rows(filtered, limit=18),
        total_count=len(filtered),
        stock_counts=stock_counts,
        chips=build_stock_chips("customer.stock", frame, state),
        sample_rows=filtered.head(20).fillna("").to_dict(orient="records"),
        sample_columns=[column for column in ["property_id", "title", "district", "price", "bedrooms", "listing_intent", "listing_url"] if column in filtered.columns],
    )


@customer_bp.get("/recommendations")
@role_required("customer")
def recommendations():
    if not session.get("customer_has_results"):
        flash("Run a search first to generate recommendations.", "info")
        return redirect(url_for("customer.search"))

    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", "house_user_input"))
    cards = recommendation_cards(bundle, single_client=True)
    return _render_customer(
        "customer/recommendations.html",
        page_title="Recommended Homes",
        sidebar_title="Recommended Homes",
        cards=cards,
        metrics=bundle.get("metrics", {}),
        fusion=bundle.get("fusion", {}),
        marketing=bundle.get("marketing", {}),
        search_summary=session.get("last_search_summary", {}),
    )


@customer_bp.get("/recommendations/<property_id>")
@role_required("customer")
def property_detail(property_id: str):
    if not session.get("customer_has_results"):
        flash("Run a search first to inspect a matched home.", "info")
        return redirect(url_for("customer.search"))

    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", "house_user_input"))
    cards = recommendation_cards(bundle, single_client=True)
    selected = next((card for card in cards if card["property_id"] == property_id), None)
    if not selected:
        flash("That matched property could not be found.", "warning")
        return redirect(url_for("customer.recommendations"))

    return _render_customer(
        "customer/property_detail.html",
        page_title="Why This Match",
        sidebar_title="Why This Match",
        card=selected,
    )


@customer_bp.get("/why-this-match")
@role_required("customer")
def property_detail_placeholder():
    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", "house_user_input"))
    cards = recommendation_cards(bundle, single_client=True)
    if cards:
        return redirect(url_for("customer.property_detail", property_id=cards[0]["property_id"]))
    flash("Run a search first to inspect match details.", "info")
    return redirect(url_for("customer.search"))


@customer_bp.get("/settings")
@role_required("customer")
def settings():
    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", "house_user_input")) if session.get("customer_has_results") else {}
    recommendation_metrics = bundle.get("metrics", {}).get("recommendation", {}) if bundle else {}
    search_summary = session.get("last_search_summary", {})
    stock_counts = _stock_summary(load_stock_frame())
    return _render_customer(
        "customer/settings.html",
        page_title="Settings",
        sidebar_title="Settings",
        search_summary=search_summary,
        recommendation_metrics=recommendation_metrics,
        stock_counts=stock_counts,
    )
