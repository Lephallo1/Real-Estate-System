"""Customer-facing routes for the Flask dashboard."""

from __future__ import annotations

from uuid import uuid4

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
from .shared_utils import format_money_input as _format_money_input
from .shared_utils import parse_budget_amount as _parse_budget_amount

customer_bp = Blueprint("customer", __name__, url_prefix="/customer")

_OLD_DEMO_PREFERENCE_EN = "Looking for a family house with secure parking, good condition, and access to schools."
_OLD_DEMO_PREFERENCE_ST = "Ke batla ntlo ya lelapa e nang le parking e sireletsehileng le boemo bo botle."

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


def _new_recommendation_prefix() -> str:
    user_id = session.get("user_id", "guest")
    return f"house_user_{user_id}_{uuid4().hex[:8]}"


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
            bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", ""))
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
            "budget_min": "",
            "budget_max": "",
            "preferred_bedrooms": 3,
            "preferred_districts": ["Maseru"],
            "top_n": 3,
            "preference_en": "",
            "preference_st": "",
        },
    )
    defaults = dict(defaults)
    if defaults.get("preference_en") == _OLD_DEMO_PREFERENCE_EN:
        defaults["preference_en"] = ""
    if defaults.get("preference_st") == _OLD_DEMO_PREFERENCE_ST:
        defaults["preference_st"] = ""
    defaults["budget_min_display"] = _format_money_input(defaults.get("budget_min"))
    defaults["budget_max_display"] = _format_money_input(defaults.get("budget_max"))

    if request.method == "POST":
        listing_intent = request.form.get("listing_intent", "sale")
        preferred_language = request.form.get("preferred_language", "en")
        preferred_districts = request.form.getlist("preferred_districts")
        budget_min_raw = request.form.get("budget_min", "")
        budget_max_raw = request.form.get("budget_max", "")
        preferred_bedrooms = int(request.form.get("preferred_bedrooms", 3) or 3)
        top_n = int(request.form.get("top_n", 3) or 3)
        preference_en = request.form.get("preference_en", "").strip()
        preference_st = request.form.get("preference_st", "").strip()
        try:
            budget_min = _parse_budget_amount(budget_min_raw, "Minimum budget")
            budget_max = _parse_budget_amount(budget_max_raw, "Maximum budget")
        except ValueError as exc:
            budget_min = budget_min_raw
            budget_max = budget_max_raw
            flash(str(exc), "danger")

        defaults = {
            "listing_intent": listing_intent,
            "preferred_language": preferred_language,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "budget_min_display": _format_money_input(budget_min),
            "budget_max_display": _format_money_input(budget_max),
            "preferred_bedrooms": preferred_bedrooms,
            "preferred_districts": preferred_districts,
            "top_n": top_n,
            "preference_en": preference_en,
            "preference_st": preference_st,
        }
        session["last_search_summary"] = defaults

        if isinstance(budget_min, str) or isinstance(budget_max, str):
            pass
        elif budget_max < budget_min:
            flash("Maximum budget must be greater than or equal to minimum budget.", "danger")
        elif budget_max <= 0:
            flash("Please enter a maximum budget so the model can strictly protect your price range.", "danger")
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
                artifact_prefix = _new_recommendation_prefix()
                result = run_house_recommendation_for_clients(
                    base_dir=current_app.config["BASE_DIR"],
                    clients=custom_client,
                    top_n=top_n,
                    listing_intent=listing_intent,
                    strict_house_only=True,
                    artifact_prefix=artifact_prefix,
                    constraint_mode="strict",
                )
            except Exception as exc:
                session["customer_has_results"] = False
                session.pop("last_near_recommendation_prefix", None)
                flash(f"The matching engine could not complete the request: {exc}", "danger")
            else:
                near_result = None
                near_prefix = None
                matches_generated = int(result.metrics["recommendation"].get("matches_generated", 0))
                if matches_generated < top_n:
                    try:
                        near_prefix = f"{artifact_prefix}_near"
                        near_result = run_house_recommendation_for_clients(
                            base_dir=current_app.config["BASE_DIR"],
                            clients=custom_client,
                            top_n=max(top_n, 5),
                            listing_intent=listing_intent,
                            strict_house_only=True,
                            artifact_prefix=near_prefix,
                            constraint_mode="near",
                        )
                    except Exception:
                        near_result = None
                        near_prefix = None

                try:
                    record_recommendation_run(
                        int(session["user_id"]),
                        search_request_id=search_request_id,
                        top_n=top_n,
                        listing_intent=listing_intent,
                        properties_considered=int(result.metrics["recommendation"].get("properties_considered", 0)),
                        matches_generated=matches_generated,
                        mean_top_match_score=float(result.metrics["recommendation"].get("mean_top_match_score", 0.0)),
                        artifact_prefix=artifact_prefix,
                    )
                except Exception as exc:
                    flash(f"MySQL recommendation logging did not complete: {exc}", "warning")

                session["customer_has_results"] = True
                session["last_recommendation_prefix"] = artifact_prefix
                near_matches_generated = (
                    int(near_result.metrics["recommendation"].get("matches_generated", 0))
                    if near_result is not None
                    else 0
                )
                if near_prefix and near_matches_generated:
                    session["last_near_recommendation_prefix"] = near_prefix
                else:
                    session.pop("last_near_recommendation_prefix", None)
                if matches_generated:
                    flash("Your strict recommendations are ready.", "success")
                elif near_matches_generated:
                    flash(
                        "No exact bedroom match was found inside your budget and district. Near-bedroom options are available separately.",
                        "warning",
                    )
                else:
                    flash("No exact matches found. Try raising your budget or changing the district/bedroom filters.", "warning")
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

    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", ""))
    cards = recommendation_cards(bundle, single_client=True)
    near_cards = []
    near_prefix = session.get("last_near_recommendation_prefix")
    if near_prefix:
        try:
            near_cards = recommendation_cards(load_recommendation_bundle(near_prefix), single_client=True)
        except Exception:
            near_cards = []
    return _render_customer(
        "customer/recommendations.html",
        page_title="Recommended Homes",
        sidebar_title="Recommended Homes",
        cards=cards,
        metrics=bundle.get("metrics", {}),
        fusion=bundle.get("fusion", {}),
        marketing=bundle.get("marketing", {}),
        search_summary=session.get("last_search_summary", {}),
        near_cards=near_cards,
    )


@customer_bp.get("/recommendations/<property_id>")
@role_required("customer")
def property_detail(property_id: str):
    if not session.get("customer_has_results"):
        flash("Run a search first to inspect a matched home.", "info")
        return redirect(url_for("customer.search"))

    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", ""))
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
    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", ""))
    cards = recommendation_cards(bundle, single_client=True)
    if cards:
        return redirect(url_for("customer.property_detail", property_id=cards[0]["property_id"]))
    flash("Run a search first to inspect match details.", "info")
    return redirect(url_for("customer.search"))


@customer_bp.get("/settings")
@role_required("customer")
def settings():
    bundle = load_recommendation_bundle(session.get("last_recommendation_prefix", "")) if session.get("customer_has_results") else {}
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
