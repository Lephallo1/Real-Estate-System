"""Customer-facing routes for the Flask dashboard."""

from __future__ import annotations

import re

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

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "a": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

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


def _format_money_input(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    whole = int(amount)
    cents = int(round((amount - whole) * 100))
    grouped = f"{whole:,}".replace(",", " ")
    if cents:
        return f"{grouped}.{cents:02d}"
    return grouped


def _parse_word_number(text: str) -> float | None:
    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    if not tokens:
        return None

    total = 0.0
    current = 0.0
    saw_number_word = False
    for token in tokens:
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            saw_number_word = True
        elif token in {"and", "ls", "lsl", "maloti", "lotis"}:
            continue
        elif token in {"hundred", "hunderd", "hundered"}:
            current = max(current, 1) * 100
            saw_number_word = True
        elif token in {"thousand", "thousands", "k"}:
            total += max(current, 1) * 1_000
            current = 0
            saw_number_word = True
        elif token in {"million", "millions", "mil", "mi", "m"}:
            total += max(current, 1) * 1_000_000
            current = 0
            saw_number_word = True

    if not saw_number_word:
        return None
    return total + current


def _parse_budget_amount(raw_value: str, field_label: str) -> float:
    text = str(raw_value or "").strip()
    if not text:
        return 0.0

    normalized = text.lower().replace(",", " ")
    number_match = re.search(r"\d[\d\s]*(?:\.\d+)?", normalized)
    if number_match:
        number_text = re.sub(r"\s+", "", number_match.group(0))
        amount = float(number_text)
        multiplier = 1
        after_number = normalized[number_match.end() :]
        if re.search(r"\b(?:m|mi|mil|million|millions)\b", after_number):
            multiplier = 1_000_000
        elif re.search(r"\b(?:k|thousand|thousands)\b", after_number):
            multiplier = 1_000
        if re.search(r"\b(?:hundred|hunderd|hundered)\b\s+\b(?:k|thousand|thousands)\b", after_number):
            multiplier = 100_000
        return amount * multiplier

    word_amount = _parse_word_number(normalized)
    if word_amount is not None:
        return float(word_amount)

    raise ValueError(
        f"{field_label} must be a number or money phrase, for example 1 200 000, 500k, or 4 million."
    )


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
