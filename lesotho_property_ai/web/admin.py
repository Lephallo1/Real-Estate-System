"""Admin-facing routes for the Flask dashboard."""

from __future__ import annotations

from statistics import mean

import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

from lesotho_property_ai.db import resolve_database_settings

from .admin_actions import ADMIN_ACTIONS
from .auth import role_required
from .demo_utils import analyze_uploaded_property, generate_nlp_demo_output
from .helpers import (
    apply_stock_filters,
    build_stock_chips,
    grouped_cards,
    load_artifact_csv,
    load_artifact_json,
    load_recommendation_bundle,
    load_stock_frame,
    preview_frame,
    recommendation_cards,
    stock_card_rows,
)
from .shared_utils import format_money_input, parse_budget_amount
from .task_actions import action_choices, read_action_job, start_action_job

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

_OLD_NLP_DEMO_DEFAULTS = {
    "full_name": "Admin Demo",
    "title": "Modern Villa",
    "district": "Maseru",
    "locality": "Maseru West",
    "price": 1850000,
    "bedrooms": 3,
    "property_type": "House",
    "condition": "Good",
    "environment": "Suburban",
    "amenities": "parking, garden",
    "preference_en": "Looking for a modern family home with secure parking and good access.",
    "preference_st": "Ke batla ntlo ya lelapa e modern e nang le parking e sireletsehileng.",
}


def _is_old_nlp_demo_form(values: dict[str, object]) -> bool:
    if not values:
        return False
    for key, old_value in _OLD_NLP_DEMO_DEFAULTS.items():
        current = values.get(key)
        if current in (old_value, str(old_value)):
            continue
        return False
    return True


@admin_bp.get("/access")
def access():
    return redirect(url_for("auth.login", next=url_for("admin.overview")))


def _numeric(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_task_accuracy(metrics: dict[str, object], task: str) -> float:
    return _numeric(metrics.get("tasks", {}).get(task, {}).get("test", {}).get("property_accuracy", 0.0))


def _safe_count_map(values: dict[str, object] | None) -> list[dict[str, object]]:
    mapping = values or {}
    rows = [{"label": str(key), "value": int(_numeric(value, 0))} for key, value in mapping.items()]
    rows.sort(key=lambda item: (-item["value"], item["label"]))
    return rows


def _bar_rows(values: dict[str, object] | None) -> list[dict[str, object]]:
    rows = _safe_count_map(values)
    maximum = max((row["value"] for row in rows), default=1)
    for row in rows:
        row["percent"] = round((row["value"] / maximum) * 100, 1) if maximum else 0.0
    return rows


def _presentation_property_type_rows(values: dict[str, object] | None) -> list[dict[str, object]]:
    hidden_labels = {"site"}
    return [
        row
        for row in _safe_count_map(values)
        if str(row["label"]).strip().lower() not in hidden_labels
    ]


def _admin_sidebar_groups() -> list[dict[str, object]]:
    try:
        stock_count = int(len(load_stock_frame()))
    except Exception:
        stock_count = 0
    try:
        recommendation_metrics = load_artifact_json("house_recommendation_metrics.json")
        matches_generated = int(recommendation_metrics.get("recommendation", {}).get("matches_generated", 0))
    except Exception:
        matches_generated = 0
    try:
        marketing_summary = load_artifact_json("house_recommendation_marketing_summary.json")
        campaigns_generated = int(marketing_summary.get("campaigns_generated", 0))
    except Exception:
        campaigns_generated = 0

    return [
        {
            "label": "Main",
            "items": [
                {"label": "Overview", "endpoint": "admin.overview", "icon": "🏠"},
                {"label": "Properties", "endpoint": "admin.properties", "icon": "🏘️", "badge": stock_count},
                {"label": "Web Scraping", "endpoint": "admin.web_scraping", "icon": "🕸️"},
                {"label": "Data Preparation", "endpoint": "admin.data_preparation", "icon": "🧹"},
            ],
        },
        {
            "label": "AI Modules",
            "items": [
                {"label": "Vision (CNN)", "endpoint": "admin.vision", "icon": "🖼️"},
                {"label": "NLP Studio", "endpoint": "admin.nlp_studio", "icon": "✍️"},
                {"label": "Fusion Engine", "endpoint": "admin.fusion_engine", "icon": "🧩"},
                {"label": "Smart Matching", "endpoint": "admin.smart_matching", "icon": "🎯", "badge": matches_generated},
            ],
        },
        {
            "label": "Marketing",
            "items": [
                {"label": "Campaigns", "endpoint": "admin.campaigns", "icon": "📣", "badge": campaigns_generated},
                {"label": "Analytics", "endpoint": "admin.analytics", "icon": "📊"},
            ],
        },
        {
            "label": "System",
            "items": [
                {"label": "Settings", "endpoint": "admin.settings", "icon": "⚙️"},
            ],
        },
    ]


def _module_action_context(action_key: str) -> dict[str, object]:
    module_action = read_action_job(current_app.config["BASE_DIR"], action_key)
    return {
        "module_action": module_action,
        "auto_refresh_seconds": 5 if module_action.running else None,
    }


def _render_admin(template_name: str, **context):
    context.setdefault("sidebar_groups", _admin_sidebar_groups())
    context.setdefault("quick_actions", action_choices())
    module_action = context.get("module_action")
    if module_action and getattr(module_action, "running", False):
        context.setdefault("auto_refresh_seconds", 5)
    return render_template(template_name, **context)


@admin_bp.post("/actions/<action_key>")
@role_required("admin")
def run_action(action_key: str):
    if action_key not in ADMIN_ACTIONS:
        flash("Unknown admin action.", "danger")
        return redirect(request.referrer or url_for("admin.overview"))
    state = start_action_job(current_app.config["BASE_DIR"], action_key)
    if state.status == "blocked":
        flash(state.message or state.availability_message, "warning")
    elif state.running and "started in the background" in state.message.lower():
        flash(state.message, "success")
    elif state.running:
        flash(f"{state.label} is already running. The page will refresh with the latest output.", "info")
    elif state.status == "failed":
        flash(state.message or f"{state.label} failed.", "danger")
    else:
        flash(state.message or f"{state.label} status updated.", "info")
    return redirect(request.referrer or url_for("admin.overview"))


@admin_bp.get("/")
@role_required("admin")
def root():
    return redirect(url_for("admin.overview"))


@admin_bp.get("/overview")
@role_required("admin")
def overview():
    scrape_summary = load_artifact_json("real_only_scrape_summary.json")
    curation_summary = load_artifact_json("curation_summary.json")
    vision_metrics = load_artifact_json("house_vision_metrics.json")
    bedroom_metrics = load_artifact_json("house_bedroom_metrics.json")
    property_type_metrics = load_artifact_json("residential_property_type_metrics.json")
    recommendation_metrics = load_artifact_json("house_recommendation_metrics.json")
    fusion_summary = load_artifact_json("house_recommendation_fusion_summary.json")

    stock = load_stock_frame()
    district_column = "district" if "district" in stock.columns else "district_canonical"
    district_counts = (
        stock[district_column].fillna("").astype(str).value_counts().to_dict()
        if not stock.empty and district_column in stock.columns
        else {}
    )
    sale_count = (
        int(stock["listing_intent"].fillna("").astype(str).str.lower().eq("sale").sum())
        if not stock.empty and "listing_intent" in stock.columns
        else 0
    )
    ai_confidence = round(
        mean(
            [
                _safe_task_accuracy(vision_metrics, "style"),
                _safe_task_accuracy(vision_metrics, "condition"),
                _safe_task_accuracy(bedroom_metrics, "cnn_bedroom_class"),
                _safe_task_accuracy(property_type_metrics, "cnn_property_type"),
            ]
        ),
        3,
    )

    overview_cards = [
        {
            "label": "Active Listings",
            "value": int(len(stock)),
            "detail_title": "Inventory breakdown",
            "detail_rows": [
                {"label": "Prepared house listings", "value": int(len(stock))},
                {"label": "Sale listings", "value": sale_count},
                {
                    "label": "Rental listings",
                    "value": int(
                        stock["listing_intent"].fillna("").astype(str).str.lower().eq("rent").sum()
                    )
                    if not stock.empty and "listing_intent" in stock.columns
                    else 0,
                },
            ],
        },
        {
            "label": "Total Sales",
            "value": sale_count,
            "detail_title": "Sales-oriented stock view",
            "detail_rows": [
                {"label": "For sale in demo inventory", "value": sale_count},
                {"label": "Mean top match score", "value": recommendation_metrics.get("recommendation", {}).get("mean_top_match_score", 0.0)},
            ],
        },
        {
            "label": "AI Confidence",
            "value": ai_confidence,
            "detail_title": "Model confidence indicators",
            "detail_rows": [
                {"label": "Style test accuracy", "value": _safe_task_accuracy(vision_metrics, "style")},
                {"label": "Condition test accuracy", "value": _safe_task_accuracy(vision_metrics, "condition")},
                {"label": "Grouped bedroom test accuracy", "value": _safe_task_accuracy(bedroom_metrics, "cnn_bedroom_class")},
                {"label": "Property-type test accuracy", "value": _safe_task_accuracy(property_type_metrics, "cnn_property_type")},
            ],
        },
        {
            "label": "Listings by District",
            "value": len(district_counts),
            "detail_title": "District distribution",
            "detail_rows": _safe_count_map(district_counts)[:6],
        },
    ]

    checkpoints = [
        {"label": "Clean records", "value": int(scrape_summary.get("clean_records", 0))},
        {"label": "Curated residential rows", "value": int(curation_summary.get("residential_curated_rows", 0))},
        {"label": "Fusion reliability", "value": fusion_summary.get("mean_fusion_reliability", 0.0)},
        {"label": "Properties considered", "value": recommendation_metrics.get("recommendation", {}).get("properties_considered", 0)},
    ]

    return _render_admin(
        "admin/overview.html",
        page_title="Overview",
        sidebar_title="Admin Dashboard",
        overview_cards=overview_cards,
        assignment_progress=[
            {"module": "Module 1", "summary": "Real scraping across reachable Lesotho sources"},
            {"module": "Module 2", "summary": "House vision models plus auxiliary bedroom/property-type classifiers"},
            {"module": "Module 3", "summary": "Bilingual NLP processing and marketing text generation"},
            {"module": "Module 4", "summary": "Fusion scoring, smart matching, and campaign automation"},
            {"module": "Module 5", "summary": "Role-based Flask dashboard redesign"},
        ],
        checkpoints=checkpoints,
        stock_snapshot=stock_card_rows(stock, limit=3),
    )


@admin_bp.get("/properties")
@role_required("admin")
def properties():
    frame = load_stock_frame()
    filtered, state = apply_stock_filters(frame, request.args.to_dict(flat=True))
    district_column = "district" if "district" in frame.columns else "district_canonical"
    district_total = (
        int(frame[district_column].fillna("").astype(str).nunique())
        if not frame.empty and district_column in frame.columns
        else 0
    )
    return _render_admin(
        "admin/properties.html",
        page_title="Properties",
        sidebar_title="Properties",
        cards=stock_card_rows(filtered, limit=18),
        total_count=len(filtered),
        total_stock=len(frame),
        district_total=district_total,
        chips=build_stock_chips("admin.properties", frame, state),
        sample_rows=filtered.head(20).fillna("").to_dict(orient="records"),
        sample_columns=[column for column in ["property_id", "title", "district", "price", "bedrooms", "listing_intent", "listing_url"] if column in filtered.columns],
    )


@admin_bp.get("/stock")
@role_required("admin")
def stock():
    return redirect(url_for("admin.properties"))


@admin_bp.get("/web-scraping")
@role_required("admin")
def web_scraping():
    summary = load_artifact_json("real_only_scrape_summary.json")
    properties = load_artifact_csv("real_only_properties_cleaned.csv")
    source_counts = _safe_count_map(summary.get("clean_source_counts", {}))
    presentation_properties = properties.copy()
    if not presentation_properties.empty and "property_type" in presentation_properties.columns:
        presentation_properties = presentation_properties.loc[
            ~presentation_properties["property_type"].fillna("").astype(str).str.lower().eq("site")
        ].copy()
    property_type_rows = _presentation_property_type_rows(summary.get("clean_property_type_counts", {}))
    scraping_rows = [
        {"label": "Raw records", "value": int(summary.get("raw_records", 0))},
        {"label": "Clean records", "value": int(summary.get("clean_records", 0))},
        {"label": "Raw images", "value": int(summary.get("raw_image_total", 0))},
        {"label": "Clean images", "value": int(summary.get("clean_image_total", 0))},
    ]
    return _render_admin(
        "admin/web_scraping.html",
        page_title="Web Scraping",
        sidebar_title="Web Scraping",
        **_module_action_context("scraper"),
        summary_cards=scraping_rows,
        source_rows=source_counts,
        source_bar_rows=_bar_rows(summary.get("clean_source_counts", {})),
        property_type_rows=property_type_rows,
        listing_intent_rows=_safe_count_map(summary.get("clean_listing_intent_counts", {})),
        sample=preview_frame(
            presentation_properties,
            ["source", "district", "property_type", "listing_intent", "price", "listing_url"],
            limit=20,
        ),
        sources_requested=summary.get("sources_requested", []),
    )


@admin_bp.get("/data-collection")
@role_required("admin")
def data_collection():
    return redirect(url_for("admin.web_scraping"))


@admin_bp.get("/data-preparation")
@role_required("admin")
def data_preparation():
    summary = load_artifact_json("curation_summary.json")
    residential = load_artifact_csv("properties_residential_curated.csv")
    cnn_candidates = load_artifact_csv("properties_residential_cnn_candidates.csv")
    excluded = load_artifact_csv("properties_residential_cnn_excluded.csv")
    review = load_artifact_csv("house_label_review.csv")

    review_status_col = "review_status" if "review_status" in review.columns else "status"
    review_priority_col = "review_priority" if "review_priority" in review.columns else "priority"
    review_action_col = "review_action" if "review_action" in review.columns else "approved_for_training"

    label_review_summary = {
        "high_priority": int(review[review_priority_col].fillna("").astype(str).str.lower().eq("high").sum())
        if not review.empty and review_priority_col in review.columns
        else 0,
        "medium_priority": int(review[review_priority_col].fillna("").astype(str).str.lower().eq("medium").sum())
        if not review.empty and review_priority_col in review.columns
        else 0,
        "reviewed_rows": int(review[review_status_col].fillna("").astype(str).str.lower().eq("reviewed").sum())
        if not review.empty and review_status_col in review.columns
        else 0,
        "excluded_rows": int(review[review_action_col].fillna("").astype(str).str.lower().eq("exclude").sum())
        if not review.empty and review_action_col in review.columns
        else 0,
    }

    return _render_admin(
        "admin/data_preparation.html",
        page_title="Data Preparation",
        sidebar_title="Data Preparation",
        **_module_action_context("prepare"),
        summary_cards=[
            {"label": "Residential Rows", "value": int(summary.get("residential_rows", 0))},
            {"label": "CNN Candidates", "value": int(summary.get("cnn_candidate_rows", 0))},
            {"label": "Residential Curated", "value": int(summary.get("residential_curated_rows", 0))},
            {"label": "CNN Image Rows", "value": int(summary.get("cnn_image_rows", 0))},
        ],
        label_review_summary=label_review_summary,
        use_bucket_rows=_safe_count_map(summary.get("use_bucket_counts", {})),
        residential_sample=preview_frame(residential, ["property_id", "source", "title", "district_canonical", "locality", "cnn_property_type", "cnn_bedroom_class", "split"], limit=25),
        candidates_sample=preview_frame(cnn_candidates, ["property_id", "title", "cnn_property_type", "cnn_bedroom_class", "split", "source"], limit=25),
        review_sample=preview_frame(review, [column for column in ["property_id", review_priority_col, review_status_col, "flag_summary", review_action_col, "review_notes"] if column in review.columns], limit=25),
        excluded_sample=preview_frame(excluded, ["property_id", "source", "title", "district", "location_text", "cnn_exclusion_reasons"], limit=25),
    )


@admin_bp.route("/vision-cnn", methods=["GET", "POST"])
@role_required("admin")
def vision():
    if request.method == "POST":
        session.pop("vision_demo_result", None)
        session.pop("vision_nlp_prefill", None)
        files = [file for file in request.files.getlist("property_images") if file and file.filename]
        try:
            result = analyze_uploaded_property(current_app.config["BASE_DIR"], files)
            session["vision_demo_result"] = result
            session["vision_nlp_prefill"] = result.get("prefill", {})
            flash("CNN analysis completed for the uploaded property image set.", "success")
        except Exception as exc:
            flash(f"Vision demo analysis could not complete: {exc}", "danger")
        return redirect(url_for("admin.vision"))

    metrics = load_artifact_json("house_vision_metrics.json")
    predictions = load_artifact_csv("house_vision_predictions.csv")
    bedroom_metrics = load_artifact_json("house_bedroom_metrics.json")
    bedroom_predictions = load_artifact_csv("house_bedroom_predictions.csv")
    bedroom_comparison = load_artifact_csv("house_bedroom_comparison.csv")
    bedroom_comparison_summary = load_artifact_json("house_bedroom_comparison.json")
    property_type_metrics = load_artifact_json("residential_property_type_metrics.json")
    property_type_predictions = load_artifact_csv("residential_property_type_predictions.csv")

    task_rows = []
    for task, split_metrics in metrics.get("tasks", {}).items():
        for split, values in split_metrics.items():
            task_rows.append({"task": task, "split": split, **values})

    return _render_admin(
        "admin/vision.html",
        page_title="Vision (CNN)",
        sidebar_title="Vision (CNN)",
        **_module_action_context("vision"),
        summary_metrics=[
            {"label": "Style Test Acc.", "value": _safe_task_accuracy(metrics, "style")},
            {"label": "Condition Test Acc.", "value": _safe_task_accuracy(metrics, "condition")},
            {"label": "Bedroom Test Acc.", "value": _safe_task_accuracy(bedroom_metrics, "cnn_bedroom_class")},
            {"label": "Property-Type Test Acc.", "value": _safe_task_accuracy(property_type_metrics, "cnn_property_type")},
        ],
        task_rows=task_rows,
        prediction_sample=preview_frame(predictions, list(predictions.columns[:10]), limit=20),
        bedroom_sample=preview_frame(bedroom_predictions, list(bedroom_predictions.columns[:10]), limit=20),
        bedroom_comparison=preview_frame(bedroom_comparison, list(bedroom_comparison.columns[:8]), limit=12),
        bedroom_improvement=bedroom_comparison_summary,
        property_type_sample=preview_frame(property_type_predictions, list(property_type_predictions.columns[:10]), limit=20),
        vision_demo=session.get("vision_demo_result"),
    )


@admin_bp.get("/vision-nlp")
@role_required("admin")
def vision_nlp():
    return redirect(url_for("admin.vision"))


@admin_bp.route("/nlp-studio", methods=["GET", "POST"])
@role_required("admin")
def nlp_studio():
    prefill = session.get("vision_nlp_prefill", {}) if request.args.get("prefill") == "vision" else {}
    saved_defaults = session.get("nlp_demo_form", {})
    if _is_old_nlp_demo_form(saved_defaults):
        saved_defaults = {}
        session.pop("nlp_demo_form", None)
        session.pop("nlp_demo_output", None)
    has_prefill = bool(prefill)
    defaults = {
        "full_name": saved_defaults.get("full_name", ""),
        "title": saved_defaults.get("title", prefill.get("title", "") if has_prefill else ""),
        "district": saved_defaults.get("district", prefill.get("district", "") if has_prefill else ""),
        "locality": saved_defaults.get("locality", prefill.get("locality", "") if has_prefill else ""),
        "price": saved_defaults.get("price", prefill.get("price", "") if has_prefill else ""),
        "bedrooms": saved_defaults.get("bedrooms", prefill.get("bedrooms", "") if has_prefill else ""),
        "property_type": saved_defaults.get("property_type", prefill.get("property_type", "") if has_prefill else ""),
        "condition": saved_defaults.get("condition", prefill.get("condition", "") if has_prefill else ""),
        "environment": saved_defaults.get("environment", prefill.get("environment", "") if has_prefill else ""),
        "amenities": saved_defaults.get("amenities", prefill.get("amenities", "") if has_prefill else ""),
        "preference_en": saved_defaults.get("preference_en", ""),
        "preference_st": saved_defaults.get("preference_st", ""),
        "language": saved_defaults.get("language", "en"),
        "tone": saved_defaults.get("tone", "professional"),
        "channel": saved_defaults.get("channel", "email"),
    }
    defaults["price_display"] = format_money_input(defaults.get("price"))
    nlp_result = session.get("nlp_demo_output")

    if request.method == "POST":
        price_raw = request.form.get("price", "")
        has_validation_error = False
        try:
            price = parse_budget_amount(price_raw, "Price")
        except ValueError as exc:
            price = price_raw
            has_validation_error = True
            flash(str(exc), "danger")
        bedrooms_raw = request.form.get("bedrooms", "").strip()
        try:
            bedrooms = int(bedrooms_raw or 0)
        except ValueError:
            bedrooms = 0
            has_validation_error = True
            flash("Bedrooms must be a whole number, for example 3 or 4.", "danger")
        defaults = {
            "full_name": request.form.get("full_name", "").strip(),
            "title": request.form.get("title", "").strip(),
            "district": request.form.get("district", "").strip(),
            "locality": request.form.get("locality", "").strip(),
            "price": price,
            "price_display": format_money_input(price),
            "bedrooms": bedrooms,
            "property_type": request.form.get("property_type", "House").strip() or "House",
            "condition": request.form.get("condition", "Good").strip() or "Good",
            "environment": request.form.get("environment", "Suburban").strip() or "Suburban",
            "amenities": request.form.get("amenities", "").strip(),
            "preference_en": request.form.get("preference_en", "").strip(),
            "preference_st": request.form.get("preference_st", "").strip(),
            "language": request.form.get("language", "en").strip() or "en",
            "tone": request.form.get("tone", "professional").strip() or "professional",
            "channel": request.form.get("channel", "email").strip() or "email",
        }
        if has_validation_error:
            session["nlp_demo_form"] = defaults
            session.pop("nlp_demo_output", None)
            nlp_result = None
        else:
            try:
                nlp_result = generate_nlp_demo_output(
                    full_name=defaults["full_name"],
                    title=defaults["title"],
                    district=defaults["district"],
                    locality=defaults["locality"],
                    price=defaults["price"],
                    bedrooms=defaults["bedrooms"],
                    property_type=defaults["property_type"],
                    condition=defaults["condition"],
                    environment=defaults["environment"],
                    amenities=[item.strip() for item in defaults["amenities"].split(",") if item.strip()],
                    preference_en=defaults["preference_en"],
                    preference_st=defaults["preference_st"],
                    language=defaults["language"],
                    tone=defaults["tone"],
                    channel=defaults["channel"],
                )
                session["nlp_demo_form"] = defaults
                session["nlp_demo_output"] = nlp_result
                flash("Marketing copy generated successfully.", "success")
            except Exception as exc:
                flash(f"NLP Studio could not generate the message: {exc}", "danger")

    nlp_metrics = load_artifact_json("house_nlp_metrics.json")
    nlp_queries = load_artifact_csv("house_nlp_query_results.csv")
    return _render_admin(
        "admin/nlp_studio.html",
        page_title="NLP Studio",
        sidebar_title="NLP Studio",
        **_module_action_context("nlp"),
        defaults=defaults,
        nlp_result=nlp_result,
        nlp_metrics=nlp_metrics,
        nlp_queries=preview_frame(nlp_queries, list(nlp_queries.columns[:10]), limit=20),
        vision_prefill=bool(prefill),
    )


@admin_bp.get("/fusion-engine")
@role_required("admin")
def fusion_engine():
    bundle = load_recommendation_bundle("house_recommendation")
    matches = bundle["matches"]
    fusion = bundle["fusion"]
    metrics = bundle["metrics"]
    return _render_admin(
        "admin/fusion_engine.html",
        page_title="Fusion Engine",
        sidebar_title="Fusion Engine",
        **_module_action_context("recommendations"),
        stats=[
            {"label": "Mean Structured", "value": fusion.get("mean_component_scores", {}).get("structured", 0.0)},
            {"label": "Mean Text", "value": fusion.get("mean_component_scores", {}).get("text", 0.0)},
            {"label": "Mean Vision", "value": fusion.get("mean_component_scores", {}).get("vision", 0.0)},
            {"label": "Reliability", "value": fusion.get("mean_fusion_reliability", 0.0)},
        ],
        fusion=fusion,
        recommendation=metrics.get("recommendation", {}),
        match_sample=preview_frame(
            matches,
            [
                "client_name",
                "property_title",
                "structured_score",
                "text_score",
                "vision_score",
                "overall_score",
                "fusion_reliability",
            ],
            limit=15,
        ),
    )


@admin_bp.get("/recommendations")
@role_required("admin")
def recommendations():
    return redirect(url_for("admin.smart_matching"))


@admin_bp.get("/smart-matching")
@role_required("admin")
def smart_matching():
    bundle = load_recommendation_bundle("house_recommendation")
    cards = recommendation_cards(bundle, single_client=False)
    grouped = grouped_cards(cards)
    client_names = list(grouped.keys())
    selected_client = request.args.get("client") or (client_names[0] if client_names else "")
    selected_cards = grouped.get(selected_client, [])
    top_card = selected_cards[0] if selected_cards else None
    campaign_frame = bundle["campaigns"]
    selected_campaigns = (
        campaign_frame.loc[campaign_frame["client_name"].fillna("").astype(str) == selected_client].copy()
        if not campaign_frame.empty and selected_client
        else pd.DataFrame()
    )
    return _render_admin(
        "admin/smart_matching.html",
        page_title="Smart Matching",
        sidebar_title="Smart Matching",
        **_module_action_context("recommendations"),
        metrics=bundle.get("metrics", {}),
        fusion=bundle.get("fusion", {}),
        client_names=client_names,
        selected_client=selected_client,
        top_card=top_card,
        alternative_cards=selected_cards[1:4],
        match_table=preview_frame(
            pd.DataFrame(selected_cards),
            [
                "property_id",
                "overall_score",
                "structured_score",
                "text_score",
                "vision_score",
                "fusion_reliability",
            ],
            limit=5,
        )
        if selected_cards
        else {"columns": [], "rows": []},
        selected_campaigns=selected_campaigns.head(3).fillna("").to_dict(orient="records"),
    )


@admin_bp.get("/campaigns")
@role_required("admin")
def campaigns():
    bundle = load_recommendation_bundle("house_recommendation")
    campaigns = bundle["campaigns"]
    marketing = bundle["marketing"]
    client_names = sorted(campaigns["client_name"].dropna().astype(str).unique().tolist()) if not campaigns.empty else []
    selected_client = request.args.get("client") or (client_names[0] if client_names else "")
    filtered = (
        campaigns.loc[campaigns["client_name"].fillna("").astype(str) == selected_client].copy()
        if not campaigns.empty and selected_client
        else campaigns.copy()
    )
    preview_row = filtered.iloc[0].to_dict() if not filtered.empty else {}
    timeline_rows = []
    if preview_row:
        timeline_rows = [
            {"label": "Top match score", "value": round(_numeric(preview_row.get("match_score", 0.0)), 3)},
            {"label": "Recommended send window", "value": preview_row.get("recommended_send_window", "")},
            {"label": "Delivery state", "value": preview_row.get("delivery_state", "")},
            {"label": "Channel", "value": preview_row.get("channel", "")},
        ]
    return _render_admin(
        "admin/campaigns.html",
        page_title="Campaigns",
        sidebar_title="Campaigns",
        **_module_action_context("recommendations"),
        marketing=marketing,
        client_names=client_names,
        selected_client=selected_client,
        preview=preview_row,
        timeline_rows=timeline_rows,
        campaigns_table=preview_frame(
            filtered,
            [
                "client_name",
                "property_title",
                "channel",
                "language",
                "campaign_variant",
                "delivery_state",
                "estimated_engagement_score",
            ],
            limit=10,
        ),
    )


@admin_bp.get("/analytics")
@role_required("admin")
def analytics():
    scrape_summary = load_artifact_json("real_only_scrape_summary.json")
    vision_metrics = load_artifact_json("house_vision_metrics.json")
    bedroom_comparison = load_artifact_json("house_bedroom_comparison.json")
    marketing_summary = load_artifact_json("house_recommendation_marketing_summary.json")
    fusion_summary = load_artifact_json("house_recommendation_fusion_summary.json")
    return _render_admin(
        "admin/analytics.html",
        page_title="Analytics",
        sidebar_title="Analytics",
        **_module_action_context("recommendations"),
        summary_cards=[
            {"label": "Clean Records", "value": int(scrape_summary.get("clean_records", 0))},
            {"label": "Style Accuracy", "value": _safe_task_accuracy(vision_metrics, "style")},
            {"label": "Condition Accuracy", "value": _safe_task_accuracy(vision_metrics, "condition")},
            {"label": "Engagement Mean", "value": marketing_summary.get("mean_estimated_engagement_score", 0.0)},
        ],
        source_bars=_bar_rows(scrape_summary.get("clean_source_counts", {})),
        model_bars=[
            {"label": "Style", "value": _safe_task_accuracy(vision_metrics, "style"), "percent": round(_safe_task_accuracy(vision_metrics, "style") * 100, 1)},
            {"label": "Condition", "value": _safe_task_accuracy(vision_metrics, "condition"), "percent": round(_safe_task_accuracy(vision_metrics, "condition") * 100, 1)},
            {"label": "Bedroom", "value": bedroom_comparison.get("improved_grouped_test_property_accuracy", 0.0), "percent": round(_numeric(bedroom_comparison.get("improved_grouped_test_property_accuracy", 0.0)) * 100, 1)},
            {"label": "Property Type", "value": load_artifact_json("residential_property_type_metrics.json").get("tasks", {}).get("cnn_property_type", {}).get("test", {}).get("property_accuracy", 0.0), "percent": round(_numeric(load_artifact_json("residential_property_type_metrics.json").get("tasks", {}).get("cnn_property_type", {}).get("test", {}).get("property_accuracy", 0.0)) * 100, 1)},
        ],
        campaign_bars=_bar_rows(marketing_summary.get("channel_counts", {})),
        fusion=fusion_summary,
    )


@admin_bp.get("/settings")
@role_required("admin")
def settings():
    setup_ok = True
    db_summary = {}
    try:
        settings = resolve_database_settings()
        db_summary = {
            "Host": settings.host,
            "Port": settings.port,
            "Database": settings.name,
            "User": settings.user,
        }
    except Exception as exc:
        setup_ok = False
        db_summary = {"Database status": str(exc)}

    vision_metrics = load_artifact_json("house_vision_metrics.json")
    nlp_metrics = load_artifact_json("house_nlp_metrics.json")
    scrape_summary = load_artifact_json("real_only_scrape_summary.json")

    return _render_admin(
        "admin/settings.html",
        page_title="Settings",
        sidebar_title="Settings",
        setup_ok=setup_ok,
        db_summary=db_summary,
        app_summary={
            "Vision epochs": vision_metrics.get("epochs", 0),
            "Vision best epoch": vision_metrics.get("best_epoch", 0),
            "NLP vocabulary": nlp_metrics.get("vocabulary_size", 0),
            "Scrape sources": ", ".join(scrape_summary.get("sources_requested", [])),
        },
    )
