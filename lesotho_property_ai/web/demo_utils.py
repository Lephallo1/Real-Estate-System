"""Small demo helpers for the Flask frontend.

These functions keep the admin-facing Vision and NLP pages interactive without
changing the underlying training/recommendation pipeline.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from lesotho_property_ai.marketing import MarketingAutomation
from lesotho_property_ai.vision import PropertyVisionAnalyzer


def _image_relative_path(base_dir: Path, absolute_path: Path) -> str:
    image_root = (base_dir / "generated" / "images").resolve()
    return absolute_path.resolve().relative_to(image_root).as_posix()


def save_uploaded_property_images(base_dir: Path, files: list[FileStorage]) -> list[dict[str, str]]:
    """Persist uploaded images under generated/images so Flask can serve them."""

    image_root = base_dir / "generated" / "images" / "uploads" / "admin-demo"
    image_root.mkdir(parents=True, exist_ok=True)

    saved_images: list[dict[str, str]] = []
    for file in files:
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        if not filename:
            continue
        target = image_root / f"{uuid4().hex[:10]}-{filename}"
        file.save(target)
        saved_images.append(
            {
                "path": str(target.resolve()),
                "relpath": _image_relative_path(base_dir, target),
                "filename": filename,
            }
        )
    return saved_images


def analyze_uploaded_property(base_dir: Path, files: list[FileStorage]) -> dict[str, object]:
    """Run a lightweight CNN demo against uploaded property images."""

    saved_images = save_uploaded_property_images(base_dir, files)
    if not saved_images:
        raise ValueError("Please upload at least one property image.")

    image_paths = [item["path"] for item in saved_images]
    analyzer = PropertyVisionAnalyzer()
    seed = hashlib.sha256(image_paths[0].encode("utf-8")).hexdigest()
    property_row = {
        "property_id": f"UPLOAD-{seed[:10]}",
        "source": "admin_upload",
        "title": "Uploaded Property",
        "description_en": "Admin-uploaded property image for CNN analysis.",
        "description_st": "",
        "price": 0,
        "currency": "LSL",
        "district": "Maseru",
        "district_canonical": "Maseru",
        "location_text": "Uploaded demo image",
        "locality": "Maseru",
        "property_type": "House",
        "bedrooms": 3,
        "bathrooms": 2,
        "image_paths": image_paths,
        "listing_url": "",
        "condition": "Good",
        "style": "Modern",
        "environment": "Suburban",
        "amenities": ["parking", "road access"],
        "listing_intent": "sale",
    }
    frame = pd.DataFrame([property_row])
    result = analyzer.analyze(frame)
    analyzed = result.dataframe.iloc[0].to_dict()

    first_image = Path(image_paths[0])
    features = analyzer._extract_image_features(first_image)  # type: ignore[attr-defined]
    if not str(analyzed.get("predicted_style", "")).strip():
        analyzed["predicted_style"] = "Modern" if features["brightness"] >= 0.58 else "Traditional"
    if not str(analyzed.get("predicted_environment", "")).strip():
        analyzed["predicted_environment"] = "Garden" if features["green"] >= 0.42 else "Suburban"
    if not str(analyzed.get("predicted_condition", "")).strip():
        analyzed["predicted_condition"] = "New" if features["brightness"] >= 0.64 else "Good"

    confidence = float(analyzed.get("vision_confidence", 0.72) or 0.72)
    confidence = max(0.58, min(confidence, 0.97))

    return {
        "image_count": len(saved_images),
        "image_relpaths": [item["relpath"] for item in saved_images],
        "predicted_property_type": str(analyzed.get("predicted_property_type", "House") or "House"),
        "predicted_condition": str(analyzed.get("predicted_condition", "Good") or "Good"),
        "predicted_style": str(analyzed.get("predicted_style", "Modern") or "Modern"),
        "predicted_environment": str(analyzed.get("predicted_environment", "Suburban") or "Suburban"),
        "predicted_bedrooms": int(float(analyzed.get("predicted_bedrooms", 3) or 3)),
        "confidence": round(confidence, 3),
        "feature_snapshot": {
            "brightness": round(float(features["brightness"]), 3),
            "contrast": round(float(features["contrast"]), 3),
            "green": round(float(features["green"]), 3),
        },
        "prefill": {
            "title": "Uploaded Property",
            "district": "Maseru",
            "locality": "Maseru",
            "price": 1850000,
            "bedrooms": int(float(analyzed.get("predicted_bedrooms", 3) or 3)),
            "property_type": str(analyzed.get("predicted_property_type", "House") or "House"),
            "condition": str(analyzed.get("predicted_condition", "Good") or "Good"),
            "environment": str(analyzed.get("predicted_environment", "Suburban") or "Suburban"),
            "amenities": "parking, road access",
        },
    }


def clear_uploaded_demo_assets(base_dir: Path) -> None:
    """Remove temporary admin demo uploads if needed."""

    uploads_root = base_dir / "generated" / "images" / "uploads" / "admin-demo"
    if uploads_root.exists():
        shutil.rmtree(uploads_root)


def classify_message_tone(message: str, tone: str) -> dict[str, object]:
    """Return a tiny, explainable sentiment/tone summary for the NLP studio."""

    text = message.lower()
    positive_hits = sum(1 for token in ("strong match", "cleaner", "good", "modern", "secure", "welcome") if token in text)
    urgency_hits = sum(1 for token in ("now", "today", "immediately", "featured") if token in text)
    polarity = round(min(0.99, 0.45 + positive_hits * 0.08 + urgency_hits * 0.04), 2)

    if tone == "urgent":
        label = "Promotional / urgent"
    elif tone == "warm":
        label = "Warm / welcoming"
    else:
        label = "Professional / informative"

    return {
        "label": label,
        "score": polarity,
        "positive_hits": positive_hits,
        "urgency_hits": urgency_hits,
    }


def generate_nlp_demo_output(
    *,
    full_name: str,
    title: str,
    district: str,
    locality: str,
    price: int,
    bedrooms: int,
    property_type: str,
    condition: str,
    environment: str,
    amenities: list[str],
    preference_en: str,
    preference_st: str,
    language: str,
    tone: str,
    channel: str,
) -> dict[str, object]:
    """Generate a lightweight marketing message using the existing marketing logic."""

    marketer = MarketingAutomation()
    client = pd.Series(
        {
            "name": full_name or "Admin Demo",
            "preferred_language": language,
            "preferred_channels": [channel],
            "free_text_preference_en": preference_en,
            "free_text_preference_st": preference_st,
        }
    )
    property_row = pd.Series(
        {
            "title": title,
            "district": district,
            "locality": locality,
            "price": price,
            "bedrooms": bedrooms,
            "property_type": property_type,
            "predicted_condition": condition,
            "predicted_environment": environment,
            "amenities": amenities,
        }
    )
    match = SimpleNamespace(
        overall_score=0.91,
        recommendation_reasons=[
            "bedroom layout is a strong fit",
            "text preferences align with the listing highlights",
            "image analysis suggests good condition",
        ],
    )
    message = marketer._build_message(client, property_row, match, language)
    if tone == "warm":
        message = message.rstrip() + (
            " We would be happy to arrange a viewing for you."
            if language == "en"
            else " Re ka thabela ho o hlophisetsa viewing."
        )
    elif tone == "urgent":
        message = (
            "Featured now: " + message
            if language == "en"
            else "Se hlahellang hona jwale: " + message
        )

    title_text = marketer._presentation_title(title, bedrooms=bedrooms, property_type=property_type)
    subject_line = marketer._build_subject_line(
        client=client,
        property_row=property_row,
        display_title=title_text,
        language=language,
        channel=channel,
    )
    preview_text = marketer._build_preview_text(
        property_row=property_row,
        display_title=title_text,
        language=language,
    )
    call_to_action = marketer._build_call_to_action(language, channel)
    tone_summary = classify_message_tone(message, tone)

    return {
        "message": message,
        "subject_line": subject_line,
        "preview_text": preview_text,
        "call_to_action": call_to_action,
        "tone_summary": tone_summary,
        "display_title": title_text,
    }
