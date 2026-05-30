"""Multimodal ranking logic for matching buyers to properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lesotho_property_ai.nlp import MultilingualTextProcessor


@dataclass(slots=True)
class MatchingWeights:
    structured: float = 0.45
    text: float = 0.35
    vision: float = 0.20


class MatchingEngine:
    def __init__(self, text_processor: MultilingualTextProcessor, weights: MatchingWeights) -> None:
        self.text_processor = text_processor
        self.weights = weights

    def rank_for_all_clients(
        self,
        properties: pd.DataFrame,
        clients: pd.DataFrame,
        top_n: int = 3,
        constraint_mode: str = "soft",
    ) -> pd.DataFrame:
        """Score every client/property pair and keep only the top N results."""

        records: list[dict[str, object]] = []
        for client in clients.itertuples(index=False):
            structured_fallback = self._use_structured_fallback(client)
            scored = []
            for property_row in properties.itertuples(index=False):
                if constraint_mode == "strict" and not self._passes_strict_constraints(client, property_row):
                    continue
                if constraint_mode == "near" and not self._passes_near_constraints(client, property_row):
                    continue
                if structured_fallback and self._is_over_budget(client, property_row):
                    continue
                structured_details = self._structured_score(client, property_row)
                vision_details = self._vision_score(client, property_row)
                if structured_fallback:
                    text_details = {
                        "score": 0.0,
                        "cosine": 0.0,
                        "keyword_overlap": 0.0,
                        "signal_alignment": 0.0,
                        "shared_keywords": [],
                    }
                    text_reliability = 0.0
                    structured = self._structured_fallback_score(
                        client=client,
                        property_row=property_row,
                        structured_details=structured_details,
                        vision_details=vision_details,
                    )
                    text = 0.0
                    vision = float(vision_details["score"])
                    weights_used = self._structured_fallback_weights(
                        vision_reliability=float(vision_details["reliability"]),
                    )
                else:
                    text_details = self.text_processor.score_client_property(client, property_row)
                    structured = float(structured_details["score"])
                    text = float(text_details["score"])
                    vision = float(vision_details["score"])
                    text_reliability = self._text_reliability(client, text_details)
                    weights_used = self._effective_weights(
                        structured_reliability=float(structured_details["reliability"]),
                        text_reliability=text_reliability,
                        vision_reliability=float(vision_details["reliability"]),
                    )

                overall = (
                    weights_used["structured"] * structured
                    + weights_used["text"] * text
                    + weights_used["vision"] * vision
                )
                fusion_reliability = round(
                    float(
                        weights_used["structured"] * float(structured_details["reliability"])
                        + weights_used["text"] * text_reliability
                        + weights_used["vision"] * float(vision_details["reliability"])
                    ),
                    4,
                )
                shared_keywords = text_details.get("shared_keywords", [])
                shared_keywords_humanized = [
                    self._humanize_keyword(token) for token in shared_keywords if self._humanize_keyword(token)
                ]
                reasons = self._build_recommendation_reasons(
                    client=client,
                    property_row=property_row,
                    structured_details=structured_details,
                    text_details=text_details,
                    vision_details=vision_details,
                    shared_keywords_humanized=shared_keywords_humanized,
                    structured_fallback=structured_fallback,
                )
                explanation = self._build_explanation(
                    property_row=property_row,
                    structured=structured,
                    text=text,
                    vision=vision,
                    weights_used=weights_used,
                    reasons=reasons,
                    structured_fallback=structured_fallback,
                )
                scored.append(
                    {
                        "client_id": client.client_id,
                        "client_name": client.name,
                        "property_id": property_row.property_id,
                        "property_title": property_row.title,
                        "district": property_row.district,
                        "price": property_row.price,
                        "structured_score": round(structured, 4),
                        "structured_budget_score": round(float(structured_details["budget_score"]), 4),
                        "structured_district_score": round(float(structured_details["district_score"]), 4),
                        "structured_type_score": round(float(structured_details["type_score"]), 4),
                        "structured_bedroom_score": round(float(structured_details["bedroom_score"]), 4),
                        "text_score": round(text, 4),
                        "text_cosine": text_details["cosine"],
                        "text_keyword_overlap": text_details["keyword_overlap"],
                        "text_signal_alignment": text_details["signal_alignment"],
                        "shared_text_cues": shared_keywords_humanized,
                        "vision_score": round(vision, 4),
                        "vision_bedroom_alignment": round(float(vision_details["bedroom_alignment"]), 4),
                        "vision_type_alignment": round(float(vision_details["type_alignment"]), 4),
                        "vision_environment_alignment": round(float(vision_details["environment_alignment"]), 4),
                        "vision_style_alignment": round(float(vision_details["style_alignment"]), 4),
                        "vision_condition_alignment": round(float(vision_details["condition_alignment"]), 4),
                        "fusion_reliability": fusion_reliability,
                        "structured_weight_used": round(float(weights_used["structured"]), 4),
                        "text_weight_used": round(float(weights_used["text"]), 4),
                        "vision_weight_used": round(float(weights_used["vision"]), 4),
                        "recommendation_reasons": reasons,
                        "overall_score": round(overall, 4),
                        "recommended_language": client.preferred_language,
                        "explanation": explanation,
                    }
                )
            scored = sorted(scored, key=lambda item: item["overall_score"], reverse=True)[:top_n]
            for rank, item in enumerate(scored, start=1):
                item["rank"] = rank
                records.append(item)
        if not records:
            return pd.DataFrame(
                columns=[
                    "client_id",
                    "client_name",
                    "property_id",
                    "property_title",
                    "district",
                    "price",
                    "rank",
                    "overall_score",
                    "recommendation_reasons",
                    "explanation",
                ]
            )
        return pd.DataFrame(records).sort_values(["client_id", "rank"]).reset_index(drop=True)

    @staticmethod
    def _use_structured_fallback(client) -> bool:
        return not any(
            [
                str(getattr(client, "free_text_preference_en", "") or "").strip(),
                str(getattr(client, "free_text_preference_st", "") or "").strip(),
            ]
        )

    @staticmethod
    def _is_over_budget(client, property_row) -> bool:
        try:
            budget_max = float(getattr(client, "budget_max", 0) or 0)
            price = float(getattr(property_row, "price", 0) or 0)
        except (TypeError, ValueError):
            return False
        return budget_max > 0 and price > budget_max

    @staticmethod
    def _passes_budget_and_district(client, property_row) -> bool:
        try:
            budget_min = float(getattr(client, "budget_min", 0) or 0)
            budget_max = float(getattr(client, "budget_max", 0) or 0)
            price = float(getattr(property_row, "price", 0) or 0)
        except (TypeError, ValueError):
            return False
        if budget_max <= 0 or price > budget_max:
            return False
        if budget_min > 0 and price < budget_min:
            return False

        preferred_districts = {
            str(item).strip().lower()
            for item in MatchingEngine._coerce_iterable(getattr(client, "preferred_districts", []))
            if str(item).strip()
        }
        district = str(getattr(property_row, "district", "") or "").strip().lower()
        if preferred_districts and district not in preferred_districts:
            return False
        return True

    @staticmethod
    def _passes_strict_constraints(client, property_row) -> bool:
        if not MatchingEngine._passes_budget_and_district(client, property_row):
            return False
        preferred_bedrooms = int(float(getattr(client, "preferred_bedrooms", 0) or 0))
        bedrooms = int(float(getattr(property_row, "bedrooms", 0) or 0))
        return preferred_bedrooms <= 0 or bedrooms == preferred_bedrooms

    @staticmethod
    def _passes_near_constraints(client, property_row) -> bool:
        if not MatchingEngine._passes_budget_and_district(client, property_row):
            return False
        preferred_bedrooms = int(float(getattr(client, "preferred_bedrooms", 0) or 0))
        bedrooms = int(float(getattr(property_row, "bedrooms", 0) or 0))
        return preferred_bedrooms > 0 and bedrooms != preferred_bedrooms

    @staticmethod
    def _structured_fallback_score(
        *,
        client,
        property_row,
        structured_details: dict[str, float],
        vision_details: dict[str, float],
    ) -> float:
        amenities = {
            str(item).strip().lower()
            for item in MatchingEngine._coerce_iterable(getattr(property_row, "amenities", []))
        }
        amenity_support = 0.0
        for preferred in ("parking", "garage", "garden", "yard", "road access"):
            if preferred in amenities:
                amenity_support = max(amenity_support, 0.8)
                break

        support_score = max(
            amenity_support,
            float(vision_details["environment_alignment"]),
            float(vision_details["style_alignment"]) * 0.8,
            float(vision_details["condition_alignment"]) * 0.75,
        )
        score = (
            0.34 * float(structured_details["district_score"])
            + 0.24 * float(structured_details["bedroom_score"])
            + 0.20 * float(structured_details["type_score"])
            + 0.14 * float(structured_details["budget_score"])
            + 0.08 * support_score
        )
        return round(float(score), 4)

    def _structured_fallback_weights(self, *, vision_reliability: float) -> dict[str, float]:
        weighted = {
            "structured": 0.84,
            "text": 0.0,
            "vision": 0.16 * max(vision_reliability, 0.35),
        }
        total = sum(weighted.values()) or 1.0
        return {name: float(value / total) for name, value in weighted.items()}

    @staticmethod
    def _structured_score(client, property_row) -> dict[str, float]:
        """Score explicit preferences like budget, district, type, and bedrooms."""

        midpoint = (client.budget_min + client.budget_max) / 2
        if client.budget_min <= property_row.price <= client.budget_max:
            budget_score = 1.0 - min(abs(property_row.price - midpoint) / max(midpoint, 1), 0.45)
        elif property_row.price < client.budget_min:
            budget_score = max(0.0, 1.0 - (client.budget_min - property_row.price) / max(client.budget_min, 1))
        else:
            budget_score = max(0.0, 1.0 - (property_row.price - client.budget_max) / max(client.budget_max, 1))

        district_score = 1.0 if property_row.district in client.preferred_districts else 0.35
        predicted_type = getattr(property_row, "predicted_property_type", property_row.property_type)
        type_score = 1.0 if predicted_type in client.preferred_property_types else 0.4
        bedroom_score = max(0.0, 1.0 - abs(property_row.bedrooms - client.preferred_bedrooms) / 4.0)
        overall_score = float(
            0.40 * budget_score + 0.25 * district_score + 0.20 * type_score + 0.15 * bedroom_score
        )
        reliability_inputs = [
            float(client.budget_max > client.budget_min),
            float(bool(client.preferred_districts)),
            float(bool(client.preferred_property_types)),
            float(int(getattr(client, "preferred_bedrooms", 0) or 0) > 0),
        ]
        reliability = 0.60 + 0.40 * float(np.mean(reliability_inputs))
        return {
            "score": round(overall_score, 4),
            "budget_score": round(float(budget_score), 4),
            "district_score": round(float(district_score), 4),
            "type_score": round(float(type_score), 4),
            "bedroom_score": round(float(bedroom_score), 4),
            "reliability": round(float(min(1.0, reliability)), 4),
        }

    @staticmethod
    def _vision_score(client, property_row) -> dict[str, float]:
        """Use vision predictions as a softer preference signal, not a hard filter."""

        bedroom_alignment = max(
            0.0, 1.0 - abs(property_row.predicted_bedrooms - client.preferred_bedrooms) / 4.0
        )
        type_alignment = (
            1.0 if property_row.predicted_property_type in client.preferred_property_types else 0.45
        )
        keywords = set(client.client_keywords)
        environment_alignment = 0.75
        if {"garden", "quiet", "family"} & keywords:
            environment_alignment = (
                1.0 if property_row.predicted_environment in {"Garden", "Suburban"} else 0.55
            )
        elif {"urban", "transport", "town"} & keywords:
            environment_alignment = 1.0 if property_row.predicted_environment == "Urban" else 0.55
        elif {"hillside", "view"} & keywords:
            environment_alignment = 1.0 if property_row.predicted_environment == "Hillside" else 0.55

        style_alignment = 0.70
        if "modern" in keywords:
            style_alignment = 1.0 if property_row.predicted_style == "Modern" else 0.45
        elif "traditional" in keywords:
            style_alignment = 1.0 if property_row.predicted_style == "Traditional" else 0.45
        elif "family" in keywords:
            style_alignment = 1.0 if property_row.predicted_style == "Family" else 0.60

        condition_alignment = 0.70
        if {"clean", "modern", "secure", "new"} & keywords:
            condition_alignment = 1.0 if property_row.predicted_condition in {"New", "Good"} else 0.55

        confidence = float(property_row.vision_confidence)
        score = float(
            0.30 * bedroom_alignment
            + 0.25 * type_alignment
            + 0.20 * environment_alignment
            + 0.15 * style_alignment
            + 0.10 * condition_alignment
        )
        availability = np.mean(
            [
                float(bool(str(getattr(property_row, "predicted_property_type", "")).strip())),
                float(bool(str(getattr(property_row, "predicted_style", "")).strip())),
                float(bool(str(getattr(property_row, "predicted_environment", "")).strip())),
                float(bool(str(getattr(property_row, "predicted_condition", "")).strip())),
                float(float(getattr(property_row, "predicted_bedrooms", 0) or 0) > 0),
            ]
        )
        reliability = min(1.0, 0.40 + 0.40 * confidence + 0.20 * float(availability))
        return {
            "score": round(score, 4),
            "bedroom_alignment": round(float(bedroom_alignment), 4),
            "type_alignment": round(float(type_alignment), 4),
            "environment_alignment": round(float(environment_alignment), 4),
            "style_alignment": round(float(style_alignment), 4),
            "condition_alignment": round(float(condition_alignment), 4),
            "reliability": round(float(reliability), 4),
        }

    def _text_reliability(self, client, text_details: dict[str, Any]) -> float:
        """Estimate how informative the buyer text is before fusion reweighting."""

        client_keywords = MatchingEngine._coerce_iterable(getattr(client, "client_keywords", []))
        free_text = " ".join(
            [
                str(getattr(client, "free_text_preference_en", "") or ""),
                str(getattr(client, "free_text_preference_st", "") or ""),
            ]
        ).strip()
        free_text_tokens = self.text_processor.tokenize(free_text) if free_text else []
        keyword_coverage = min(len(client_keywords) / 6.0, 1.0)
        text_richness = min(len(set(free_text_tokens)) / 10.0, 1.0)
        shared_keyword_strength = min(len(text_details.get("shared_keywords", [])) / 4.0, 1.0)
        reliability = (
            0.42
            + 0.14 * keyword_coverage
            + 0.10 * text_richness
            + 0.14 * shared_keyword_strength
            + 0.10 * float(text_details.get("keyword_overlap", 0.0))
            + 0.10 * float(text_details.get("signal_alignment", 0.0))
        )
        return round(float(min(1.0, reliability)), 4)

    def _effective_weights(
        self,
        *,
        structured_reliability: float,
        text_reliability: float,
        vision_reliability: float,
    ) -> dict[str, float]:
        """Rebalance the base weights so weak evidence contributes less aggressively."""

        weighted = {
            "structured": self.weights.structured * max(structured_reliability, 0.25),
            "text": self.weights.text * max(text_reliability, 0.25),
            "vision": self.weights.vision * max(vision_reliability, 0.25),
        }
        total = sum(weighted.values()) or (
            self.weights.structured + self.weights.text + self.weights.vision
        )
        return {name: float(value / total) for name, value in weighted.items()}

    def _build_recommendation_reasons(
        self,
        *,
        client,
        property_row,
        structured_details: dict[str, float],
        text_details: dict[str, Any],
        vision_details: dict[str, float],
        shared_keywords_humanized: list[str],
        structured_fallback: bool,
    ) -> list[str]:
        """Turn numeric fusion evidence into short human-friendly reasons."""

        candidates: list[tuple[float, str]] = []
        preferred_types = {
            str(item).strip().lower() for item in self._coerce_iterable(getattr(client, "preferred_property_types", []))
        }
        if structured_fallback:
            candidates.append((1.0, "structured fallback respected the budget ceiling"))
        if structured_details["budget_score"] >= 0.82:
            candidates.append((structured_details["budget_score"], "budget is closely aligned"))
        if structured_details["district_score"] >= 0.95:
            candidates.append((structured_details["district_score"], "it matches the preferred district"))
        if structured_details["type_score"] >= 0.95 and (
            len(preferred_types) > 1
            or str(property_row.predicted_property_type).strip().lower() != "house"
        ):
            candidates.append((structured_details["type_score"], "property type matches the buyer preference"))
        if max(structured_details["bedroom_score"], vision_details["bedroom_alignment"]) >= 0.82:
            candidates.append((max(structured_details["bedroom_score"], vision_details["bedroom_alignment"]), "bedroom layout is a strong fit"))
        if (
            not structured_fallback
            and
            shared_keywords_humanized
            and (
                float(text_details.get("keyword_overlap", 0.0)) >= 0.12
                or float(text_details.get("signal_alignment", 0.0)) >= 0.18
            )
        ):
            candidates.append(
                (
                    max(
                        float(text_details.get("keyword_overlap", 0.0)),
                        float(text_details.get("signal_alignment", 0.0)),
                    ),
                    f"text preferences align on {', '.join(shared_keywords_humanized[:3])}",
                )
            )
        if vision_details["environment_alignment"] >= 0.9:
            candidates.append(
                (
                    vision_details["environment_alignment"],
                    f"vision model supports the {str(property_row.predicted_environment).lower()} setting",
                )
            )
        if vision_details["style_alignment"] >= 0.9:
            candidates.append(
                (
                    vision_details["style_alignment"],
                    f"vision model supports the {str(property_row.predicted_style).lower()} style",
                )
            )
        if vision_details["condition_alignment"] >= 0.9:
            candidates.append(
                (
                    vision_details["condition_alignment"],
                    f"image analysis suggests {str(property_row.predicted_condition).lower()} condition",
                )
            )

        if not candidates:
            candidates.append((0.5, "the combined structured, text, and image signals are reasonably aligned"))

        ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
        reasons: list[str] = []
        for _, reason in ordered:
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) == 3:
                break
        return reasons

    def _build_explanation(
        self,
        *,
        property_row,
        structured: float,
        text: float,
        vision: float,
        weights_used: dict[str, float],
        reasons: list[str],
        structured_fallback: bool,
    ) -> str:
        reason_sentence = "; ".join(reasons)
        property_type = str(getattr(property_row, "predicted_property_type", getattr(property_row, "property_type", "house"))).strip().lower() or "house"
        district = str(getattr(property_row, "district", "")).strip() or "the target district"
        if structured_fallback:
            return (
                f"Structured fallback mode was used for this {property_type} in {district} because the buyer left both "
                f"description fields blank. Ranking prioritized the preferred district, bedroom fit, property type, and "
                f"budget-safe inventory, with image evidence used only as light support. Reasons: {reason_sentence}. "
                f"Component scores: structured {structured:.2f}, text {text:.2f}, vision {vision:.2f}."
            )
        return (
            f"Strong {property_type} match in {district} because {reason_sentence}. "
            f"Fusion score used structured {weights_used['structured']:.2f}, text {weights_used['text']:.2f}, "
            f"and vision {weights_used['vision']:.2f} weights after confidence rebalancing. "
            f"Component scores: structured {structured:.2f}, text {text:.2f}, vision {vision:.2f}."
        )

    @staticmethod
    def _humanize_keyword(token: str) -> str:
        """Turn internal tokens like `bedroom_3` into display-friendly text."""

        cleaned = str(token or "").strip().replace("_", " ")
        if not cleaned:
            return ""
        if cleaned.startswith("bedroom "):
            return cleaned.replace("bedroom ", "") + " bedrooms"
        if cleaned.startswith("bathroom "):
            return cleaned.replace("bathroom ", "") + " bathrooms"
        if cleaned.startswith("district "):
            return cleaned.replace("district ", "")
        if cleaned.startswith("environment "):
            return cleaned.replace("environment ", "")
        if cleaned.startswith("style "):
            return cleaned.replace("style ", "")
        return cleaned

    @staticmethod
    def _coerce_iterable(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            cleaned = cleaned.strip("[]")
            return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip()]
        return []
