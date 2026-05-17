"""Bilingual NLP utilities for the Lesotho property recommender.

The assignment asks for English + Sesotho preference handling, description
enhancement, and recommendation-ready text scoring. This module keeps that work
simple and explainable:
- normalize common bilingual housing phrases into shared tokens
- build TF-IDF-style embeddings locally
- score client/property fit using cosine similarity plus interpretable signals
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "with",
    "near",
    "in",
    "of",
    "to",
    "ya",
    "ke",
    "le",
    "e",
    "ena",
    "tsa",
    "tse",
    "ka",
    "yao",
    "na",
    "ba",
    "bo",
    "di",
    "se",
    "ho",
    "ha",
    "o",
    "i",
    "am",
    "is",
    "are",
    "looking",
    "need",
    "want",
    "batla",
}

TOKEN_ALIASES = {
    "ntlo": "house",
    "house": "house",
    "apartment": "apartment",
    "flat": "apartment",
    "townhouse": "townhouse",
    "dikamore": "bedrooms",
    "kamore": "bedroom",
    "serapa": "garden",
    "lebala": "yard",
    "yard": "yard",
    "garden": "garden",
    "toropo": "urban",
    "haufi": "near",
    "mabenkele": "shops",
    "dikolo": "schools",
    "lelapa": "family",
    "family": "family",
    "parking": "parking",
    "garage": "garage",
    "sireletsehileng": "secure",
    "secure": "secure",
    "hlwekileng": "clean",
    "bohleki": "clean",
    "khutsitseng": "quiet",
    "kgutsitseng": "quiet",
    "quiet": "quiet",
    "kajeno": "modern",
    "modern": "modern",
    "maralleng": "hillside",
    "hillside": "hillside",
    "pono": "view",
    "view": "view",
    "setso": "traditional",
    "traditional": "traditional",
    "suburban": "suburban",
    "urban": "urban",
    "road_access": "road_access",
}

PREFERENCE_SIGNALS = {
    "parking",
    "garage",
    "garden",
    "yard",
    "secure",
    "quiet",
    "modern",
    "traditional",
    "family",
    "view",
    "hillside",
    "urban",
    "suburban",
    "road_access",
}

DISTRICT_NAMES = {
    "maseru",
    "berea",
    "leribe",
    "mafeteng",
    "butha-buthe",
    "quthing",
    "mohaleshoek",
    "mohale'shoek",
    "mohale",
}

PHRASE_NORMALIZATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:1|one)\s*[- ]?\s*bedrooms?\b", re.I), " bedroom_1 "),
    (re.compile(r"\b(?:2|two)\s*[- ]?\s*bedrooms?\b", re.I), " bedroom_2 "),
    (re.compile(r"\b(?:3|three)\s*[- ]?\s*bedrooms?\b", re.I), " bedroom_3 "),
    (re.compile(r"\b(?:4|four)\s*[- ]?\s*bedrooms?\b", re.I), " bedroom_4 "),
    (re.compile(r"\b(?:5|five)\s*[- ]?\s*bedrooms?\b", re.I), " bedroom_5 "),
    (re.compile(r"\bdikamore\s+tse\s+pedi\b", re.I), " bedroom_2 "),
    (re.compile(r"\bdikamore\s+tse\s+peli\b", re.I), " bedroom_2 "),
    (re.compile(r"\bdikamore\s+tse\s+tharo\b", re.I), " bedroom_3 "),
    (re.compile(r"\bdikamore\s+tse\s+nne\b", re.I), " bedroom_4 "),
    (re.compile(r"\bdikamore\s+tse\s+hlano\b", re.I), " bedroom_5 "),
    (re.compile(r"\bparking\s+e\s+sireletsehileng\b", re.I), " parking secure "),
    (re.compile(r"\btikoloho\s+e\s+k?g?h?utsitseng\b", re.I), " quiet "),
    (re.compile(r"\broad\s+access\b", re.I), " road_access "),
)


@dataclass(slots=True)
class TextProcessingResult:
    properties: pd.DataFrame
    clients: pd.DataFrame
    metrics: dict[str, object]


class MultilingualTextProcessor:
    """Local bilingual text processor used by both evaluation and recommendation."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def process(self, properties: pd.DataFrame, clients: pd.DataFrame) -> TextProcessingResult:
        """Prepare embeddings, keywords, and summary metrics for properties and clients."""

        property_df = properties.copy()
        client_df = clients.copy()

        for predicted_column, fallback_column in (
            ("predicted_property_type", "property_type"),
            ("predicted_condition", "condition"),
            ("predicted_style", "style"),
            ("predicted_environment", "environment"),
            ("predicted_bedrooms", "bedrooms"),
        ):
            if predicted_column not in property_df.columns and fallback_column in property_df.columns:
                property_df[predicted_column] = property_df[fallback_column]

        property_df["combined_text"] = property_df.apply(self._build_property_text, axis=1)
        client_df["combined_text"] = client_df.apply(self._build_client_text, axis=1)

        corpus = property_df["combined_text"].tolist() + client_df["combined_text"].tolist()
        matrix = self.fit_transform(corpus)
        property_vectors = matrix[: len(property_df)]
        client_vectors = matrix[len(property_df) :]

        property_df["text_embedding"] = [vector.tolist() for vector in property_vectors]
        client_df["text_embedding"] = [vector.tolist() for vector in client_vectors]
        property_df["property_keywords"] = property_df["combined_text"].map(self.extract_keywords)
        client_df["client_keywords"] = client_df["combined_text"].map(self.extract_keywords)
        property_df["enhanced_description"] = property_df.apply(self._build_enhanced_description, axis=1)

        metrics = self._build_sanity_metrics(property_df)
        return TextProcessingResult(property_df, client_df, metrics)

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        """Fit a light TF-IDF representation and return normalized vectors."""

        tokenized = [self.tokenize(text) for text in texts]
        vocabulary: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                vocabulary.setdefault(token, len(vocabulary))
        self.vocabulary = vocabulary

        term_matrix = np.zeros((len(tokenized), len(vocabulary)), dtype=float)
        doc_frequency = np.zeros(len(vocabulary), dtype=float)

        for row_index, tokens in enumerate(tokenized):
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                column = vocabulary[token]
                term_matrix[row_index, column] = count / len(tokens)
                doc_frequency[column] += 1.0

        self.idf = np.array(
            [math.log((1.0 + len(tokenized)) / (1.0 + count)) + 1.0 for count in doc_frequency],
            dtype=float,
        )
        matrix = term_matrix * self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Project new text into the already-fit vocabulary."""

        if self.idf is None or not self.vocabulary:
            raise RuntimeError("The text processor must be fit before transform is called.")

        tokenized = [self.tokenize(text) for text in texts]
        term_matrix = np.zeros((len(tokenized), len(self.vocabulary)), dtype=float)
        for row_index, tokens in enumerate(tokenized):
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                if token in self.vocabulary:
                    counts[token] = counts.get(token, 0) + 1
            total = max(sum(counts.values()), 1)
            for token, count in counts.items():
                column = self.vocabulary[token]
                term_matrix[row_index, column] = count / total

        matrix = term_matrix * self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def tokenize(self, text: str) -> list[str]:
        """Normalize English/Sesotho housing phrases into shared tokens."""

        normalized_text = self._normalize_text_for_tokenization(text)
        base_tokens = re.findall(r"[a-z0-9_']+", normalized_text)
        tokens: list[str] = []
        for token in base_tokens:
            normalized = token.strip("'").strip()
            if not normalized or normalized in STOPWORDS:
                continue
            tokens.append(normalized)
            alias = TOKEN_ALIASES.get(normalized)
            if alias and alias != normalized:
                tokens.append(alias)
            if normalized in DISTRICT_NAMES:
                tokens.append(f"district_{self._normalize_key_fragment(normalized)}")
        return tokens

    def extract_keywords(self, text: str, limit: int = 10) -> list[str]:
        """Return the highest-frequency normalized terms from a piece of text."""

        counts: dict[str, int] = {}
        for token in self.tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [token for token, _ in ranked[:limit]]

    def cosine_similarity(self, left: list[float], right: list[float]) -> float:
        left_vector = np.asarray(left, dtype=float)
        right_vector = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left_vector, right_vector) / denominator)

    def score_client_property(self, client_row, property_row) -> dict[str, object]:
        """Explainable text score used by the matching engine."""

        client_embedding = self._coerce_vector(self._row_value(client_row, "text_embedding", []))
        property_embedding = self._coerce_vector(self._row_value(property_row, "text_embedding", []))
        if not client_embedding or not property_embedding:
            transformed = self.transform(
                [
                    self._build_client_text(self._coerce_series(client_row)),
                    self._build_property_text(self._coerce_series(property_row)),
                ]
            )
            client_embedding = transformed[0].tolist()
            property_embedding = transformed[1].tolist()

        cosine = self.cosine_similarity(client_embedding, property_embedding)
        client_keywords = set(self._coerce_list(self._row_value(client_row, "client_keywords", [])))
        property_keywords = set(self._coerce_list(self._row_value(property_row, "property_keywords", [])))
        shared_keywords = sorted(client_keywords & property_keywords)
        keyword_overlap = len(shared_keywords) / max(len(client_keywords), 1)
        signal_alignment = self._signal_alignment(client_row, property_row, client_keywords, property_keywords)

        score = 0.55 * cosine + 0.25 * keyword_overlap + 0.20 * signal_alignment
        return {
            "score": round(float(score), 4),
            "cosine": round(float(cosine), 4),
            "keyword_overlap": round(float(keyword_overlap), 4),
            "signal_alignment": round(float(signal_alignment), 4),
            "shared_keywords": shared_keywords,
        }

    @staticmethod
    def _build_property_text(row: pd.Series) -> str:
        amenities = " ".join(MultilingualTextProcessor._coerce_list(row.get("amenities", [])))
        property_type = str(row.get("predicted_property_type", row.get("property_type", "")))
        style = str(row.get("predicted_style", row.get("style", "")))
        environment = str(row.get("predicted_environment", row.get("environment", "")))
        bedrooms = MultilingualTextProcessor._bedroom_token(row.get("predicted_bedrooms", row.get("bedrooms", 0)))
        district = str(row.get("district", ""))
        district_token = (
            f"district_{MultilingualTextProcessor._normalize_key_fragment(district)}" if district else ""
        )
        return " ".join(
            [
                str(row.get("title", "")),
                str(row.get("description_en", "")),
                str(row.get("description_st", "")),
                district,
                district_token,
                str(row.get("location_text", "")),
                property_type,
                style,
                environment,
                bedrooms,
                amenities,
            ]
        )

    @staticmethod
    def _build_client_text(row: pd.Series) -> str:
        districts = " ".join(str(item) for item in MultilingualTextProcessor._coerce_list(row.get("preferred_districts", [])))
        district_tokens = " ".join(
            f"district_{MultilingualTextProcessor._normalize_key_fragment(str(item))}"
            for item in MultilingualTextProcessor._coerce_list(row.get("preferred_districts", []))
        )
        property_types = " ".join(str(item) for item in MultilingualTextProcessor._coerce_list(row.get("preferred_property_types", [])))
        property_type_tokens = " ".join(
            f"property_type_{str(item).lower()}" for item in MultilingualTextProcessor._coerce_list(row.get("preferred_property_types", []))
        )
        bedroom_token = MultilingualTextProcessor._bedroom_token(row.get("preferred_bedrooms", 0))
        return " ".join(
            [
                str(row.get("free_text_preference_en", "")),
                str(row.get("free_text_preference_st", "")),
                districts,
                district_tokens,
                property_types,
                property_type_tokens,
                bedroom_token,
            ]
        )

    @staticmethod
    def _build_enhanced_description(row: pd.Series) -> str:
        amenities = ", ".join(MultilingualTextProcessor._coerce_list(row.get("amenities", [])))
        condition = row.get("predicted_condition", row.get("condition", "Good"))
        environment = row.get("predicted_environment", row.get("environment", "Suburban"))
        return (
            f"{row.get('title', 'Property')} | {row.get('district', 'Unknown district')} | "
            f"{int(float(row.get('bedrooms', 0) or 0))} bedrooms, "
            f"{int(float(row.get('bathrooms', 0) or 0))} bathrooms, {condition} condition, "
            f"{environment} environment. Key features: {amenities}."
        )

    def _build_sanity_metrics(self, property_df: pd.DataFrame) -> dict[str, object]:
        """Run lightweight bilingual checks so we can report NLP quality."""

        if property_df.empty:
            return {
                "vocabulary_size": len(self.vocabulary),
                "avg_property_keyword_count": 0.0,
                "query_success_rate": 0.0,
                "query_cases": [],
            }

        query_cases = [
            {
                "name": "English family query",
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "Looking for a modern family house in Maseru with secure parking and a yard.",
                "free_text_preference_st": "",
                "preferred_language": "en",
                "expected_district": "Maseru",
            },
            {
                "name": "Sesotho family query",
                "preferred_districts": ["Maseru"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 3,
                "free_text_preference_en": "",
                "free_text_preference_st": "Ke batla ntlo ya lelapa Maseru e nang le parking e sireletsehileng le serapa.",
                "preferred_language": "st",
                "expected_district": "Maseru",
            },
            {
                "name": "Berea garden query",
                "preferred_districts": ["Berea"],
                "preferred_property_types": ["House"],
                "preferred_bedrooms": 2,
                "free_text_preference_en": "Need a 2 bedroom house in Berea with a garden and road access.",
                "free_text_preference_st": "",
                "preferred_language": "en",
                "expected_district": "Berea",
            },
        ]

        successful = 0
        evaluated_cases: list[dict[str, object]] = []
        for case in query_cases:
            client_series = pd.Series(
                {
                    "preferred_districts": case["preferred_districts"],
                    "preferred_property_types": case["preferred_property_types"],
                    "preferred_bedrooms": case["preferred_bedrooms"],
                    "free_text_preference_en": case["free_text_preference_en"],
                    "free_text_preference_st": case["free_text_preference_st"],
                }
            )
            client_series["combined_text"] = self._build_client_text(client_series)
            client_series["text_embedding"] = self.transform([client_series["combined_text"]])[0].tolist()
            client_series["client_keywords"] = self.extract_keywords(client_series["combined_text"])

            scored_rows = [
                (
                    self.score_client_property(client_series, property_row),
                    property_row,
                )
                for property_row in property_df.itertuples(index=False)
            ]
            best_score, best_property = max(scored_rows, key=lambda item: item[0]["score"])
            district_match = str(best_property.district) == str(case["expected_district"])
            successful += int(district_match)
            evaluated_cases.append(
                {
                    "query_name": case["name"],
                    "expected_district": case["expected_district"],
                    "top_property_id": str(best_property.property_id),
                    "top_district": str(best_property.district),
                    "top_score": best_score["score"],
                    "success": district_match,
                }
            )

        property_keyword_lengths = property_df["property_keywords"].map(len) if "property_keywords" in property_df else pd.Series(dtype=float)
        english_case = evaluated_cases[0] if len(evaluated_cases) > 0 else {}
        sesotho_case = evaluated_cases[1] if len(evaluated_cases) > 1 else {}
        return {
            "vocabulary_size": int(len(self.vocabulary)),
            "avg_property_keyword_count": round(float(property_keyword_lengths.mean()) if not property_keyword_lengths.empty else 0.0, 2),
            "query_success_rate": round(successful / max(len(query_cases), 1), 3),
            "query_cases": evaluated_cases,
            "english_top_property_id": english_case.get("top_property_id"),
            "english_top_district": english_case.get("top_district"),
            "sesotho_top_property_id": sesotho_case.get("top_property_id"),
            "sesotho_top_district": sesotho_case.get("top_district"),
            "english_query_matches_maseru": bool(english_case.get("success", False)),
            "sesotho_query_matches_maseru": bool(sesotho_case.get("success", False)),
        }

    def _signal_alignment(
        self,
        client_row,
        property_row,
        client_keywords: set[str],
        property_keywords: set[str],
    ) -> float:
        """Capture interpretable preference overlap beyond raw cosine similarity."""

        client_signals = {token for token in client_keywords if token in PREFERENCE_SIGNALS or token.startswith(("bedroom_", "district_", "property_type_"))}
        property_signals = {token for token in property_keywords if token in PREFERENCE_SIGNALS or token.startswith(("bedroom_", "district_"))}
        property_signals.add(self._bedroom_token(self._row_value(property_row, "predicted_bedrooms", self._row_value(property_row, "bedrooms", 0))))
        property_signals.add(
            f"property_type_{str(self._row_value(property_row, 'predicted_property_type', self._row_value(property_row, 'property_type', 'house'))).lower()}"
        )
        district = str(self._row_value(property_row, "district", "")).lower().replace(" ", "").replace("-", "").replace("'", "")
        if district:
            property_signals.add(f"district_{district}")

        client_districts = [str(item) for item in self._coerce_list(self._row_value(client_row, "preferred_districts", []))]
        client_types = [str(item) for item in self._coerce_list(self._row_value(client_row, "preferred_property_types", []))]
        preferred_bedrooms = self._coerce_int(self._row_value(client_row, "preferred_bedrooms", 0))
        if client_districts:
            client_signals.update(
                f"district_{self._normalize_key_fragment(item)}" for item in client_districts
            )
        if client_types:
            client_signals.update(f"property_type_{item.lower()}" for item in client_types)
        if preferred_bedrooms:
            client_signals.add(self._bedroom_token(preferred_bedrooms))

        shared = client_signals & property_signals
        signal_score = len(shared) / max(len(client_signals), 1)

        property_amenities = set(self._coerce_list(self._row_value(property_row, "amenities", [])))
        amenity_bonus = len(client_keywords & property_amenities) / max(len(property_amenities), 1) if property_amenities else 0.0
        return min(1.0, 0.8 * signal_score + 0.2 * amenity_bonus)

    @staticmethod
    def _normalize_text_for_tokenization(text: str) -> str:
        normalized = str(text or "").lower()
        normalized = normalized.replace("mohale's hoek", "mohaleshoek")
        normalized = normalized.replace("mohale s hoek", "mohaleshoek")
        for pattern, replacement in PHRASE_NORMALIZATIONS:
            normalized = pattern.sub(replacement, normalized)
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9_'\s]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _bedroom_token(value: object) -> str:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            return ""
        return f"bedroom_{min(max(numeric, 0), 5)}" if numeric > 0 else ""

    @staticmethod
    def _row_value(row, key: str, default=None):
        if isinstance(row, pd.Series):
            return row.get(key, default)
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    @staticmethod
    def _coerce_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [part.strip() for part in re.split(r"[|,]", stripped) if part.strip()]
        if value is None:
            return []
        return [str(value).strip()]

    @staticmethod
    def _coerce_int(value: object) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _coerce_vector(value: object) -> list[float]:
        if isinstance(value, list):
            return [float(item) for item in value]
        return []

    @staticmethod
    def _coerce_series(row) -> pd.Series:
        if isinstance(row, pd.Series):
            return row
        if isinstance(row, dict):
            return pd.Series(row)
        if hasattr(row, "_asdict"):
            return pd.Series(row._asdict())
        return pd.Series()

    @staticmethod
    def _normalize_key_fragment(value: object) -> str:
        return str(value or "").lower().replace(" ", "").replace("-", "").replace("'", "")
