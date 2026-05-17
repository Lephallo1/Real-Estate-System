"""Presentation-friendly marketing message generation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd


class MarketingAutomation:
    def generate(
        self,
        matches: pd.DataFrame,
        properties: pd.DataFrame,
        clients: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create one top-message campaign row per client."""

        top_matches = matches.sort_values(["client_id", "rank"]).groupby("client_id").head(1)
        property_lookup = properties.set_index("property_id")
        client_lookup = clients.set_index("client_id")

        records: list[dict[str, object]] = []
        for match in top_matches.itertuples(index=False):
            client = client_lookup.loc[match.client_id]
            property_row = property_lookup.loc[match.property_id]
            language = client["preferred_language"]
            channel = client["preferred_channels"][0] if client["preferred_channels"] else "email"
            display_title = self._presentation_title(
                property_row.get("title", ""),
                bedrooms=property_row.get("bedrooms", 0),
                property_type=property_row.get("property_type", "House"),
            )
            message = self._build_message(client, property_row, match, language)
            recommendation_reasons = self._coerce_list(getattr(match, "recommendation_reasons", []))
            subject_line = self._build_subject_line(
                client=client,
                property_row=property_row,
                display_title=display_title,
                language=language,
                channel=channel,
            )
            preview_text = self._build_preview_text(
                property_row=property_row,
                display_title=display_title,
                language=language,
            )
            call_to_action = self._build_call_to_action(language, channel)
            estimated_engagement = self._estimate_engagement_score(
                match_score=float(match.overall_score),
                channel=channel,
                recommendation_reasons=recommendation_reasons,
            )
            records.append(
                {
                    "campaign_id": f"CAMP-{match.client_id}-{match.property_id}",
                    "client_id": match.client_id,
                    "client_name": client["name"],
                    "property_id": match.property_id,
                    "property_title": display_title,
                    "channel": channel,
                    "language": language,
                    "campaign_variant": self._campaign_variant(channel),
                    "subject_line": subject_line,
                    "preview_text": preview_text,
                    "message": message,
                    "call_to_action": call_to_action,
                    "recommendation_reasons": recommendation_reasons,
                    "status": "simulated_sent",
                    "delivery_state": "delivered_to_demo_queue",
                    "recommended_send_window": self._recommended_send_window(channel),
                    "sent_at_utc": datetime.now(timezone.utc).isoformat(),
                    "match_score": float(match.overall_score),
                    "estimated_engagement_score": estimated_engagement,
                }
            )
        return pd.DataFrame(records)

    @staticmethod
    def _build_message(client: pd.Series, property_row: pd.Series, match, language: str) -> str:
        """Generate a lightweight bilingual message from the best match."""

        title = MarketingAutomation._presentation_title(
            property_row.get("title", ""),
            bedrooms=property_row.get("bedrooms", 0),
            property_type=property_row.get("property_type", "House"),
        )
        district = str(property_row.get("district", "") or "").strip()
        locality = str(property_row.get("locality", "") or "").strip()
        place_text = locality if locality and locality.lower() != district.lower() else district
        price_text = MarketingAutomation._format_currency(property_row.get("price"))
        condition = str(
            property_row.get("predicted_condition", property_row.get("condition", "Good")) or "Good"
        ).strip()
        environment = str(
            property_row.get("predicted_environment", property_row.get("environment", "Suburban"))
            or "Suburban"
        ).strip()
        recommendation_reasons = MarketingAutomation._coerce_list(getattr(match, "recommendation_reasons", []))
        reason_sentence_en = MarketingAutomation._reason_sentence_english(recommendation_reasons)
        highlights_en = MarketingAutomation._highlights_english(property_row, client)
        highlights_st = MarketingAutomation._highlights_sesotho(property_row, client)

        if language == "st":
            return (
                f"Lumela {client['name']}, re fumane {title} ho {place_text} ka theko ya {price_text}. "
                f"E na le dikamore tse {int(property_row['bedrooms'])}, e boemong ba {MarketingAutomation._condition_st(condition)}, "
                f"mme e le {MarketingAutomation._environment_st(environment)}. {highlights_st} "
                f"Match score ya yona ke {match.overall_score:.2f}. Re ka o romella lintlha tse ding kapa ra hlophisa viewing."
            )
        return (
            f"Hi {client['name']}, we found a strong match for you: {title} in {place_text} priced at {price_text}. "
            f"It offers {int(property_row['bedrooms'])} bedrooms, is in {MarketingAutomation._environment_en(environment)}, "
            f"and is in {condition.lower()} condition. {reason_sentence_en}{highlights_en} "
            f"Match score: {match.overall_score:.2f}. Reply if you would like viewing details or similar options."
        )

    @staticmethod
    def _presentation_title(title: object, bedrooms: object = 0, property_type: object = "House") -> str:
        """Normalize noisy scraped titles into cleaner display names."""

        cleaned = str(title or "").replace("_", " ").strip()
        cleaned = re.sub(r"(?i)\bserious potential buyers only\b", "", cleaned)
        cleaned = re.sub(r"(?i)\bsize\s+\d+\s+square\s+meters\b", "", cleaned)
        cleaned = re.sub(r"(?i)^\s*1\s+unit\s*,\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
        bedroom_count = MarketingAutomation._coerce_int(bedrooms)
        property_type_text = str(property_type or "House").strip().title()

        match = re.fullmatch(r"(?i)(\d+)\s+bedrooms?,\s*house", cleaned)
        if match:
            return f"{match.group(1)}-Bedroom House"
        promotional_house_pattern = re.search(
            r"(?i)\bhouse\b.*\b(for sale|for rent)\b|\b(for sale|for rent)\b.*\bhouse\b|beautiful|highly finished|newly built",
            cleaned,
        )
        if promotional_house_pattern and bedroom_count > 0:
            return f"{bedroom_count}-Bedroom {property_type_text}"
        if cleaned.lower() in {"house", property_type_text.lower()} and bedroom_count > 0:
            return f"{bedroom_count}-Bedroom {property_type_text}"
        if cleaned.lower() == "ground floor" and bedroom_count > 0:
            return f"{bedroom_count}-Bedroom House"
        return cleaned.title() if cleaned.islower() else cleaned

    @staticmethod
    def _format_currency(value: object) -> str:
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            return "LSL -"
        return f"LSL {numeric:,}"

    @staticmethod
    def _build_subject_line(
        *,
        client: pd.Series,
        property_row: pd.Series,
        display_title: str,
        language: str,
        channel: str,
    ) -> str:
        district = str(property_row.get("district", "") or "").strip()
        if language == "st":
            if channel == "social":
                return f"Ntlo e loketseng wena: {display_title} ho {district}"
            return f"{client['name']}, re o fumanetse {display_title} ho {district}"
        if channel == "social":
            return f"Property match for you: {display_title} in {district}"
        return f"{client['name']}, your top house match is {display_title}"

    @staticmethod
    def _build_preview_text(*, property_row: pd.Series, display_title: str, language: str) -> str:
        price_text = MarketingAutomation._format_currency(property_row.get("price"))
        district = str(property_row.get("district", "") or "").strip()
        bedrooms = int(float(property_row.get("bedrooms", 0) or 0))
        if language == "st":
            return f"{display_title}, dikamore tse {bedrooms}, {district}, theko {price_text}."
        return f"{display_title} with {bedrooms} bedrooms in {district}, priced at {price_text}."

    @staticmethod
    def _build_call_to_action(language: str, channel: str) -> str:
        if language == "st":
            return (
                "Araba molaetsa ona bakeng sa viewing details."
                if channel == "email"
                else "Re romelle molaetsa haeba o batla viewing details."
            )
        return (
            "Reply to this message for viewing details."
            if channel == "email"
            else "Send us a message if you want viewing details."
        )

    @staticmethod
    def _campaign_variant(channel: str) -> str:
        variants = {
            "email": "email_property_spotlight",
            "social": "social_property_spotlight",
            "dashboard": "dashboard_demo_followup",
        }
        return variants.get(channel, "generic_property_spotlight")

    @staticmethod
    def _recommended_send_window(channel: str) -> str:
        windows = {
            "email": "weekday_evening",
            "social": "early_evening",
            "dashboard": "on_demand_demo",
        }
        return windows.get(channel, "weekday_evening")

    @staticmethod
    def _estimate_engagement_score(
        *,
        match_score: float,
        channel: str,
        recommendation_reasons: list[str],
    ) -> float:
        channel_bonus = {"email": 0.04, "social": 0.02, "dashboard": 0.01}.get(channel, 0.0)
        evidence_bonus = min(len(recommendation_reasons), 3) * 0.03
        return round(min(0.99, max(0.0, match_score + channel_bonus + evidence_bonus)), 4)

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
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError:
                    return [stripped]
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return []

    @staticmethod
    def _coerce_int(value: object) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _environment_en(value: str) -> str:
        mapping = {
            "Urban": "an urban area",
            "Suburban": "a suburban area",
            "Garden": "a garden setting",
            "Hillside": "a hillside setting",
        }
        return mapping.get(value, value.lower() if value else "a residential area")

    @staticmethod
    def _environment_st(value: str) -> str:
        mapping = {
            "Urban": "tikolohong ya toropo",
            "Suburban": "tikolohong e kgutsitseng ya matlo",
            "Garden": "tikolohong e nang le serapa",
            "Hillside": "tikolohong ya maralla",
        }
        return mapping.get(value, "tikolohong ya matlo")

    @staticmethod
    def _condition_st(value: str) -> str:
        mapping = {
            "New": "bo botjha",
            "Good": "bo botle",
            "Fair": "bo mahareng",
            "Needs Work": "bo hlokang tokiso",
        }
        return mapping.get(value, "bo botle")

    @staticmethod
    def _highlights_english(property_row: pd.Series, client: pd.Series) -> str:
        amenities = MarketingAutomation._coerce_list(property_row.get("amenities", []))
        client_text = " ".join(
            [
                str(client.get("free_text_preference_en", "")),
                str(client.get("free_text_preference_st", "")),
            ]
        ).lower()
        preferred_highlights: list[str] = []
        highlight_map = {
            "parking": "secure parking",
            "garage": "a garage",
            "yard": "a yard",
            "garden": "a garden",
            "road access": "good road access",
            "furnished": "furnished space",
        }
        for amenity, label in highlight_map.items():
            if amenity in amenities or amenity.replace(" ", "") in "".join(amenities):
                if amenity.split()[0] in client_text or amenity in {"parking", "yard", "garden"}:
                    preferred_highlights.append(label)
        if not preferred_highlights:
            for amenity in amenities[:2]:
                preferred_highlights.append(highlight_map.get(amenity, amenity))
        if preferred_highlights:
            return f"Highlights include {MarketingAutomation._join_phrases(preferred_highlights)}."
        return "It remains one of the cleaner house options in the current inventory."

    @staticmethod
    def _highlights_sesotho(property_row: pd.Series, client: pd.Series) -> str:
        amenities = MarketingAutomation._coerce_list(property_row.get("amenities", []))
        translated = {
            "parking": "parking e sireletsehileng",
            "garage": "garaji",
            "yard": "lebala",
            "garden": "serapa",
            "road access": "tsela e bonolo",
            "furnished": "disebediswa tse seng di lokile",
        }
        highlights = [translated[item] for item in amenities if item in translated][:2]
        if highlights:
            return f"E boetse e fana ka {MarketingAutomation._join_phrases(highlights)}."
        return "Le yona ke nngwe ya dikgetho tse hlwekileng tseo re nang le tsona hona jwale."

    @staticmethod
    def _join_phrases(values: list[str]) -> str:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    @staticmethod
    def _reason_sentence_english(reasons: list[str]) -> str:
        cleaned = [str(reason).strip().rstrip(".") for reason in reasons if str(reason).strip()]
        if not cleaned:
            return ""
        return f"It stands out because {MarketingAutomation._join_phrases(cleaned[:2])}. "
