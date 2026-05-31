"""Presentation-friendly marketing message generation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd


CAMPAIGN_COLUMNS = [
    "campaign_id",
    "client_id",
    "client_name",
    "property_id",
    "property_title",
    "rank",
    "channel",
    "language",
    "campaign_variant",
    "subject_line",
    "preview_text",
    "message",
    "call_to_action",
    "recommendation_reasons",
    "status",
    "delivery_state",
    "recommended_send_window",
    "sent_at_utc",
    "match_score",
    "estimated_engagement_score",
]


class MarketingAutomation:
    def generate(
        self,
        matches: pd.DataFrame,
        properties: pd.DataFrame,
        clients: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create one rank-aware campaign/message row for every returned match."""

        ranked_matches = matches.sort_values(["client_id", "rank"])
        property_lookup = properties.set_index("property_id")
        client_lookup = clients.set_index("client_id")

        records: list[dict[str, object]] = []
        for match in ranked_matches.itertuples(index=False):
            client = client_lookup.loc[match.client_id]
            property_row = property_lookup.loc[match.property_id]
            language = client["preferred_language"]
            preferred_channels = self._coerce_list(client.get("preferred_channels", []))
            channel = preferred_channels[0] if preferred_channels else "email"
            rank = self._coerce_int(getattr(match, "rank", 1)) or 1
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
                rank=rank,
            )
            preview_text = self._build_preview_text(
                property_row=property_row,
                display_title=display_title,
                language=language,
                rank=rank,
            )
            call_to_action = self._build_call_to_action(language, channel)
            estimated_engagement = self._estimate_engagement_score(
                match_score=float(match.overall_score),
                channel=channel,
                recommendation_reasons=recommendation_reasons,
                rank=rank,
            )
            records.append(
                {
                    "campaign_id": f"CAMP-{match.client_id}-R{rank}-{match.property_id}",
                    "client_id": match.client_id,
                    "client_name": client["name"],
                    "property_id": match.property_id,
                    "property_title": display_title,
                    "rank": rank,
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
        return pd.DataFrame(records, columns=CAMPAIGN_COLUMNS)

    @staticmethod
    def _build_message(client: pd.Series, property_row: pd.Series, match, language: str) -> str:
        """Generate a lightweight bilingual message from a ranked match."""

        rank = MarketingAutomation._coerce_int(getattr(match, "rank", 1)) or 1
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
        reason_sentence_st = MarketingAutomation._reason_sentence_sesotho(recommendation_reasons)
        bedroom_note_en = MarketingAutomation._bedroom_fit_sentence_english(client, property_row, rank)
        bedroom_note_st = MarketingAutomation._bedroom_fit_sentence_sesotho(client, property_row, rank)

        if language == "st":
            return (
                f"Lumela {client['name']}, ena ke {MarketingAutomation._rank_label_sesotho(rank)}: re fumane {title} ho {place_text} ka theko ya {price_text}. "
                f"E na le dikamore tse {int(property_row['bedrooms'])}, e boemong ba {MarketingAutomation._condition_st(condition)}, "
                f"mme e fumaneha {MarketingAutomation._environment_st(environment)}. {bedroom_note_st}{reason_sentence_st}{highlights_st} "
                f"Tekanyo ya ho tshwana ke {match.overall_score:.2f}. Re ka o romella dintlha tse ding kapa ra hlophisa ketelo."
            )
        hook = MarketingAutomation._english_hook(rank, title, place_text)
        return (
            f"Hi {client['name']}, {hook} Priced at {price_text}, this {int(property_row['bedrooms'])}-bedroom option "
            f"puts you in {MarketingAutomation._environment_en(environment)} with a home that is {MarketingAutomation._condition_phrase_en(condition)}. "
            f"{bedroom_note_en}{reason_sentence_en}{highlights_en} "
            "In plain terms, it is not just another listing; it is one of the clearest signals that your search is pointing in the right direction. "
            f"Match confidence: {match.overall_score:.2f}. Reply if you would like viewing details, a side-by-side comparison, or similar homes before this opportunity moves."
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
        rank: int,
    ) -> str:
        district = str(property_row.get("district", "") or "").strip()
        rank_label_en = MarketingAutomation._rank_label_english(rank)
        rank_label_st = MarketingAutomation._rank_label_sesotho(rank)
        if language == "st":
            if channel == "social":
                return f"{rank_label_st.title()} ya ntlo: {display_title} ho {district}"
            return f"{client['name']}, {rank_label_st}: {display_title} ho {district}"
        hook = "Don't miss this match" if rank == 1 else "Worth a closer look"
        if channel == "social":
            return f"{hook}: {display_title} in {district}"
        return f"{client['name']}, {hook}: {display_title}"

    @staticmethod
    def _build_preview_text(*, property_row: pd.Series, display_title: str, language: str, rank: int) -> str:
        price_text = MarketingAutomation._format_currency(property_row.get("price"))
        district = str(property_row.get("district", "") or "").strip()
        bedrooms = int(float(property_row.get("bedrooms", 0) or 0))
        rank_label_en = MarketingAutomation._rank_label_english(rank)
        rank_label_st = MarketingAutomation._rank_label_sesotho(rank)
        if language == "st":
            return f"{rank_label_st.title()}: {display_title}, dikamore tse {bedrooms}, {district}, theko ya {price_text}."
        lead = "Strongest lead" if rank == 1 else f"{rank_label_en.title()} backup"
        return f"{lead}: {display_title} with {bedrooms} bedrooms in {district} at {price_text}; open it before comparing weaker options."

    @staticmethod
    def _build_call_to_action(language: str, channel: str) -> str:
        if language == "st":
            return (
                "Araba molaetsa ona bakeng sa dintlha tsa ketelo."
                if channel == "email"
                else "Re romelle molaetsa haeba o batla dintlha tsa ketelo."
            )
        return (
            "Reply to this message to lock in viewing details or compare this against similar homes."
            if channel == "email"
            else "Send us a message to secure viewing details or compare similar homes."
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
        rank: int,
    ) -> float:
        channel_bonus = {"email": 0.04, "social": 0.02, "dashboard": 0.01}.get(channel, 0.0)
        evidence_bonus = min(len(recommendation_reasons), 3) * 0.03
        rank_penalty = max(0, rank - 1) * 0.015
        return round(min(0.99, max(0.0, match_score + channel_bonus + evidence_bonus - rank_penalty)), 4)

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
    def _rank_label_english(rank: int) -> str:
        labels = {
            1: "top match",
            2: "second choice",
            3: "third choice",
            4: "fourth choice",
            5: "fifth choice",
        }
        return labels.get(rank, f"choice #{rank}")

    @staticmethod
    def _english_hook(rank: int, title: str, place_text: str) -> str:
        if rank == 1:
            return (
                f"pause on this one: your top match, {title} in {place_text}, deserves the first viewing slot."
            )
        rank_label = MarketingAutomation._rank_label_english(rank)
        return (
            f"this {rank_label} is still worth attention: {title} in {place_text} stayed competitive after the stronger filters ran."
        )

    @staticmethod
    def _condition_phrase_en(value: str) -> str:
        cleaned = str(value or "").strip().lower()
        if cleaned in {"new", "good", "fair"}:
            return f"in {cleaned} condition"
        if cleaned in {"renovation needed", "needs work"}:
            return "a renovation opportunity"
        return "ready for review"

    @staticmethod
    def _rank_label_sesotho(rank: int) -> str:
        labels = {
            1: "kgetho ya pele",
            2: "kgetho ya bobedi",
            3: "kgetho ya boraro",
            4: "kgetho ya bone",
            5: "kgetho ya bohlano",
        }
        return labels.get(rank, f"kgetho ya maemo a {rank}")

    @staticmethod
    def _bedroom_fit_sentence_english(client: pd.Series, property_row: pd.Series, rank: int) -> str:
        preferred = MarketingAutomation._coerce_int(client.get("preferred_bedrooms", 0))
        actual = MarketingAutomation._coerce_int(property_row.get("bedrooms", 0))
        if not preferred or not actual:
            return ""
        if preferred == actual:
            return f"It matches your preferred {preferred}-bedroom requirement. "
        difference = actual - preferred
        direction = "above" if difference > 0 else "short of"
        plural = "bedroom" if abs(difference) == 1 else "bedrooms"
        if rank == 1:
            return (
                f"Bedroom note: it has {actual} bedrooms, which is {abs(difference)} {plural} {direction} "
                f"your preferred {preferred}, but other signals kept it competitive. "
            )
        return (
            f"Bedroom note: it has {actual} bedrooms, which is {abs(difference)} {plural} {direction} "
            f"your preferred {preferred}, so it is presented as a backup option rather than the strongest bedroom-fit choice. "
        )

    @staticmethod
    def _bedroom_fit_sentence_sesotho(client: pd.Series, property_row: pd.Series, rank: int) -> str:
        preferred = MarketingAutomation._coerce_int(client.get("preferred_bedrooms", 0))
        actual = MarketingAutomation._coerce_int(property_row.get("bedrooms", 0))
        if not preferred or not actual:
            return ""
        if preferred == actual:
            return f"Palo ya dikamore e dumellana le dikamore tse {preferred} tseo o di batlang. "
        if rank == 1:
            return (
                f"Tlhokomeliso ya dikamore: e na le dikamore tse {actual}, athe o kgethile tse {preferred}; "
                "matshwao a mang a ntse a e phahamisitse. "
            )
        return (
            f"Tlhokomeliso ya dikamore: e na le dikamore tse {actual}, athe o kgethile tse {preferred}; "
            "ka hona e bontshwa e le kgetho e nngwe, eseng kgetho e matla ka palo ya dikamore. "
        )

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
            return f"Highlights include {MarketingAutomation._join_phrases_english(preferred_highlights)}."
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
            return f"E boetse e fana ka {MarketingAutomation._join_phrases_sesotho(highlights)}."
        return "Le yona ke nngwe ya dikgetho tse hlwekileng tseo re nang le tsona hona jwale."

    @staticmethod
    def _join_phrases_english(values: list[str]) -> str:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    @staticmethod
    def _join_phrases_sesotho(values: list[str]) -> str:
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} le {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", le {cleaned[-1]}"

    @staticmethod
    def _reason_sentence_english(reasons: list[str]) -> str:
        cleaned = [str(reason).strip().rstrip(".") for reason in reasons if str(reason).strip()]
        if not cleaned:
            return ""
        return f"It stands out because {MarketingAutomation._join_phrases_english(cleaned[:2])}. "

    @staticmethod
    def _reason_sentence_sesotho(reasons: list[str]) -> str:
        translated: list[str] = []
        for reason in reasons:
            cleaned = str(reason).strip().rstrip(".")
            if not cleaned:
                continue
            translated.append(MarketingAutomation._translate_reason_to_sesotho(cleaned))
        if not translated:
            return ""
        return f"E hlahella hobane {MarketingAutomation._join_phrases_sesotho(translated[:2])}. "

    @staticmethod
    def _translate_reason_to_sesotho(reason: str) -> str:
        lowered = reason.lower()
        if "budget ceiling" in lowered:
            return "mokgwa wa kgetho o hlomphile moedi wa tekanyetso ya hao"
        if "budget is closely aligned" in lowered:
            return "theko e atamela tekanyetso ya hao"
        if "preferred district" in lowered:
            return "sebaka sena se dumellana le setereke seo o se ratang"
        if "property type matches" in lowered:
            return "mofuta wa thepa o dumellana le khetho ya hao"
        if "bedroom layout" in lowered:
            return "palo ya dikamore e dumellana hantle le seo o se batlang"
        if "text preferences align on" in lowered:
            detail = reason.split("on", 1)[-1].strip() if "on" in reason else reason
            return f"dikgetho tsa hao tsa mongolo di tsamaisana le {detail}"
        if "vision model supports the" in lowered:
            detail = reason.split("supports the", 1)[-1].strip() if "supports the" in lowered else reason
            return f"tlhahlobo ya ditshwantsho e tshehetsa {detail}"
        if "image analysis suggests" in lowered:
            detail = reason.split("suggests", 1)[-1].strip() if "suggests" in lowered else reason
            return f"tlhahlobo ya ditshwantsho e bontsha boemo bo {detail}"
        return reason
