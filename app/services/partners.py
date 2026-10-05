"""Partner matching - deterministic filter over a small DEMO dataset.

No LLM, no scraping, no real SCA directory. data/partners_sample.json is clearly
labelled demo data; matching is scheme_id + state -> matching partner(s).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import get_settings

ANY_STATE = {"*", "all", "all india", "any"}


class PartnerRecord(BaseModel):
    partner_id: str
    name: str
    states: list[str] = Field(default_factory=list)
    scheme_ids: list[str] = Field(default_factory=list)
    contact_email: str | None = None
    contact_phone: str | None = None
    address: str | None = None
    languages: list[str] = Field(default_factory=list)
    is_demo: bool = True
    note: str | None = None


class PartnerDataset(BaseModel):
    demo_data: bool = True
    disclaimer: str = ""
    partners: list[PartnerRecord] = Field(default_factory=list)


class PartnerMatch(BaseModel):
    partner: PartnerRecord
    matched_on: dict[str, str]
    is_demo: bool = True


def _normalize(value: str | None) -> str:
    return (value or "").strip().casefold()


class PartnerService:
    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.dataset = self._load()

    def _load(self) -> PartnerDataset:
        raw = json.loads(self.data_file.read_text(encoding="utf-8"))
        return PartnerDataset.model_validate(raw)

    def match(self, scheme_id: str, state: str | None) -> list[PartnerMatch]:
        state_key = _normalize(state)
        matches: list[PartnerMatch] = []
        for partner in self.dataset.partners:
            scheme_ok = any(_normalize(s) == _normalize(scheme_id) for s in partner.scheme_ids)
            if not scheme_ok:
                continue
            partner_states = {_normalize(s) for s in partner.states}
            state_ok = bool(state_key and state_key in partner_states) or bool(partner_states & ANY_STATE)
            if not state_ok:
                continue
            if not state_key and not (partner_states & ANY_STATE):
                continue
            matches.append(
                PartnerMatch(
                    partner=partner,
                    matched_on={"scheme_id": scheme_id, "state": state or "not_provided"},
                    is_demo=self.dataset.demo_data,
                )
            )
        return sorted(matches, key=lambda m: m.partner.partner_id)

    def match_many(self, scheme_ids: list[str], state: str | None) -> dict[str, list[PartnerMatch]]:
        return {scheme_id: self.match(scheme_id, state) for scheme_id in scheme_ids}


@lru_cache
def get_partner_service() -> PartnerService:
    return PartnerService(get_settings().partners_file)
