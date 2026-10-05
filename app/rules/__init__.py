"""Rules package - deterministic, data-driven eligibility logic (NO LLM)."""

from app.rules.engine import evaluate, evaluate_rules, run_rule, summarize
from app.rules.loader import (
    RuleDataError,
    RuleRepository,
    SchemeFinancials,
    SchemeRuleSet,
    SchemeSource,
    get_rule_repository,
)

__all__ = [
    "evaluate",
    "evaluate_rules",
    "run_rule",
    "summarize",
    "RuleDataError",
    "RuleRepository",
    "SchemeRuleSet",
    "SchemeFinancials",
    "SchemeSource",
    "get_rule_repository",
]
