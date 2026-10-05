"""Agentic layer (LangGraph + LangChain).

WHAT THIS LAYER MAY DO: interpret free text, retrieve evidence, classify/verify claims,
explain and route.
WHAT THIS LAYER MUST NEVER DO: compute EMI/interest, thresholds or eligibility outcomes.
Those come exclusively from app/services/* and app/rules/* and are passed in as facts.
"""
