"""NidhiSetu - evidence-backed agentic decision-support API for government loan/education schemes.

Package layout:
  app/rules       -> deterministic, data-driven rule engine (NO LLM)
  app/services    -> deterministic financial / eligibility / partner / freshness services (NO LLM)
  app/ai          -> LangGraph agentic layer (interpret / retrieve / explain / route ONLY)
  app/tools       -> single tool gateway all agent tool calls must pass through
  app/rag         -> ingestion + retrieval + evidence verification
  app/workers     -> Redis/RQ background jobs (inline fallback when Redis is absent)
"""

__version__ = "1.0.0"
