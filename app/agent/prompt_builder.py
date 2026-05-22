"""
PromptBuilder -- system prompt assembly and self-correction message formatting.

Extracting prompts from business logic provides three benefits:
  1. Prompts can be tested and version-controlled independently of the orchestrator.
  2. Taxonomy and severity rules are defined in one place (DRY).
  3. The analyzer does not know formatting details -- it just calls build_*.
"""
import json

from app.context import SYSTEM_ARCHITECTURE
from app.schemas import IncidentAnalysis

# ---------------------------------------------------------------------------
# Taxonomy and severity rubric -- single source of truth.
# Used in the prompt and (in future) for category validation.
# ---------------------------------------------------------------------------
TAXONOMY_CATEGORIES = [
    "External payment provider issue",
    "Database resource saturation",
    "Authentication token failure",
    "Notification delivery degradation",
    "Compound infrastructure degradation",
    "Network routing issue",
]

SEVERITY_RUBRIC = """SEVERITY RUBRIC:
  - 'high':   core financial operations or authentication are blocked,
              mass payment failures, or DB saturation with cascading timeouts.
  - 'medium': authentication degraded for a subset of users,
              or critical MFA SMS delayed.
  - 'low':    financial operations are fully intact. Assign 'low' when the
              transaction succeeded but a confirmation notification was delayed."""


class PromptBuilder:
    """Builds text blocks for LLM interaction."""

    def build_system_prompt(self, retrieved_context: str) -> str:
        """
        Assembles the system prompt by injecting:
          - platform architecture description;
          - relevant past incidents (RAG context);
          - incident taxonomy and severity rubric;
          - JSON schema of the expected response.
        """
        taxonomy_list = "\n".join(f'  "{c}"' for c in TAXONOMY_CATEGORIES)
        schema_json = json.dumps(IncidentAnalysis.model_json_schema(), indent=2)

        return (
            "You are an SRE Incident Triage Assistant.\n"
            "Analyze the incident report using the provided architecture and relevant past incidents.\n\n"
            f"=== PLATFORM ARCHITECTURE ===\n{SYSTEM_ARCHITECTURE}\n\n"
            f"=== RELEVANT PAST INCIDENTS (RAG context) ===\n{retrieved_context}\n\n"
            f"=== INCIDENT TAXONOMY ===\n"
            f"Choose the most precise category from the list (or a close equivalent if none fits exactly):\n"
            f"{taxonomy_list}\n\n"
            f"=== {SEVERITY_RUBRIC} ===\n\n"
            f"=== RESPONSE JSON SCHEMA ===\n{schema_json}\n\n"
            "Return ONLY a valid JSON object matching the schema above. No surrounding text."
        )

    def build_correction_message(self, validation_errors_json: str) -> str:
        """
        Formats a corrective message for the self-healing retry loop.

        Appended to the conversation history after a failed validation so that
        the LLM knows the exact errors and can fix them.
        """
        return (
            "Your previous response failed Pydantic validation. "
            "Fix the listed errors and return ONLY valid JSON:\n"
            f"{validation_errors_json}"
        )
