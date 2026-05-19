import os
import json
import logging
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Dict, Any
from openai import AsyncOpenAI
from app.context import SYSTEM_ARCHITECTURE, PAST_INCIDENTS_LIST

logger = logging.getLogger(__name__)

class Hypothesis(BaseModel):
    title: str = Field(..., min_length=5, max_length=100, description="Short title of the hypothesis")
    reasoning: str = Field(..., min_length=10, description="Reasoning based strictly on logs and architecture")
    next_steps: List[str] = Field(..., min_length=2, max_length=3, description="2-3 concrete diagnostic steps")

class IncidentAnalysis(BaseModel):
    category: str = Field(..., min_length=3, description="Classification category")
    severity: Literal["low", "medium", "high"] = Field(...)
    severity_reason: str = Field(..., min_length=10, description="Why this severity was chosen")
    affected_users: str = Field(..., description="Who is affected (e.g., 'All card users', 'Internal team')")
    summary: str = Field(..., min_length=10, description="Short summary of what is happening")
    hypotheses: List[Hypothesis] = Field(..., min_length=1, max_length=3, description="Up to 3 hypotheses")

class IncidentAgent:
    def __init__(self, api_key: str):
        self.model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.client = AsyncOpenAI(api_key=api_key)

    def _parse_input(self, text: str) -> Dict[str, Any]:
        """Stage 1: Parsing. Extracts raw text and keywords for context retrieval."""
        cleaned_text = text.strip()
        keywords = [word.lower() for word in cleaned_text.replace('-', ' ').split() if len(word) > 4]
        return {"raw_text": cleaned_text, "keywords": keywords}

    def _retrieve_context(self, parsed_data: Dict[str, Any]) -> str:
        """Stage 2: Context Retrieval. Simulating a RAG approach using keyword overlap."""
        relevant_incidents = []
        for incident in PAST_INCIDENTS_LIST:
            if any(kw in incident.lower() for kw in parsed_data["keywords"]):
                relevant_incidents.append(incident)
        
        if not relevant_incidents:
            relevant_incidents = PAST_INCIDENTS_LIST[:2]
            
        return "\n".join(relevant_incidents)

    async def analyze(self, user_input: str, max_retries: int = None) -> dict:
        if max_retries is None:
            max_retries = int(os.getenv("MAX_RETRIES", "3"))
            
        parsed_data = self._parse_input(user_input)
        retrieved_incidents = self._retrieve_context(parsed_data)
        schema_json = json.dumps(IncidentAnalysis.model_json_schema(), indent=2)
        
        # ADDED: Taxonomy and Severity Rubric for strict alignment
        system_prompt = f"""You are an SRE Incident Triage Assistant. 
        Analyze the incident report using the provided architecture and RELEVANT past incidents.
        
        System Architecture:
        {SYSTEM_ARCHITECTURE}
        
        Relevant Past Incidents (RAG Context):
        {retrieved_incidents}
        
        --- TRIAGE GUIDELINES ---
        1. TAXONOMY: Choose a precise, stable category. Preferred categories:
           "External payment provider issue", "Database resource saturation", 
           "Authentication token failure", "Notification delivery degradation", 
           "Compound infrastructure degradation", "Network routing issue".
        2. SEVERITY RUBRIC:
           - 'high': Core financial operations/logins are blocked, mass payment failures, or DB saturation.
           - 'medium': Authentication degraded for a subset of users, or critical MFA SMS delayed.
           - 'low': Financial operations are completely intact. You MUST assign 'low' if the core transaction succeeded but the non-critical confirmation email/notification failed or delayed.
        
        You MUST reply STRICTLY in valid JSON format matching this exact schema:
        {schema_json}
        
        Return ONLY the raw JSON object."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this incident:\n{parsed_data['raw_text']}"}
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature
                )
                
                raw_response = response.choices[0].message.content
                analysis_result = IncidentAnalysis.model_validate_json(raw_response)
                return analysis_result.model_dump()

            except ValidationError as e:
                logger.warning(f"Attempt {attempt} failed schema validation: {e.errors()}")
                if attempt == max_retries:
                    raise ValueError(f"Failed to generate a valid response after {max_retries} attempts.")
                
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user", 
                    "content": f"Validation failed. Fix these errors and return ONLY valid JSON matching the schema:\n{e.json()}"
                })
            except Exception as e:
                logger.error(f"LLM API Error: {e}")
                raise Exception(f"LLM Service error: {str(e)}")