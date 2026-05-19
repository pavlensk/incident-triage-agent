import os
import sys
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.agent import IncidentAgent

load_dotenv()

# --- Logging Configuration ---
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_file = os.getenv("LOG_FILE", "app.log")

logging_handlers = [logging.StreamHandler(sys.stdout)]
if log_file:
    logging_handlers.append(logging.FileHandler(log_file))

logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=logging_handlers
)

logger = logging.getLogger(__name__)
# -----------------------------

app = FastAPI(title="AI Incident Triage API")

# Singleton initialization at startup
api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
if api_key == "dummy_key":
    logger.warning("OPENAI_API_KEY is missing. LLM calls will fail.")
    
global_agent = IncidentAgent(api_key=api_key)

# Dependency Injection function
def get_agent() -> IncidentAgent:
    return global_agent

class AnalyzeRequest(BaseModel):
    incident_text: str = Field(
        ..., 
        description="The raw text of the incident or logs."
    )

@app.post("/api/v1/analyze")
async def analyze_incident(req: AnalyzeRequest, agent: IncidentAgent = Depends(get_agent)):
    text = req.incident_text.strip()
    words = text.split()
    
    # Unified manual validation for both word count and character length
    if len(words) < 5 or len(text) < 30:
        logger.warning("Rejected payload due to insufficient length or word count.")
        raise HTTPException(
            status_code=400, 
            detail="Incident text is too short or meaningless. Please provide at least 5 words and 30 characters of context."
        )
        
    try:
        logger.info("Received incident analysis request.")
        result = await agent.analyze(req.incident_text)
        logger.info(f"Successfully analyzed incident. Assigned category: {result.get('category')}")
        return result
    except ValueError as e:
        logger.error(f"Validation error during LLM analysis: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error processing incident: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Mount static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))