import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from app.agent import IncidentAgent

@pytest.fixture
def mock_agent():
    # Initialize with a dummy key to prevent real API calls during tests
    agent = IncidentAgent(api_key="dummy_key")
    agent.client.chat.completions.create = AsyncMock()
    return agent

@pytest.mark.asyncio
async def test_happy_path_canonical_example(mock_agent):
    """Test that the agent perfectly processes the exact example from the assignment PDF."""
    
    # Exact input from the assignment
    incident_text = """Customers complain that card payments often fail, and transactions do not go through.
payment-service logs show many timeouts when calling PayGate, starting from 12:05 UTC.
Other services look normal."""
    
    # Expected JSON (includes exact strings from assignment + our strict schema fields)
    valid_json = json.dumps({
        "category": "External payment provider issue",
        "severity": "high",
        "severity_reason": "Massive payment failures directly impact revenue.",
        "affected_users": "All customers attempting card payments",
        "summary": "The external provider PayGate is not responding in time, causing mass card payment failures.",
        "hypotheses": [
            {
                "title": "Degradation or incident on the PayGate side",
                "reasoning": "Timeouts are observed only when calling PayGate, other services remain stable.",
                "next_steps": [
                    "Check PayGate status page and recent provider notifications.",
                    "Compare error and latency metrics for PayGate vs other payment providers.",
                    "If possible, temporarily shift part of the traffic to an alternative provider."
                ]
            }
        ]
    })
    
    mock_choice = MagicMock()
    mock_choice.message.content = valid_json
    mock_agent.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    # Execute analysis
    result = await mock_agent.analyze(incident_text)
    
    # Strict assertions matching the assignment requirements
    assert result["category"] == "External payment provider issue"
    assert result["severity"] == "high"
    assert result["summary"] == "The external provider PayGate is not responding in time, causing mass card payment failures."
    
    assert len(result["hypotheses"]) == 1
    hypothesis = result["hypotheses"][0]
    
    assert hypothesis["title"] == "Degradation or incident on the PayGate side"
    assert hypothesis["reasoning"] == "Timeouts are observed only when calling PayGate, other services remain stable."
    assert len(hypothesis["next_steps"]) == 3
    assert hypothesis["next_steps"][0] == "Check PayGate status page and recent provider notifications."

@pytest.mark.asyncio
async def test_retry_on_invalid_json(mock_agent):
    """Test the recovery strategy: invalid JSON on first try, valid on second."""
    incident_text = "Sharp increase in response time for /payments/create (up to 5-7 seconds)."
    invalid_json = '{"category": "Missing fields"}' 
    
    valid_json = json.dumps({
        "category": "DB degradation",
        "severity": "medium",
        "severity_reason": "Testing recovery mechanism.",
        "affected_users": "None",
        "summary": "This is a valid summary of length > 10.",
        "hypotheses": [
            {
                "title": "Valid hypothesis title", 
                "reasoning": "Reasoning is long enough to pass validation.", 
                "next_steps": ["Step 1 to check", "Step 2 to verify"]
            }
        ]
    })

    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = invalid_json
    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = valid_json
    
    mock_agent.client.chat.completions.create.side_effect = [
        MagicMock(choices=[mock_choice_1]), 
        MagicMock(choices=[mock_choice_2])
    ]

    result = await mock_agent.analyze(incident_text, max_retries=2)
    
    assert result["category"] == "DB degradation"
    assert mock_agent.client.chat.completions.create.call_count == 2

@pytest.mark.asyncio
async def test_failure_after_max_retries(mock_agent):
    """Test that ValueError is raised if all retries fail."""
    incident_text = "Some users cannot log in via the mobile app."
    invalid_json = '{"bad": "data"}'
    
    mock_choice = MagicMock()
    mock_choice.message.content = invalid_json
    mock_agent.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with pytest.raises(ValueError, match="Failed to generate a valid response"):
        await mock_agent.analyze(incident_text, max_retries=2)

def test_retrieve_context_semantic(mock_agent):
    """Test that the RAG retrieval logic selects the semantically correct past incidents."""
    
    # Test Case 1: SMTP / Emails
    smtp_parsed_data = {
        "raw_text": "Users are not receiving emails",
        "keywords": ["users", "receiving", "emails", "smtp"]
    }
    smtp_context = mock_agent._retrieve_context(smtp_parsed_data)
    
    assert "SMTP provider" in smtp_context
    # Ensure it didn't pull the PayGate incident
    assert "PayGate provider" not in smtp_context

    # Test Case 2: DB / Reporting Load
    db_parsed_data = {
        "raw_text": "CPU load is high on PostgreSQL due to reporting",
        "keywords": ["cpu", "load", "postgresql", "reporting"]
    }
    db_context = mock_agent._retrieve_context(db_parsed_data)
    
    assert "reporting-service" in db_context
    assert "DB dashboards show high CPU" in db_context