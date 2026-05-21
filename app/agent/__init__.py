"""
Agent package -- implementation of the incident analysis agent.

Internal structure:
  exceptions.py     -- typed domain exceptions
  llm_client.py     -- LLM client protocol and OpenAI implementation
  retriever.py      -- relevant context retrieval (RAG stage)
  prompt_builder.py -- system prompt assembly
  analyzer.py       -- pipeline orchestrator (main entry point)
"""
