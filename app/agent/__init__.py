"""
Agent package -- implementation of the incident analysis agent.

Internal structure:
  exceptions.py     -- typed domain exceptions
  llm_client.py     -- LLM client protocol and OpenAI implementation
  input_parser.py   -- Stage 1: input normalisation and keyword extraction
  retriever.py      -- Stage 2: relevant context retrieval (RAG)
  prompt_builder.py -- Stage 3: system prompt assembly
  analyzer.py       -- Stage 4: pipeline orchestrator (main entry point)
"""
