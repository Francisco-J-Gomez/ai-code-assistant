from dotenv import load_dotenv

# Unlike a plain script, pytest doesn't auto-load .env - opt-in integration
# tests that hit real OpenAI/Pinecone/Tavily (test_agent_integration.py)
# need real credentials from it. Harmless for every other test: load_dotenv()
# is a no-op if .env doesn't exist, and mocked tests never read these vars.
load_dotenv()
