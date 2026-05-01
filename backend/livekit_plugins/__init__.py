"""LiveKit STT/TTS adapters that route through medkit's provider layer.

These wrap `WhisperProvider` and `PiperProvider` in the interfaces the
LiveKit Agents framework expects, so `voice_agent_local.py` can plug them
into an `AgentSession` exactly like the cloud Deepgram/Cartesia plugins.
"""
