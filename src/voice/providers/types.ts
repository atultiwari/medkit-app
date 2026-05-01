/**
 * Provider abstraction types — frontend mirror of the Python provider layer.
 *
 * The browser doesn't talk to LLMs directly; everything routes through the
 * backend. These types exist so the UI can label what's active and so the
 * `useAttendingDebrief` hook can pick between the Managed Agents and local
 * SSE endpoints based on configuration.
 */

export type LLMProviderName = 'ollama' | 'gemini' | 'claude';
export type STTProviderName = 'whisper' | 'deepgram';
export type TTSProviderName = 'piper' | 'cartesia';
export type LiveKitMode = 'local' | 'cloud';

export interface ProviderConfig {
  llm: LLMProviderName;
  stt: STTProviderName;
  tts: TTSProviderName;
  livekitMode: LiveKitMode;
  /** Ordered fallback chain for the LLM. The first entry equals `llm`. */
  fallbackChain: LLMProviderName[];
}

export interface BackendHealth {
  ok: boolean;
  providers?: Partial<ProviderConfig> & { fallback_chain?: string };
  voice?: {
    transport?: string;
    livekit_configured?: boolean;
    deepgram_configured?: boolean;
    cartesia_configured?: boolean;
    whisper_configured?: boolean;
    piper_configured?: boolean;
  };
  agent?: {
    anthropic_sdk_installed?: boolean;
    api_key_configured?: boolean;
    bootstrapped?: boolean;
    ollama_configured?: boolean;
    gemini_configured?: boolean;
  };
}
