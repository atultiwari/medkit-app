/**
 * Frontend provider config — read once at startup from `/health`.
 *
 * The browser never instantiates an LLM client; this module exists so UI
 * components can show "ollama (offline)" vs "claude (cloud)" and so the
 * debrief hook knows whether to hit the Managed Agents endpoint or the
 * local SSE endpoint.
 */

import type {
  BackendHealth,
  LLMProviderName,
  LiveKitMode,
  ProviderConfig,
  STTProviderName,
  TTSProviderName,
} from './types';

const DEFAULT_CONFIG: ProviderConfig = {
  llm: 'ollama',
  stt: 'whisper',
  tts: 'piper',
  livekitMode: 'local',
  fallbackChain: ['ollama'],
};

const VALID_LLMS: ReadonlySet<LLMProviderName> = new Set([
  'ollama',
  'gemini',
  'claude',
]);
const VALID_STTS: ReadonlySet<STTProviderName> = new Set(['whisper', 'deepgram']);
const VALID_TTSS: ReadonlySet<TTSProviderName> = new Set(['piper', 'cartesia']);
const VALID_LK_MODES: ReadonlySet<LiveKitMode> = new Set(['local', 'cloud']);

let cachedConfig: ProviderConfig | null = null;
let inflight: Promise<ProviderConfig> | null = null;

function asLLM(value: unknown): LLMProviderName | null {
  return typeof value === 'string' && VALID_LLMS.has(value as LLMProviderName)
    ? (value as LLMProviderName)
    : null;
}

function asSTT(value: unknown): STTProviderName | null {
  return typeof value === 'string' && VALID_STTS.has(value as STTProviderName)
    ? (value as STTProviderName)
    : null;
}

function asTTS(value: unknown): TTSProviderName | null {
  return typeof value === 'string' && VALID_TTSS.has(value as TTSProviderName)
    ? (value as TTSProviderName)
    : null;
}

function asMode(value: unknown): LiveKitMode | null {
  return typeof value === 'string' && VALID_LK_MODES.has(value as LiveKitMode)
    ? (value as LiveKitMode)
    : null;
}

function parseChain(raw: string | undefined, primary: LLMProviderName): LLMProviderName[] {
  if (!raw) return [primary];
  const seen = new Set<LLMProviderName>();
  const out: LLMProviderName[] = [];
  for (const part of raw.split(',')) {
    const llm = asLLM(part.trim().toLowerCase());
    if (llm && !seen.has(llm)) {
      seen.add(llm);
      out.push(llm);
    }
  }
  if (!out.includes(primary)) out.unshift(primary);
  return out;
}

function fromHealth(body: BackendHealth): ProviderConfig {
  const p = body.providers ?? {};
  const llm = asLLM(p.llm) ?? DEFAULT_CONFIG.llm;
  const stt = asSTT(p.stt) ?? DEFAULT_CONFIG.stt;
  const tts = asTTS(p.tts) ?? DEFAULT_CONFIG.tts;
  const livekitMode =
    asMode((p as Record<string, unknown>).livekit_mode ?? p.livekitMode) ??
    DEFAULT_CONFIG.livekitMode;
  return {
    llm,
    stt,
    tts,
    livekitMode,
    fallbackChain: parseChain(p.fallback_chain, llm),
  };
}

/** Resolve once and cache. Subsequent callers share the same promise. */
export async function getProviderConfig(): Promise<ProviderConfig> {
  if (cachedConfig) return cachedConfig;
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const res = await fetch('/health', { cache: 'no-store' });
      if (!res.ok) {
        cachedConfig = DEFAULT_CONFIG;
        return DEFAULT_CONFIG;
      }
      const body = (await res.json()) as BackendHealth;
      cachedConfig = fromHealth(body);
      return cachedConfig;
    } catch {
      cachedConfig = DEFAULT_CONFIG;
      return DEFAULT_CONFIG;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

/** Synchronous accessor — returns cache or defaults. Components that need
 *  blocking config should call `getProviderConfig` first and then read this. */
export function getProviderConfigSync(): ProviderConfig {
  return cachedConfig ?? DEFAULT_CONFIG;
}

/** Convenience: is the active LLM provider Anthropic Claude (Managed Agents)?
 *  When false, the debrief hook routes to `/agent/local/debrief`. */
export function isManagedAgentMode(config?: ProviderConfig): boolean {
  return (config ?? getProviderConfigSync()).llm === 'claude';
}

export type {
  BackendHealth,
  LLMProviderName,
  LiveKitMode,
  ProviderConfig,
  STTProviderName,
  TTSProviderName,
};
