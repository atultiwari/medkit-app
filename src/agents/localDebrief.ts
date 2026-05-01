/**
 * Browser-side client for the local (offline) attending debrief endpoint.
 *
 * When the active LLM provider is not Anthropic, the debrief hook bypasses
 * Managed Agents and posts the same `DebriefRequest` to `/agent/local/debrief`
 * instead. The backend yields SSE events that mirror the Managed-Agents
 * event-stream shape, so consumers (eventStreamRenderer, useAttendingDebrief)
 * can iterate them with the same reducer.
 */

import type { ManagedAgentEvent } from './managedAgent';
import type { DebriefRequest } from './debriefRequest';

const LOCAL_DEBRIEF_URL = '/agent/local/debrief';

export interface LocalDebriefStreamOptions {
  signal?: AbortSignal;
}

export async function* openLocalDebriefStream(
  request: DebriefRequest,
  opts: LocalDebriefStreamOptions = {},
): AsyncGenerator<ManagedAgentEvent, void, unknown> {
  const res = await fetch(LOCAL_DEBRIEF_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '');
    throw new Error(`local debrief failed: ${res.status} ${detail}`.trim());
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const event = parseSseFrame(frame);
      if (event) yield event;
    }
  }
}

function parseSseFrame(frame: string): ManagedAgentEvent | null {
  let eventName = 'message';
  const dataLines: string[] = [];
  for (const rawLine of frame.split('\n')) {
    if (rawLine.startsWith(':')) continue; // SSE comment / keepalive
    if (rawLine.startsWith('event:')) {
      eventName = rawLine.slice(6).trim();
      continue;
    }
    if (rawLine.startsWith('data:')) {
      dataLines.push(rawLine.slice(5).trimStart());
    }
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join('\n');
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }
  const id = typeof parsed.id === 'string' ? parsed.id : `local-${Math.random().toString(36).slice(2)}`;
  const type = typeof parsed.type === 'string' ? parsed.type : eventName;
  return { ...parsed, id, type };
}
