/**
 * Real-time voice conversation, backed by LiveKit + Deepgram + Cartesia.
 *
 * Public API surface kept from the legacy push-to-talk implementation so
 * existing consumers (PatientPanel, FloatingVoicePanel, Face, ExamRoom,
 * PatientChatPanel, conversationStore) keep compiling without changes.
 *
 * What changed inside:
 *   - No more browser Whisper / Kokoro / MediaRecorder. The mic streams
 *     over WebRTC into a LiveKit room; the Python voice agent worker
 *     does Deepgram STT → Claude Haiku 4.5 → Cartesia TTS and pipes the
 *     audio back over the same room.
 *   - `init()` now connects the room (and starts the patient's greeting).
 *   - `startListening()` / `stopListeningAndRespond()` are no-ops kept
 *     for backwards compat — real-time means the mic is open the whole
 *     time, no push-to-talk turn boundary.
 *   - `getMouthAmplitude()` reads from an AnalyserNode tapped onto the
 *     remote audio track, so the 3D face lip-sync still works.
 *   - Transcripts (both sides) come from LiveKit transcription events.
 *   - `sendTextMessage()` keeps the legacy /agent/patient/stream path so
 *     the typed-chat panel still functions while voice runs in parallel.
 */

import {
  Room,
  RoomEvent,
  Track,
  type RemoteAudioTrack,
  type RemoteTrack,
  type RemoteTrackPublication,
  type RemoteParticipant,
  type Participant,
  type TranscriptionSegment,
} from 'livekit-client';
import { hasClaudeKey, streamClaude, type ChatMessage } from './claude';

const FALLBACK_PERSONA =
  'You are a patient speaking to a doctor. Keep replies to 1–2 short spoken sentences. ' +
  'Output spoken dialogue only — no stage directions, no asterisks, no brackets.';
const FALLBACK_INITIAL: { role: 'assistant'; content: string } = {
  role: 'assistant',
  content: 'Hi doc.',
};

export type ConversationStatus =
  | 'uninitialized'
  | 'loading'
  | 'ready'
  | 'listening'
  | 'thinking'
  | 'speaking'
  | 'error';

export interface SubtitleEvent {
  who: 'patient' | 'you';
  text: string;
  partial?: boolean;
}

export type PatientEmotion = 'neutral' | 'pain' | 'fear' | 'relief' | 'confused';

export interface ConversationListeners {
  onStatus?: (status: ConversationStatus, detail?: string) => void;
  onProgress?: (msg: string) => void;
  onSubtitle?: (sub: SubtitleEvent) => void;
  onError?: (err: string) => void;
  onEmotion?: (e: PatientEmotion) => void;
}

function detectEmotion(text: string): PatientEmotion {
  const t = text.toLowerCase();
  if (/\b(hurts?|pain(?:ful)?|ache|aching|ow|ouch|sore|burn(?:s|ing)?|sharp|throb|stab|rip(?:ping)?|tear(?:ing)?|crush(?:ing)?|killing me|agony|can'?t breathe|chest)\b/.test(t)) return 'pain';
  if (/\b(scared|afraid|frightened|terrified|worr(?:ied|y)|nervous|anxious|help me|please help|dying|die|gonna die|save me)\b/.test(t)) return 'fear';
  if (/\b(okay|i'?m okay|fine|better|good now|thanks|thank you|relieved?|easier|breathing better|no more pain)\b/.test(t)) return 'relief';
  if (/\b(i don'?t (?:know|understand)|not sure|confused|what does that mean|what do you mean|sorry what)\b/.test(t)) return 'confused';
  return 'neutral';
}

export interface ConversationOptions {
  systemPrompt?: string;
  initialMessage?: { role: 'assistant'; content: string };
  /** Speaker gender ('M'|'F') — used by backend voice picker. */
  voiceGender?: 'M' | 'F';
  /** Stable case id — backend uses this for logging + voice slot pick. */
  caseId?: string;
  /** Preserved for back-compat. Ignored — voice ID is chosen server-side now. */
  voice?: string;
  /** If set, persist/restore the conversation history to localStorage under this key. */
  storageKey?: string;
}

interface VoiceTokenResponse {
  token: string;
  url: string;
  roomName: string;
}

async function fetchVoiceToken(opts: {
  caseId: string;
  systemPrompt: string;
  initialLine: string;
  gender: 'M' | 'F';
  isPediatric?: boolean;
}): Promise<VoiceTokenResponse> {
  const r = await fetch('/voice/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  });
  if (!r.ok) throw new Error(`/voice/token ${r.status}: ${await r.text().catch(() => '')}`);
  return r.json();
}

export class Conversation {
  private messages: ChatMessage[] = [];
  private listeners: ConversationListeners = {};
  private status: ConversationStatus = 'uninitialized';

  private room: Room | null = null;
  private remoteAudioTrack: RemoteAudioTrack | null = null;
  private analyser: AnalyserNode | null = null;
  private audioCtx: AudioContext;
  private ampBuf: Uint8Array;

  private systemPrompt: string;
  private initialMessage: { role: 'assistant'; content: string };
  private voiceGender: 'M' | 'F';
  private caseId: string;
  private storageKey: string | null = null;

  private messageSubscribers = new Set<(msgs: ReadonlyArray<ChatMessage>) => void>();

  // Latest agent text spoken — drives the 3D face emotion.
  private currentEmotion: PatientEmotion = 'neutral';

  constructor(
    audioCtx: AudioContext,
    listeners: ConversationListeners = {},
    options: ConversationOptions = {}
  ) {
    this.audioCtx = audioCtx;
    this.listeners = listeners;
    this.systemPrompt = options.systemPrompt ?? FALLBACK_PERSONA;
    this.initialMessage = options.initialMessage ?? FALLBACK_INITIAL;
    this.voiceGender = options.voiceGender ?? 'M';
    this.caseId = options.caseId ?? 'unknown';
    this.storageKey = options.storageKey ?? null;
    this.ampBuf = new Uint8Array(1024);

    const restored = this.loadMessages();
    this.messages = restored && restored.length > 0 ? restored : [this.initialMessage];
  }

  private saveMessages() {
    if (!this.storageKey) return;
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.messages));
    } catch { /* quota or privacy-mode — silently ignore */ }
  }

  private loadMessages(): ChatMessage[] | null {
    if (!this.storageKey) return null;
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return null;
      return parsed.filter(
        (m: any) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string'
      );
    } catch {
      return null;
    }
  }

  setListeners(listeners: ConversationListeners) {
    this.listeners = listeners;
  }

  getStatus(): ConversationStatus {
    return this.status;
  }

  getMessages(): ReadonlyArray<ChatMessage> {
    return [...this.messages];
  }

  subscribeMessages(fn: (msgs: ReadonlyArray<ChatMessage>) => void): () => void {
    this.messageSubscribers.add(fn);
    return () => { this.messageSubscribers.delete(fn); };
  }

  private emitMessages() {
    const snap = [...this.messages];
    this.messageSubscribers.forEach((fn) => fn(snap));
  }

  /** Live mouth amplitude from the AnalyserNode tapped onto the remote
   *  audio track. 0..1, usable directly as a mouth-open factor. */
  getMouthAmplitude(): number {
    if (!this.analyser) return 0;
    if (this.ampBuf.length !== this.analyser.frequencyBinCount) {
      this.ampBuf = new Uint8Array(this.analyser.frequencyBinCount);
    }
    this.analyser.getByteTimeDomainData(this.ampBuf as any);
    let sum = 0;
    const n = this.ampBuf.length;
    for (let i = 0; i < n; i++) {
      const d = (this.ampBuf[i] - 128) / 128;
      sum += d * d;
    }
    const rms = Math.sqrt(sum / n);
    // Cartesia output peaks around ~0.3 RMS on loud vowels — same multiplier
    // we used for Kokoro looks right here too.
    return Math.min(1, rms * 3.2);
  }

  getCurrentEmotion(): PatientEmotion {
    return this.currentEmotion;
  }

  private setEmotion(e: PatientEmotion) {
    if (e === this.currentEmotion) return;
    this.currentEmotion = e;
    this.listeners.onEmotion?.(e);
  }

  private setStatus(s: ConversationStatus, detail?: string) {
    this.status = s;
    this.listeners.onStatus?.(s, detail);
  }

  private attachAnalyser(track: RemoteAudioTrack) {
    try {
      const mst = track.mediaStreamTrack;
      if (!mst) return;
      const stream = new MediaStream([mst]);
      const source = this.audioCtx.createMediaStreamSource(stream);
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.55;
      source.connect(analyser);
      // We do NOT connect analyser to destination — LiveKit attaches the
      // track to a hidden <audio> element for actual playback. The analyser
      // is read-only.
      this.analyser = analyser;
      this.ampBuf = new Uint8Array(analyser.frequencyBinCount);
    } catch (err) {
      console.warn('analyser attach failed:', err);
    }
  }

  private detachAnalyser() {
    this.analyser = null;
  }

  private wireRoomEvents(room: Room) {
    room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack, _pub: RemoteTrackPublication, participant: RemoteParticipant) => {
      if (track.kind === Track.Kind.Audio) {
        this.remoteAudioTrack = track as RemoteAudioTrack;
        this.attachAnalyser(this.remoteAudioTrack);
        // LiveKit auto-attaches remote audio for playback in the page;
        // call attach() explicitly to be safe across browsers.
        const el = (track as RemoteAudioTrack).attach();
        el.style.display = 'none';
        document.body.appendChild(el);
        // Tag for dispose-time cleanup.
        (el as any).__convRoomId = room.name;
        void participant;
      }
    });

    room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
      if (track === this.remoteAudioTrack) {
        try { (track as RemoteAudioTrack).detach().forEach((el) => el.remove()); } catch { /* noop */ }
        this.remoteAudioTrack = null;
        this.detachAnalyser();
      }
    });

    // Transcription events fire for both local (user) and remote (agent)
    // participants. We distinguish by participant identity vs. local id.
    room.on(RoomEvent.TranscriptionReceived, (segments: TranscriptionSegment[], participant?: Participant) => {
      const isLocal = participant?.identity === room.localParticipant.identity;
      for (const seg of segments) {
        if (isLocal) {
          this.handleUserTranscription(seg);
        } else {
          this.handleAgentTranscription(seg);
        }
      }
    });

    room.on(RoomEvent.Disconnected, () => {
      this.setStatus('uninitialized');
    });
  }

  private handleUserTranscription(seg: TranscriptionSegment) {
    if (!seg.text) return;
    if (!seg.final) {
      this.listeners.onSubtitle?.({ who: 'you', text: seg.text, partial: true });
      if (this.status !== 'listening') this.setStatus('listening');
      return;
    }
    const final = seg.text.trim();
    if (!final) return;
    this.listeners.onSubtitle?.({ who: 'you', text: final });
    this.messages.push({ role: 'user', content: final });
    this.saveMessages();
    this.emitMessages();
    this.setStatus('thinking', 'Thinking…');
  }

  private handleAgentTranscription(seg: TranscriptionSegment) {
    if (!seg.text) return;
    if (!seg.final) {
      this.setEmotion(detectEmotion(seg.text));
      this.listeners.onSubtitle?.({ who: 'patient', text: seg.text, partial: true });
      if (this.status !== 'speaking') this.setStatus('speaking');
      return;
    }
    const final = seg.text.trim();
    if (!final) return;
    this.setEmotion(detectEmotion(final));
    this.listeners.onSubtitle?.({ who: 'patient', text: final });
    this.messages.push({ role: 'assistant', content: final });
    this.saveMessages();
    this.emitMessages();
    this.setStatus('ready');
  }

  async init() {
    if (this.status !== 'uninitialized') return;
    this.setStatus('loading');

    try {
      if (!hasClaudeKey()) {
        throw new Error('Patient backend unavailable.');
      }

      this.listeners.onProgress?.('Requesting voice session…');
      const initialLine = this.initialMessage.content;
      const tok = await fetchVoiceToken({
        caseId: this.caseId,
        systemPrompt: this.systemPrompt,
        initialLine,
        gender: this.voiceGender,
      });

      this.listeners.onProgress?.('Connecting to voice room…');
      const room = new Room({ adaptiveStream: true, dynacast: true });
      this.wireRoomEvents(room);
      await room.connect(tok.url, tok.token);
      this.room = room;

      // Resume audio context after the user gesture chain finishes so
      // both playback and the analyser tap can run.
      if (this.audioCtx.state === 'suspended') {
        try { await this.audioCtx.resume(); } catch { /* noop */ }
      }

      // Open the mic. Real-time means it stays open until dispose().
      this.listeners.onProgress?.('Enabling microphone…');
      await room.localParticipant.setMicrophoneEnabled(true);

      this.listeners.onProgress?.('Live.');
      this.setStatus('ready');

      // Always start a fresh encounter with a placeholder subtitle. The
      // patient's actual opening line will replace it as soon as the
      // agent's first transcription arrives.
      this.listeners.onSubtitle?.({ who: 'patient', text: '…' });
      this.setEmotion(detectEmotion(initialLine));
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      console.error('Conversation init failed:', err);
      this.setStatus('error', msg);
      this.listeners.onError?.(msg);
      throw err;
    }
  }

  /** Back-compat no-op. Real-time means mic is always live. */
  async startListening() {
    if (this.status === 'uninitialized') return;
    if (this.room) {
      try { await this.room.localParticipant.setMicrophoneEnabled(true); } catch { /* noop */ }
    }
  }

  /** Back-compat no-op. The agent decides turn boundaries via VAD. */
  async stopListeningAndRespond() {
    /* no-op */
  }

  cancel() {
    // Best-effort: mute mic until next turn. The agent's VAD will pick
    // back up when we unmute, but real-time has no abort signal we own.
    if (this.room) {
      this.room.localParticipant.setMicrophoneEnabled(false).catch(() => undefined);
    }
    this.setStatus('ready');
  }

  /** Trigger the LiveKit voice agent to speak ONE short farewell out loud
   *  and wait until the audio finishes before resolving. We publish a tiny
   *  data-channel signal; the Python worker listens and calls
   *  `session.generate_reply()` with a farewell instruction so the patient
   *  actually *speaks* the goodbye via Cartesia TTS. Falls back to the
   *  HTTP text-only path when voice isn't connected. */
  async sayFarewell(): Promise<void> {
    this.setStatus('thinking', 'Saying goodbye…');
    this.listeners.onSubtitle?.({ who: 'patient', text: '…' });

    // Stop the doctor's mic so the agent's farewell can't trigger another
    // STT round-trip and a follow-up reply.
    if (this.room) {
      try { await this.room.localParticipant.setMicrophoneEnabled(false); } catch { /* noop */ }
    }

    if (this.room && this.room.state === 'connected') {
      try {
        // RPC into the agent participant. We're the only OTHER participant
        // (doctor) in the room, so any remote participant is the patient
        // agent. Match by identity prefix or, failing that, take the first.
        const remotes = Array.from(this.room.remoteParticipants.values());
        console.log('[farewell] remote participants:', remotes.map((p) => p.identity));
        const agent =
          remotes.find((p) => p.identity.startsWith('agent-')) ?? remotes[0];
        if (agent) {
          console.log('[farewell] performing RPC to', agent.identity);
          const result = await this.room.localParticipant.performRpc({
            destinationIdentity: agent.identity,
            method: 'farewell',
            payload: '',
          });
          console.log('[farewell] RPC result:', result);
        } else {
          console.warn('[farewell] no agent participant in room');
        }
      } catch (err) {
        console.error('[farewell] RPC failed:', err);
      }

      // Wait for the agent to actually start speaking (status flips to
      // 'speaking' when LiveKit forwards the first transcription segment),
      // then wait for it to drop back to 'ready'. Pad a short tail so the
      // last syllable plays through before the room is torn down. Hard cap
      // at 7 s in case events are missed.
      const deadline = Date.now() + 7000;
      let sawSpeaking = false;
      while (Date.now() < deadline) {
        if (this.status === 'speaking') sawSpeaking = true;
        if (sawSpeaking && this.status !== 'speaking') break;
        await new Promise((r) => window.setTimeout(r, 100));
      }
      // Tail flush — Cartesia stream lags the final transcription by ~0.5s.
      await new Promise((r) => window.setTimeout(r, 800));
      // Wipe persisted history so a re-encounter with this case doesn't
      // pop the farewell as the "last assistant message" on greeting.
      if (this.storageKey) {
        try { localStorage.removeItem(this.storageKey); } catch { /* noop */ }
      }
      this.setStatus('ready');
      return;
    }

    // Voice not connected — fall back to the old text-only flow so the
    // subtitle still shows a goodbye line.
    const farewellPrompt =
      "Okay, we're all done. Take care of yourself. Goodbye. " +
      "[This is your final reply. Say a short goodbye — thanks, okay, bye — " +
      "and NOTHING ELSE. Do NOT ask any questions. No follow-ups. " +
      "One or two short sentences, then silence.]";
    this.messages.push({ role: 'user', content: farewellPrompt });
    this.emitMessages();

    const controller = new AbortController();
    let assistantText = '';
    try {
      for await (const token of streamClaude(this.systemPrompt, this.messages, controller.signal)) {
        assistantText += token;
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') console.error('Farewell LLM error:', err);
    }
    const trimmed = assistantText.trim();
    if (trimmed) {
      this.setEmotion(detectEmotion(trimmed));
      this.listeners.onSubtitle?.({ who: 'patient', text: trimmed });
      this.messages.push({ role: 'assistant', content: trimmed });
      this.saveMessages();
      this.emitMessages();
    }
    this.setStatus('ready');
  }

  /** Text-chat turn — used by PatientChatPanel. Routes through the legacy
   *  /agent/patient/stream so it doesn't fight the live voice session. */
  async sendTextMessage(text: string, _opts?: { speak?: boolean }): Promise<void> {
    const clean = text.trim();
    if (!clean) return;
    if (this.status !== 'ready') return;

    this.listeners.onSubtitle?.({ who: 'you', text: clean });
    this.messages.push({ role: 'user', content: clean });
    this.saveMessages();
    this.emitMessages();
    this.setStatus('thinking', 'Thinking…');

    const controller = new AbortController();
    let assistantText = '';
    try {
      for await (const token of streamClaude(this.systemPrompt, this.messages, controller.signal)) {
        assistantText += token;
      }
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        console.error('Text-chat LLM error:', err);
        this.listeners.onError?.(`Claude failed: ${err?.message ?? err}`);
        this.setStatus('ready');
        return;
      }
    }

    const trimmed = assistantText.trim();
    if (trimmed) {
      this.setEmotion(detectEmotion(trimmed));
      this.listeners.onSubtitle?.({ who: 'patient', text: trimmed });
      this.messages.push({ role: 'assistant', content: trimmed });
      this.saveMessages();
      this.emitMessages();
    }
    this.setStatus('ready');
  }

  reset() {
    this.cancel();
    this.messages = [this.initialMessage];
    this.emitMessages();
    if (this.storageKey) {
      try { localStorage.removeItem(this.storageKey); } catch { /* noop */ }
    }
  }

  dispose() {
    if (this.room) {
      try { this.room.disconnect(); } catch { /* noop */ }
      this.room = null;
    }
    this.detachAnalyser();
    this.remoteAudioTrack = null;
    // Remove any audio elements we appended for playback.
    document.querySelectorAll('audio[__convRoomId]').forEach((el) => el.remove());
    this.setStatus('uninitialized');
  }
}
