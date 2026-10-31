// Deterministic workload: requests, schematic tokenization, router choices and
// decode placement. Everything derives from (prompt, workload seed) only, so
// toggling wiring, decode relay or the ECMP seed never changes it (ADR §7).

import { ROUTED_EXPERTS, TOP_K } from './constants';
import { hashFields, prng } from './hash';
import type { Phase, Roles } from './scenario';
import { GPU_COUNT } from './topology';

/** Fixed number of requests in a batch, independent of mode. */
export const REQUESTS = GPU_COUNT;
export const TRACKED_REQUEST = 0;
export const WORKLOAD_SEED = 20261031;
export const MAX_PROMPT_TOKENS = 8192;

export interface Request {
  id: number;
  promptTokens: number;
}

export interface Workload {
  prompt: string;
  /** Seed for request lengths and router choices. */
  seed: number;
  /** Schematic tokens of the tracked request's prompt (display only). */
  trackedTokens: string[];
  outputTokens: number;
  requests: Request[];
}

export interface TokenRoute {
  experts: number[];
  /** Router weights, aligned with experts; sum to 1. */
  weights: number[];
}

const CJK = /[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ]/;

/**
 * Schematic tokenizer: one token per CJK character, ASCII words split into
 * chunks of up to 4 characters. Not a real tokenizer (ADR §9).
 */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  let word = '';
  const flush = () => {
    for (let i = 0; i < word.length; i += 4) out.push(word.slice(i, i + 4));
    word = '';
  };
  for (const ch of text) {
    if (/\s/.test(ch)) {
      flush();
    } else if (CJK.test(ch) || /[^\p{L}\p{N}]/u.test(ch)) {
      flush();
      out.push(ch);
    } else {
      word += ch;
    }
  }
  flush();
  return out.slice(0, MAX_PROMPT_TOKENS);
}

export function buildWorkload(prompt: string, outputTokens: number, seed = WORKLOAD_SEED): Workload {
  const trackedTokens = tokenize(prompt);
  const base = Math.max(1, trackedTokens.length);
  const rand = prng(hashFields(seed, 'lengths', base));
  const requests: Request[] = Array.from({ length: REQUESTS }, (_, id) => ({
    id,
    promptTokens:
      id === TRACKED_REQUEST ? base : Math.min(MAX_PROMPT_TOKENS, Math.max(1, Math.round(base * (0.5 + rand())))),
  }));
  return { prompt, seed, trackedTokens, outputTokens: Math.max(1, outputTokens), requests };
}

/**
 * Router choice for one token: TOP_K distinct routed experts with weights.
 * Prefill tokens are indexed by prompt position, decode tokens by output index.
 */
export function tokenRoute(
  requestId: number,
  phase: Phase,
  tokenIdx: number,
  seed = WORKLOAD_SEED,
  expertCount = ROUTED_EXPERTS,
  topK = TOP_K,
): TokenRoute {
  const rand = prng(hashFields(seed, 'route', requestId, phase, tokenIdx));
  const experts: number[] = [];
  while (experts.length < topK) {
    const e = Math.floor(rand() * expertCount);
    if (!experts.includes(e)) experts.push(e);
  }
  const raw = experts.map(() => 0.05 + rand());
  const sum = raw.reduce((a, b) => a + b, 0);
  return { experts, weights: raw.map((w) => w / sum) };
}

/** Request i is served by prefill attention GPU i mod |attn|. */
export function prefillGpuOf(roles: Roles, requestId: number): string {
  return roles.prefill.attn[requestId % roles.prefill.attn.length]!;
}

function localOf(gpu: string): number {
  return Number(gpu.split('.')[1]);
}

/**
 * Decode placement, reserved when prefill starts (ADR §7). Prefers a decode
 * attention GPU with the same local index (= same rail in rail wiring; defined
 * on the local index so it is identical under both wirings), balancing load.
 */
export function assignDecode(roles: Roles): Map<number, string> {
  const out = new Map<number, string>();
  if (!roles.kvTransfer) {
    for (let r = 0; r < REQUESTS; r++) out.set(r, prefillGpuOf(roles, r));
    return out;
  }
  const load = new Map(roles.decode.attn.map((g) => [g, 0]));
  for (let r = 0; r < REQUESTS; r++) {
    const local = localOf(prefillGpuOf(roles, r));
    const same = roles.decode.attn.filter((g) => localOf(g) === local);
    const pool = same.length ? same : roles.decode.attn;
    let best = pool[0]!;
    for (const g of pool) if (load.get(g)! < load.get(best)!) best = g;
    load.set(best, load.get(best)! + 1);
    out.set(r, best);
  }
  return out;
}
