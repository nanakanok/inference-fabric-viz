// Autoplay "pipeline" view (ADR §7): several batches overlap in time.
//
// PD on, one repeating cycle of 4 slots with three batches in flight
// (n = newest, n-1, n-2):
//   - Prefill of batch n: two representative MoE layers, each dispatch then combine.
//     Slots 0-1 finish layers 1-30, slots 2-3 finish layers 31-61.
//   - Layer-wise KV: slots 2-3 stream batch n's layers 1-30 (already finished);
//     slots 0-1 stream batch n-1's layers 31-61 (finished in the previous cycle).
//   - Decode of batch n-2, whose KV completed in an earlier cycle.
// PD off: the same GPUs alternate between Prefill of a batch and its Decode.
// AF on the decode side: two micro-batches ping-pong, so A->F of one overlaps
// F->A of the other. This is an example of overlap, not a claim about the
// optimal micro-batch count or full latency hiding.

import { LAYERS } from './constants';
import type { I18n } from '../content/text';
import { buildRoles, type Roles, type Scenario } from './scenario';
import type { Activity, Stage } from './steps';
import { kvTraffic, moeLayerTraffic, range, type Traffic } from './traffic';
import { assignDecode, buildWorkload, prefillGpuOf, REQUESTS, WORKLOAD_SEED, type Workload } from './workload';

export const MICRO_BATCHES = 2;
/** Prefill layers finished by the end of the first half of a batch's prefill. */
export const FIRST_HALF_LAYERS = 30;

export type Lane = 'prefill' | 'kv' | 'decode';

export interface DecodeOp {
  direction: 'send' | 'return';
  /** Micro-batch index, or null when the whole batch moves together. */
  microBatch: number | null;
  iteration: number;
}

export interface LaneInfo {
  lane: Lane;
  /** Batch relative to the newest one: 0 = n, -1 = n-1, -2 = n-2. */
  batch: number;
  text: I18n;
  /** Prefill: layers finished after this slot. KV: layer range [from, to] sent in this slot. */
  layers?: [number, number];
  decodeOps?: DecodeOp[];
}

export interface Slot {
  index: number;
  stage: Stage;
  lanes: LaneInfo[];
}

export interface Pipeline {
  roles: Roles;
  /** Slots in order. When `cyclic`, the view loops and the sequence must be consistent across the wrap. */
  slots: Slot[];
  cyclic: boolean;
}

const t = (ja: string, en: string): I18n => ({ ja, en });

function prefix(traffic: Traffic, lane: string, scale = 1): Traffic {
  return {
    flows: traffic.flows.map((f) => ({ ...f, id: `${lane}:${f.id}`, logicalBytes: f.logicalBytes * scale })),
    transfers: traffic.transfers.map((x) => ({ ...x, id: `${lane}:${x.id}`, bytes: x.bytes * scale })),
  };
}

function merge(parts: Traffic[]): Traffic | null {
  if (!parts.length) return null;
  return { flows: parts.flatMap((p) => p.flows), transfers: parts.flatMap((p) => p.transfers) };
}

/** Workload seed of a pipeline batch relative to the newest one. */
export function pipelineSeed(rel: number): number {
  return WORKLOAD_SEED + 10 + rel;
}

const batchName = (rel: number) => (rel === 0 ? 'n' : `n${rel}`.replace('-', '−'));

/**
 * Decode operations per slot. Ping-pong (AF): micro-batch k%2 sends iteration
 * floor(k/2) while the other returns what it sent in the previous slot.
 * Without AF: send and return alternate for the whole batch.
 * Cyclic schedules wrap; linear ones drain the last send at the end.
 */
export function decodeSchedule(slots: number, pingPong: boolean, cyclic: boolean): DecodeOp[][] {
  const out: DecodeOp[][] = [];
  if (!pingPong) {
    for (let k = 0; k < slots; k++) {
      out.push([{ direction: k % 2 === 0 ? 'send' : 'return', microBatch: null, iteration: Math.floor(k / 2) }]);
    }
    return out;
  }
  const sends = cyclic ? slots : slots - 1;
  for (let k = 0; k < slots; k++) {
    const ops: DecodeOp[] = [];
    if (k < sends) ops.push({ direction: 'send', microBatch: k % 2, iteration: Math.floor(k / 2) });
    const prev = cyclic ? (k - 1 + slots) % slots : k - 1;
    if (prev >= 0) ops.push({ direction: 'return', microBatch: prev % 2, iteration: Math.floor(prev / 2) });
    out.push(ops);
  }
  return out;
}

export function buildPipeline(scenario: Scenario, prompt: string, outputTokens: number): Pipeline {
  const roles = buildRoles(scenario);
  const decodeOf = assignDecode(roles);
  const batch = (rel: number): Workload => buildWorkload(prompt, outputTokens, pipelineSeed(rel));

  const prefillLayer = (w: Workload, direction: 'send' | 'return') =>
    moeLayerTraffic({
      scenario,
      roles,
      workload: w,
      phase: 'prefill',
      direction,
      attnGpuOf: (r) => prefillGpuOf(roles, r),
      tokensOf: (r) => range(w.requests[r]!.promptTokens),
    });
  const decodeLayer = (w: Workload, op: DecodeOp) =>
    moeLayerTraffic({
      scenario,
      roles,
      workload: w,
      phase: 'decode',
      direction: op.direction,
      attnGpuOf: (r) => decodeOf.get(r)!,
      tokensOf: (r) => (op.microBatch === null || r % MICRO_BATCHES === op.microBatch ? [op.iteration] : []),
    });

  const prefillAttn = [...new Set(range(REQUESTS).map((r) => prefillGpuOf(roles, r)))];
  const decodeAttn = [...new Set(decodeOf.values())];
  const pingPong = roles.decode.disaggregated;

  const decodeLane = (w: Workload, rel: number, ops: DecodeOp[], parts: Traffic[], activity: Map<string, Activity>): LaneInfo => {
    for (const op of ops) {
      parts.push(prefix(decodeLayer(w, op), `d-${op.direction}-${op.microBatch ?? 'all'}`));
      for (const g of op.direction === 'send' ? decodeAttn : roles.decode.experts) {
        activity.set(g, op.direction === 'send' ? 'router' : 'experts');
      }
    }
    const b = batchName(rel);
    const describe = (op: DecodeOp, lang: 'ja' | 'en') => {
      const what = op.direction === 'send' ? (pingPong ? 'A→F' : 'dispatch') : pingPong ? 'F→A' : 'combine';
      if (op.microBatch === null) return lang === 'ja' ? `トークン ${op.iteration + 1} の ${what}` : `${what} for token ${op.iteration + 1}`;
      return lang === 'ja'
        ? `マイクロバッチ ${op.microBatch} がトークン ${op.iteration + 1} の ${what}`
        : `micro-batch ${op.microBatch} does ${what} for token ${op.iteration + 1}`;
    };
    const both = pingPong && ops.length === 2;
    return {
      lane: 'decode',
      batch: rel,
      decodeOps: ops,
      text: t(
        `Decode（バッチ ${b}）: ${ops.map((o) => describe(o, 'ja')).join('、')}${both ? '。A と F が同時に動きます' : ''}`,
        `Decode (batch ${b}): ${ops.map((o) => describe(o, 'en')).join('; ')}${both ? ', keeping A and F busy' : ''}`,
      ),
    };
  };

  const slots: Slot[] = [];
  const push = (parts: Traffic[], activity: Map<string, Activity>, lanes: LaneInfo[]) =>
    slots.push({ index: slots.length, stage: { activity, traffic: merge(parts) }, lanes });

  // Branch on effective roles: scale-up only keeps the PD toggle's state but has no KV transfer.
  if (roles.kvTransfer) {
    const newest = batch(0);
    const previous = batch(-1);
    const decoding = batch(-2);
    const kvNew = kvTraffic(scenario, roles, newest, decodeOf);
    const kvPrev = kvTraffic(scenario, roles, previous, decodeOf);
    const ops = decodeSchedule(4, pingPong, true);
    for (let k = 0; k < 4; k++) {
      const parts: Traffic[] = [];
      const activity = new Map<string, Activity>();
      const lanes: LaneInfo[] = [];
      const send = k % 2 === 0;
      const secondHalf = k >= 2;
      parts.push(prefix(prefillLayer(newest, send ? 'send' : 'return'), `p-${k}`));
      for (const g of send ? prefillAttn : roles.prefill.experts) activity.set(g, send ? 'router' : 'experts');
      const range_ = secondHalf ? `${FIRST_HALF_LAYERS + 1}–${LAYERS}` : `4–${FIRST_HALF_LAYERS}`;
      lanes.push({
        lane: 'prefill',
        batch: 0,
        layers: [1, send ? (secondHalf ? FIRST_HALF_LAYERS : 3) : secondHalf ? LAYERS : FIRST_HALF_LAYERS],
        text: t(
          `Prefill（バッチ n）: ${range_} 層目の代表 MoE 層の ${send ? 'dispatch' : 'combine'}`,
          `Prefill (batch n): ${send ? 'dispatch' : 'combine'} of the representative MoE layer for layers ${range_}`,
        ),
      });
      // KV of finished layers only: batch n's first half once it is done, otherwise batch n-1's tail.
      const kvLayers: [number, number] = secondHalf ? [1, FIRST_HALF_LAYERS] : [FIRST_HALF_LAYERS + 1, LAYERS];
      const share = (kvLayers[1] - kvLayers[0] + 1) / LAYERS / 2;
      parts.push(prefix(secondHalf ? kvNew : kvPrev, `kv${k}`, share));
      lanes.push({
        lane: 'kv',
        batch: secondHalf ? 0 : -1,
        layers: kvLayers,
        text: t(
          `KV（バッチ ${secondHalf ? 'n' : 'n−1'}）: 計算済みの ${kvLayers[0]}–${kvLayers[1]} 層目の KV を予約済みの Decode GPU へ（${(k % 2) + 1}/2）`,
          `KV (batch ${secondHalf ? 'n' : 'n−1'}): finished layers ${kvLayers[0]}–${kvLayers[1]} stream to the reserved Decode GPUs (${(k % 2) + 1}/2)`,
        ),
      });
      lanes.push(decodeLane(decoding, -2, ops[k]!, parts, activity));
      push(parts, activity, lanes);
    }
    return { roles, slots, cyclic: true };
  }

  // PD off: the same GPUs run Prefill of a batch, then its Decode; nothing overlaps across phases.
  const w = batch(0);
  for (const direction of ['send', 'return'] as const) {
    const activity = new Map<string, Activity>();
    for (const g of direction === 'send' ? prefillAttn : roles.prefill.experts) activity.set(g, direction === 'send' ? 'router' : 'experts');
    const what = roles.prefill.disaggregated ? (direction === 'send' ? 'A→F' : 'F→A') : direction === 'send' ? 'dispatch' : 'combine';
    push([prefix(prefillLayer(w, direction), `p-${direction}`)], activity, [
      {
        lane: 'prefill',
        batch: 0,
        text: t(
          `Prefill（バッチ n）: 代表 MoE 層の ${what}。この間、同じ GPU の Decode は待たされます`,
          `Prefill (batch n): ${what} of the representative MoE layer. Decode on these GPUs has to wait`,
        ),
      },
    ]);
  }
  for (const ops of decodeSchedule(pingPong ? 5 : 4, pingPong, false)) {
    const parts: Traffic[] = [];
    const activity = new Map<string, Activity>();
    const lane = decodeLane(w, 0, ops, parts, activity);
    push(parts, activity, [lane]);
  }
  return { roles, slots, cyclic: false };
}

export const PIPELINE_NOTE: I18n = t(
  '並行動作の例です。代表の層だけを描いています。マイクロバッチ 2 個のピンポンは通信を計算と重ねる一例で、最適な数でも、通信を完全に隠せることの保証でもありません。',
  'An illustration of overlap using representative layers. Two-micro-batch ping-pong is one way to overlap communication with compute; it is neither the optimal count nor a guarantee of fully hidden communication.',
);
