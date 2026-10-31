// Inside-the-GPU view (ADR §22): one representative GPU per role (U, P/D,
// A/F, P/D-A/D-F), showing which compute blocks are active, which local
// experts the tracked request selected, layer progress and an HBM breakdown.

import type { I18n } from '../content/text';
import { DEEPSEEK_V3, type ModelSpec } from './constants';
import { expertOwner, expertsOn, modelFor, type Roles, type Scenario } from './scenario';
import type { Activity, Stage, StepPlan } from './steps';
import type { StepKind } from '../content/text';
import { MICRO_BATCHES, pipelineSeed, type Slot } from './pipeline';
import { prefillGpuOf, tokenRoute, TRACKED_REQUEST, type Workload } from './workload';
import { GPUS_PER_NODE } from './topology';

export type CardRole = 'U' | 'P' | 'D' | 'A' | 'F' | 'DA' | 'DF';
export type BlockId = 'embed' | 'attention' | 'kv' | 'router' | 'dense' | 'experts' | 'combine' | 'head' | 'send' | 'recv';

export interface HbmUsage {
  weights: number;
  kv: number;
  activations: number;
}

export interface GpuCard {
  role: CardRole;
  gpu: string;
  /** Blocks drawn on this card, in pipeline order. */
  blocks: BlockId[];
  active: Set<BlockId>;
  /** Routed experts hosted on this GPU and the ones the tracked request uses now. */
  experts: number[];
  lit: Set<number>;
  layer: I18n;
  hbm: HbmUsage;
  /** Token of the tracked request the card describes now (for the block zoom), or null if idle. */
  routeRef: RouteRef | null;
}

export interface RouteRef {
  phase: 'prefill' | 'decode';
  seed: number;
  tokenIdx: number;
  attnGpu: string;
}

const t = (ja: string, en: string): I18n => ({ ja, en });

/**
 * Parameter split used for weight estimates: routed-expert parameters and the
 * rest (attention, shared expert, dense FFN, embeddings), in bytes as served
 * (DeepSeek-V3: FP8 weights; Qwen3-30B-A3B: BF16 weights).
 */
export function weightSplit(m: ModelSpec): { expertBytes: number; otherBytes: number } {
  if (m.id === DEEPSEEK_V3.id) {
    // 58 MoE layers x 256 experts x (3 x 7168 x 2048) params, FP8; ~671B params total.
    const expert = 58 * 256 * 3 * 7168 * 2048;
    return { expertBytes: expert, otherBytes: 671e9 - expert };
  }
  // 48 layers x 128 experts x (3 x 2048 x 768) params, BF16; ~30.5B params total.
  const expert = 48 * 128 * 3 * 2048 * 768;
  return { expertBytes: expert * 2, otherBytes: (30.5e9 - expert) * 2 };
}

/** Which blocks an activity lights up. */
const ACTIVE_BLOCKS: Record<Activity, BlockId[]> = {
  scheduler: ['embed'],
  attention: ['attention', 'kv'],
  dense: ['attention', 'kv', 'dense'],
  router: ['attention', 'kv', 'router', 'send'],
  experts: ['recv', 'experts', 'send'],
  aggregate: ['recv', 'combine'],
  head: ['combine', 'head'],
};

function blocksFor(role: CardRole, m: ModelSpec): BlockId[] {
  const dense: BlockId[] = m.denseLayers > 0 ? ['dense'] : [];
  switch (role) {
    case 'A':
    case 'DA':
      return ['embed', 'attention', 'kv', ...dense, 'router', 'send', 'recv', 'combine', 'head'];
    case 'F':
    case 'DF':
      return ['recv', 'experts', 'send'];
    default:
      return ['embed', 'attention', 'kv', ...dense, 'router', 'experts', 'combine', 'head'];
  }
}

export interface Representatives {
  cards: { role: CardRole; gpu: string }[];
}

/** Representative GPUs: the ones the tracked request runs on; F is the owner of its first decode expert. */
export function representatives(
  scenario: Scenario,
  roles: Roles,
  decodeOf: Map<number, string>,
  workload: Workload,
  decodeSeed = workload.seed,
): Representatives {
  const m = modelFor(scenario);
  const p = prefillGpuOf(roles, TRACKED_REQUEST);
  const d = decodeOf.get(TRACKED_REQUEST)!;
  const firstExpert = tokenRoute(TRACKED_REQUEST, 'decode', 0, decodeSeed, m.experts, m.topK).experts[0]!;
  const fOwner = expertOwner(roles.decode, firstExpert, m, d);
  if (!roles.kvTransfer && !roles.decode.disaggregated) return { cards: [{ role: 'U', gpu: p }] };
  if (!roles.kvTransfer) return { cards: [{ role: 'A', gpu: p }, { role: 'F', gpu: fOwner }] };
  if (!roles.decode.disaggregated) return { cards: [{ role: 'P', gpu: p }, { role: 'D', gpu: d }] };
  return { cards: [{ role: 'P', gpu: p }, { role: 'DA', gpu: d }, { role: 'DF', gpu: fOwner }] };
}

/** Where the tracked request is, for KV and token accounting. */
export interface Progress {
  kind: StepKind | 'pipeline';
  /** Fraction of the prompt's KV built on the prefill GPU (0..1). */
  prefillKv: number;
  /** KV has reached the decode GPU. */
  kvMoved: boolean;
  /** Output tokens generated so far. */
  generated: number;
  phase: 'prefill' | 'decode';
  decodeIteration: number;
}

export function progressOf(kind: StepKind, outputSoFar: number, decodeIteration: number | undefined, layers: number, denseLayers: number): Progress {
  const base = { kind, generated: outputSoFar, decodeIteration: decodeIteration ?? 0 };
  switch (kind) {
    case 'arrive':
      return { ...base, prefillKv: 0, kvMoved: false, phase: 'prefill' };
    case 'prefill-dense':
      return { ...base, prefillKv: denseLayers / layers, kvMoved: false, phase: 'prefill' };
    case 'prefill-moe':
      return { ...base, prefillKv: 0.5, kvMoved: false, phase: 'prefill' };
    case 'prefill-done':
      return { ...base, prefillKv: 1, kvMoved: false, phase: 'prefill' };
    case 'kv':
      return { ...base, prefillKv: 1, kvMoved: true, phase: 'prefill' };
    default:
      return { ...base, prefillKv: 1, kvMoved: true, phase: 'decode' };
  }
}

function layerText(kind: StepKind | 'pipeline', m: ModelSpec): I18n {
  const firstMoe = m.denseLayers + 1;
  switch (kind) {
    case 'arrive':
      return t('割り当て待ち', 'Being scheduled');
    case 'prefill-dense':
      return t(`層 1–${m.denseLayers} / ${m.layers}（dense）`, `Layers 1–${m.denseLayers} / ${m.layers} (dense)`);
    case 'prefill-moe':
    case 'decode-attn':
    case 'decode-send':
    case 'decode-return':
      return t(`代表 MoE 層（${firstMoe}–${m.layers} 層目）`, `Representative MoE layer (${firstMoe}–${m.layers})`);
    case 'prefill-done':
    case 'decode-done':
      return t(`${m.layers} / ${m.layers} 層 → 出力 head`, `${m.layers} / ${m.layers} layers → output head`);
    case 'kv':
      return t('KV Cache を転送中', 'Transferring KV Cache');
    case 'repeat':
      return t('Decode を繰り返し', 'Decode repeats');
    case 'pipeline':
      return t('並行動作（代表層）', 'Overlap (representative layers)');
  }
}

export interface CardInput {
  scenario: Scenario;
  roles: Roles;
  decodeOf: Map<number, string>;
  workload: Workload;
  stage: Stage | null;
  progress: Progress;
  /**
   * Which routing the expert highlights follow. Omitted in step-through mode
   * (derived from progress). The overlap view passes the batch seeds and the
   * tracked request's decode iteration for the current slot, or null when the
   * tracked request has no traffic of that phase in the slot.
   */
  lit?: {
    prefill: { seed: number } | null;
    decode: { seed: number; iteration: number } | null;
    /** Seed of the decoding batch, used to pick the representative F GPU. */
    decodeRepSeed: number;
  };
}

export function gpuCards(input: CardInput): GpuCard[] {
  const { scenario, roles, decodeOf, workload, stage, progress } = input;
  const m = modelFor(scenario);
  const split = weightSplit(m);
  const reps = representatives(scenario, roles, decodeOf, workload, input.lit?.decodeRepSeed);
  const kvPerToken = m.kvBytesPerToken;
  const requestsOn = (gpu: string, of: (r: number) => string) => workload.requests.filter((r) => of(r.id) === gpu);

  return reps.cards.map(({ role, gpu }) => {
    const isAttn = role !== 'F' && role !== 'DF';
    const isPrefill = role === 'P' || role === 'U' || role === 'A';
    const groupPhase = role === 'P' ? 'prefill' : 'decode';
    const group = roles[groupPhase];
    const experts = expertsOn(group, gpu, m);
    const hostsExperts = experts.length > 0 && role !== 'A' && role !== 'DA';

    // Weights: attention-side GPUs hold the non-expert weights; expert-hosting GPUs hold their share.
    const weights = (isAttn ? split.otherBytes : 0) + (hostsExperts ? (split.expertBytes / m.experts) * experts.length : 0);

    // KV of the requests this GPU serves.
    let kv = 0;
    let activations = 0;
    if (isAttn) {
      const served = isPrefill ? requestsOn(gpu, (r) => prefillGpuOf(roles, r)) : requestsOn(gpu, (r) => decodeOf.get(r)!);
      for (const r of served) {
        const prompt = r.promptTokens;
        if (role === 'P') kv += progress.kvMoved && progress.kind !== 'pipeline' ? 0 : prompt * progress.prefillKv * kvPerToken;
        else if (role === 'D' || role === 'DA') kv += progress.kvMoved ? (prompt + Math.max(0, progress.generated - 1)) * kvPerToken : 0;
        else kv += (prompt * progress.prefillKv + Math.max(0, progress.generated - 1)) * kvPerToken;
        activations += (progress.phase === 'prefill' ? prompt : 1) * m.hidden * 2;
      }
    } else {
      // Expert GPU: activations of the tokens routed to it (one decode token per request, BF16).
      activations = workload.requests.length * m.hidden * 2;
    }

    const active = new Set<BlockId>();
    const act = stage?.activity.get(gpu);
    if (act) for (const b of ACTIVE_BLOCKS[act]) active.add(b);
    if (progress.kind === 'kv' && (role === 'P' || role === 'D' || role === 'DA')) active.add('kv');
    const blocks = blocksFor(role, m);
    for (const b of [...active]) if (!blocks.includes(b)) active.delete(b);

    // Which token of the tracked request this card is about right now.
    // P cards always follow prefill routing; the others follow the current phase.
    let phase: 'prefill' | 'decode' | null = role === 'P' ? 'prefill' : progress.phase;
    let seed = workload.seed;
    let iteration = progress.decodeIteration;
    if (input.lit) {
      const { prefill, decode } = input.lit;
      if (role === 'P') phase = prefill ? 'prefill' : null;
      else if (decode) phase = 'decode';
      else phase = prefill && !roles.kvTransfer ? 'prefill' : null;
      if (phase === 'prefill') seed = prefill!.seed;
      if (phase === 'decode') {
        seed = decode!.seed;
        iteration = decode!.iteration;
      }
    }
    // Decode-side cards (D, D-A, D-F) take no part in Prefill when PD is on.
    if (phase === 'prefill' && roles.kvTransfer && (role === 'D' || role === 'DA' || role === 'DF')) phase = null;
    const routeRef: RouteRef | null = phase
      ? { phase, seed, tokenIdx: phase === 'prefill' ? 0 : iteration, attnGpu: phase === 'prefill' ? prefillGpuOf(roles, TRACKED_REQUEST) : decodeOf.get(TRACKED_REQUEST)! }
      : null;

    // Experts the tracked request selects on this GPU (prefill: union over the first 64 prompt tokens).
    const lit = new Set<number>();
    if (hostsExperts && routeRef) {
      const tokens = routeRef.phase === 'prefill' ? Math.min(workload.requests[TRACKED_REQUEST]!.promptTokens, 64) : 1;
      const groupNow = roles[routeRef.phase];
      for (let i = 0; i < tokens; i++) {
        const idx = routeRef.phase === 'prefill' ? i : routeRef.tokenIdx;
        for (const e of tokenRoute(TRACKED_REQUEST, routeRef.phase, idx, routeRef.seed, m.experts, m.topK).experts) {
          if (expertOwner(groupNow, e, m, routeRef.attnGpu) === gpu) lit.add(e);
        }
      }
    }

    return {
      role,
      gpu,
      blocks,
      active,
      experts: hostsExperts ? experts : [],
      lit,
      layer: layerText(progress.kind, m),
      hbm: { weights, kv, activations },
      routeRef,
    };
  });
}

export function cardsForStep(plan: StepPlan, scenario: Scenario, workload: Workload, stepIndex: number, stageIndex: number): GpuCard[] {
  const step = plan.steps[stepIndex]!;
  const m = modelFor(scenario);
  return gpuCards({
    scenario,
    roles: plan.roles,
    decodeOf: plan.decodeOf,
    workload,
    stage: step.stages[stageIndex] ?? null,
    progress: progressOf(step.kind, step.outputSoFar, step.decodeIteration, m.layers, m.denseLayers),
  });
}

export function cardsForPipeline(plan: StepPlan, scenario: Scenario, workload: Workload, slot: Slot | null): GpuCard[] {
  // Batch seeds follow buildPipeline: PD on decodes batch n-2, PD off decodes batch n.
  const decodeRel = plan.roles.kvTransfer ? -2 : 0;
  const prefillLane = slot?.lanes.find((l) => l.lane === 'prefill');
  const decodeLane = slot?.lanes.find((l) => l.lane === 'decode');
  // The tracked request is request 0, so it belongs to micro-batch 0.
  const op = decodeLane?.decodeOps?.find((o) => o.microBatch === null || o.microBatch === TRACKED_REQUEST % MICRO_BATCHES);
  return gpuCards({
    scenario,
    roles: plan.roles,
    decodeOf: plan.decodeOf,
    workload,
    stage: slot?.stage ?? null,
    progress: { kind: 'pipeline', prefillKv: 0.5, kvMoved: true, generated: 2, phase: 'decode', decodeIteration: 0 },
    lit: {
      prefill: prefillLane ? { seed: pipelineSeed(prefillLane.batch) } : null,
      decode: decodeLane && op ? { seed: pipelineSeed(decodeLane.batch), iteration: op.iteration } : null,
      decodeRepSeed: pipelineSeed(decodeRel),
    },
  });
}

/** Byte value for a full HBM bar: the largest card total over every step (stage-less ones included) and slot. */
export function hbmScaleFor(plan: StepPlan, scenario: Scenario, workload: Workload, slots: Slot[] = []): number {
  const totals: number[] = [];
  plan.steps.forEach((step, i) => {
    for (let j = 0; j < Math.max(1, step.stages.length); j++) {
      for (const c of cardsForStep(plan, scenario, workload, i, j)) totals.push(c.hbm.weights + c.hbm.kv + c.hbm.activations);
    }
  });
  for (const slot of slots.length ? slots : [null]) {
    for (const c of cardsForPipeline(plan, scenario, workload, slot)) totals.push(c.hbm.weights + c.hbm.kv + c.hbm.activations);
  }
  return Math.max(1, ...totals);
}

export const LOCAL_GPUS = GPUS_PER_NODE;
