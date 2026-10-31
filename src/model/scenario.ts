// Mode -> GPU roles, expert placement and transfer methods (ADR §3, §5, §12, §17).

import { DEEPSEEK_V3, QWEN3_30B_A3B, type ModelSpec } from './constants';
import type { Method, Purpose } from './routing';
import { buildTopology, GPUS_PER_NODE, gpuId, type Topology, type Wiring } from './topology';

export type Fabric = 'scaleout' | 'scaleup';

export interface Scenario {
  /**
   * 'scaleup': a lightweight model fits in one node, so every node is an
   * independent replica and no traffic leaves NVLink (PD/AF/wiring ignored).
   * Defaults to 'scaleout'.
   */
  fabric?: Fabric;
  pd: boolean;
  af: boolean;
  wiring: Wiring;
  /** Advanced setting: relay Decode MoE / A<->F traffic over NVLink too (default off). */
  decodeRelay: boolean;
  /** ECMP hash seed; re-rolling it must not change the workload (ADR §13). */
  ecmpSeed: number;
}

export type Phase = 'prefill' | 'decode';

export interface PhaseGroup {
  /** GPUs running attention (one DP replica each). */
  attn: string[];
  /** GPUs hosting routed experts (EP). */
  experts: string[];
  /** Attention and experts live on disjoint GPUs (AF disaggregation). */
  disaggregated: boolean;
  /** Each node holds a full expert set and serves only its own attention GPUs (scale-up only). */
  perNode?: boolean;
}

export interface Roles {
  prefill: PhaseGroup;
  decode: PhaseGroup;
  /** Prefill and decode run on different GPUs, so KV must move. */
  kvTransfer: boolean;
}

export type RoleLabel = 'unified' | 'prefill' | 'decode' | 'attention' | 'expert' | 'decode-attention' | 'decode-expert';

function nodesGpus(...nodes: number[]): string[] {
  return nodes.flatMap((n) => Array.from({ length: GPUS_PER_NODE }, (_, l) => gpuId(n, l)));
}

export function isScaleUpOnly(s: Pick<Scenario, 'fabric'>): boolean {
  return s.fabric === 'scaleup';
}

/** Reference model: lightweight MoE when everything fits in one node (ADR §21). */
export function modelFor(s: Pick<Scenario, 'fabric'>): ModelSpec {
  return isScaleUpOnly(s) ? QWEN3_30B_A3B : DEEPSEEK_V3;
}

export function buildRoles(s: Pick<Scenario, 'pd' | 'af' | 'fabric'>): Roles {
  const all = nodesGpus(0, 1, 2, 3);
  if (isScaleUpOnly(s)) {
    const g: PhaseGroup = { attn: all, experts: all, disaggregated: false, perNode: true };
    return { prefill: g, decode: g, kvTransfer: false };
  }
  if (!s.pd && !s.af) {
    const g: PhaseGroup = { attn: all, experts: all, disaggregated: false };
    return { prefill: g, decode: g, kvTransfer: false };
  }
  if (s.pd && !s.af) {
    return {
      prefill: { attn: nodesGpus(0, 1), experts: nodesGpus(0, 1), disaggregated: false },
      decode: { attn: nodesGpus(2, 3), experts: nodesGpus(2, 3), disaggregated: false },
      kvTransfer: true,
    };
  }
  if (!s.pd && s.af) {
    const g: PhaseGroup = { attn: nodesGpus(0, 1), experts: nodesGpus(2, 3), disaggregated: true };
    return { prefill: g, decode: g, kvTransfer: false };
  }
  return {
    prefill: { attn: nodesGpus(0, 1), experts: nodesGpus(0, 1), disaggregated: false },
    decode: { attn: nodesGpus(2), experts: nodesGpus(3), disaggregated: true },
    kvTransfer: true,
  };
}

function nodeOf(gpu: string): number {
  return Number(gpu.slice(1).split('.')[0]);
}

/**
 * Experts are placed in contiguous, equal blocks over the group's expert GPUs.
 * With per-node replicas, the owner is the GPU on the attention GPU's own node.
 */
export function expertOwner(group: PhaseGroup, expert: number, model: ModelSpec = DEEPSEEK_V3, attnGpu?: string): string {
  if (group.perNode) {
    if (!attnGpu) throw new Error('per-node placement needs the attention GPU');
    const per = model.experts / GPUS_PER_NODE;
    return gpuId(nodeOf(attnGpu), Math.floor(expert / per));
  }
  const per = model.experts / group.experts.length;
  return group.experts[Math.floor(expert / per)]!;
}

export function expertsOn(group: PhaseGroup, gpu: string, model: ModelSpec = DEEPSEEK_V3): number[] {
  if (group.perNode) {
    const per = model.experts / GPUS_PER_NODE;
    const local = Number(gpu.split('.')[1]);
    return Array.from({ length: per }, (_, k) => local * per + k);
  }
  const i = group.experts.indexOf(gpu);
  if (i < 0) return [];
  const per = model.experts / group.experts.length;
  return Array.from({ length: per }, (_, k) => i * per + k);
}

export function roleLabel(roles: Roles, gpu: string): RoleLabel {
  const { prefill, decode, kvTransfer } = roles;
  if (!kvTransfer) {
    if (!prefill.disaggregated) return 'unified';
    return prefill.attn.includes(gpu) ? 'attention' : 'expert';
  }
  if (prefill.attn.includes(gpu)) return 'prefill';
  if (!decode.disaggregated) return 'decode';
  return decode.attn.includes(gpu) ? 'decode-attention' : 'decode-expert';
}

export function sendPurpose(group: PhaseGroup): Purpose {
  return group.disaggregated ? 'a2f' : 'dispatch';
}

export function returnPurpose(group: PhaseGroup): Purpose {
  return group.disaggregated ? 'f2a' : 'combine';
}

/**
 * Prefill MoE traffic uses DeepEP Normal (NVLink relay); Decode uses
 * Low-Latency (direct) unless the decode-relay setting is on (ADR §12).
 */
export function methodFor(phase: Phase, direction: 'send' | 'return', s: Pick<Scenario, 'decodeRelay'>): Method {
  const relay = phase === 'prefill' || s.decodeRelay;
  if (!relay) return 'direct';
  return direction === 'send' ? 'relayRecv' : 'relaySendAggregate';
}

const topoCache = new Map<Wiring, Topology>();
export function topologyFor(wiring: Wiring): Topology {
  let t = topoCache.get(wiring);
  if (!t) {
    t = buildTopology(wiring);
    topoCache.set(wiring, t);
  }
  return t;
}
