// Builds the traffic of one representative MoE layer (or one KV handoff) as
// logical flows plus deduplicated physical transfers (ADR §8, §12, §17).
//
// - Flow: logical (src GPU -> dst GPU) traffic with its full path, for the UI.
// - Transfer: one physical segment (an RDMA or NVLink leg) with the bytes that
//   actually cross it. Relays share legs, so a token sent to three GPUs on one
//   node by relay costs one RDMA leg plus NVLink fan-out, not three RDMA legs.

import { returnBytesPerToken, sendBytesPerToken } from './constants';
import { route, type Hop, type Method, type Path, type Purpose, type Segment } from './routing';
import {
  expertOwner,
  methodFor,
  modelFor,
  returnPurpose,
  sendPurpose,
  topologyFor,
  type Phase,
  type Roles,
  type Scenario,
} from './scenario';
import type { Topology } from './topology';
import { prefillGpuOf, tokenRoute, TRACKED_REQUEST, type Workload } from './workload';

export interface Flow {
  id: string;
  phase: Phase;
  purpose: Purpose;
  method: Method;
  /** Tracked request id, or 'batch' for the aggregated other requests. */
  request: number | 'batch';
  tracked: boolean;
  src: string;
  dst: string;
  path: Path;
  /** Tokens carried (after per-token destination dedup). */
  tokens: number;
  /** Bytes this flow would carry without relay sharing. */
  logicalBytes: number;
}

export interface Transfer {
  id: string;
  phase: Phase;
  purpose: Purpose;
  kind: Segment['kind'];
  tracked: boolean;
  from: string;
  to: string;
  hops: Hop[];
  bytes: number;
  tokens: number;
}

export interface Traffic {
  flows: Flow[];
  transfers: Transfer[];
}

export interface TrafficTotals {
  rdmaBytes: number;
  nvlinkBytes: number;
  /** RDMA bytes that traverse at least one spine. */
  spineBytes: number;
}

class Builder {
  private flows = new Map<string, Flow>();
  private transfers = new Map<string, Transfer>();
  private paths = new Map<string, Path>();

  constructor(
    private topo: Topology,
    private seed: number,
  ) {}

  path(src: string, dst: string, purpose: Purpose, method: Method): Path {
    const key = `${src}>${dst}|${purpose}|${method}`;
    let p = this.paths.get(key);
    if (!p) {
      p = route(this.topo, src, dst, purpose, method, { seed: this.seed });
      this.paths.set(key, p);
    }
    return p;
  }

  /**
   * Add one token's worth of traffic from src to each destination. Segments
   * shared by several destinations of the same token are charged once.
   */
  addToken(
    phase: Phase,
    purpose: Purpose,
    method: Method,
    request: number,
    pairs: [string, string][],
    bytesPerToken: number,
  ) {
    const tracked = request === TRACKED_REQUEST;
    const req = tracked ? request : 'batch';
    const legs = new Map<string, Segment>();
    for (const [src, dst] of pairs) {
      if (src === dst) continue;
      const path = this.path(src, dst, purpose, method);
      const fid = `${phase}|${purpose}|${req}|${src}>${dst}`;
      let f = this.flows.get(fid);
      if (!f) {
        f = { id: fid, phase, purpose, method, request: req, tracked, src, dst, path, tokens: 0, logicalBytes: 0 };
        this.flows.set(fid, f);
      }
      f.tokens++;
      f.logicalBytes += bytesPerToken;
      for (const seg of path.segments) legs.set(`${seg.kind}|${seg.from}>${seg.to}`, seg);
    }
    for (const [leg, seg] of legs) {
      const tid = `${phase}|${purpose}|${tracked ? 't' : 'b'}|${leg}`;
      let t = this.transfers.get(tid);
      if (!t) {
        t = { id: tid, phase, purpose, kind: seg.kind, tracked, from: seg.from, to: seg.to, hops: seg.hops, bytes: 0, tokens: 0 };
        this.transfers.set(tid, t);
      }
      t.tokens++;
      t.bytes += bytesPerToken;
    }
  }

  addBulk(phase: Phase, purpose: Purpose, method: Method, request: number, src: string, dst: string, tokens: number, bytes: number) {
    if (src === dst) return;
    const tracked = request === TRACKED_REQUEST;
    const req = tracked ? request : 'batch';
    const path = this.path(src, dst, purpose, method);
    const fid = `${phase}|${purpose}|${req}|${src}>${dst}`;
    const f = this.flows.get(fid) ?? { id: fid, phase, purpose, method, request: req, tracked, src, dst, path, tokens: 0, logicalBytes: 0 };
    f.tokens += tokens;
    f.logicalBytes += bytes;
    this.flows.set(fid, f);
    for (const seg of path.segments) {
      const tid = `${phase}|${purpose}|${tracked ? 't' : 'b'}|${seg.kind}|${seg.from}>${seg.to}`;
      const t = this.transfers.get(tid) ?? { id: tid, phase, purpose, kind: seg.kind, tracked, from: seg.from, to: seg.to, hops: seg.hops, bytes: 0, tokens: 0 };
      t.tokens += tokens;
      t.bytes += bytes;
      this.transfers.set(tid, t);
    }
  }

  result(): Traffic {
    return { flows: [...this.flows.values()], transfers: [...this.transfers.values()] };
  }
}

export interface MoeLayerInput {
  scenario: Scenario;
  roles: Roles;
  workload: Workload;
  phase: Phase;
  direction: 'send' | 'return';
  /** Attention GPU of each request in this phase. */
  attnGpuOf: (request: number) => string;
  /** Token indices processed per request in this layer (prompt positions or one decode index). */
  tokensOf: (request: number) => Iterable<number>;
}

/** Traffic of one representative MoE layer in one direction. */
export function moeLayerTraffic(input: MoeLayerInput): Traffic {
  const { scenario, roles, workload, phase, direction } = input;
  const group = roles[phase];
  const b = new Builder(topologyFor(scenario.wiring), scenario.ecmpSeed);
  const method = methodFor(phase, direction, scenario);
  const purpose = direction === 'send' ? sendPurpose(group) : returnPurpose(group);
  const model = modelFor(scenario);
  const bytes = direction === 'send' ? sendBytesPerToken(model) : returnBytesPerToken(model);
  for (const req of workload.requests) {
    const attn = input.attnGpuOf(req.id);
    for (const t of input.tokensOf(req.id)) {
      const route = tokenRoute(req.id, phase, t, workload.seed, model.experts, model.topK);
      const owners = new Set(route.experts.map((e) => expertOwner(group, e, model, attn)));
      const pairs: [string, string][] = [...owners].map((o) => (direction === 'send' ? [attn, o] : [o, attn]));
      b.addToken(phase, purpose, method, req.id, pairs, bytes);
    }
  }
  return b.result();
}

/** KV handoff from each request's prefill GPU to its reserved decode GPU. */
export function kvTraffic(scenario: Scenario, roles: Roles, workload: Workload, decodeOf: Map<number, string>): Traffic {
  const b = new Builder(topologyFor(scenario.wiring), scenario.ecmpSeed);
  if (!roles.kvTransfer) return b.result();
  for (const req of workload.requests) {
    const src = prefillGpuOf(roles, req.id);
    const dst = decodeOf.get(req.id)!;
    b.addBulk('prefill', 'kv', 'direct', req.id, src, dst, req.promptTokens, req.promptTokens * modelFor(scenario).kvBytesPerToken);
  }
  return b.result();
}

export function totals(transfers: Transfer[]): TrafficTotals {
  let rdmaBytes = 0;
  let nvlinkBytes = 0;
  let spineBytes = 0;
  for (const t of transfers) {
    if (t.kind === 'nvlink') {
      nvlinkBytes += t.bytes;
      continue;
    }
    rdmaBytes += t.bytes;
    if (t.hops.some((h) => h.to.startsWith('spine'))) spineBytes += t.bytes;
  }
  return { rdmaBytes, nvlinkBytes, spineBytes };
}

/** Bytes per directed link traversal: key `${linkId}|${from}>${to}`. */
export function linkLoad(transfers: Transfer[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const t of transfers) {
    for (const h of t.hops) {
      const k = `${h.link.id}|${h.from}>${h.to}`;
      out.set(k, (out.get(k) ?? 0) + t.bytes);
    }
  }
  return out;
}

export function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}
