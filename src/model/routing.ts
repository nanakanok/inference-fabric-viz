// Path computation between GPUs. Purpose (what the traffic is for) and
// transfer method (how it is carried) are separate attributes (ADR §12).
// Switch traversal counts are always derived from the walked link path.

import { hashFields } from './hash';
import { gpuById, gpuId, nvswitchId, type Link, type Topology } from './topology';

export type Purpose = 'kv' | 'dispatch' | 'combine' | 'a2f' | 'f2a';

/**
 * - direct: send from the source GPU's own NIC (or NVLink inside a node)
 * - relayRecv: RDMA on the sender's rail to the same-local-index GPU on the
 *   destination node, then NVLink to the destination (DeepEP Normal dispatch)
 * - relaySendAggregate: NVLink on the source node to the GPU whose local index
 *   matches the destination's rail, then RDMA on that rail (DeepEP Normal combine)
 */
export type Method = 'direct' | 'relayRecv' | 'relaySendAggregate';

export interface Hop {
  link: Link;
  from: string;
  to: string;
}

export interface Segment {
  kind: 'nvlink' | 'rdma';
  from: string;
  to: string;
  hops: Hop[];
}

export interface Path {
  src: string;
  dst: string;
  purpose: Purpose;
  method: Method;
  segments: Segment[];
  /** Leaf and spine switches traversed by RDMA segments (the in-node scale-up switch is excluded). */
  switchCount: number;
  /** NVLink segments used as a relay in an inter-node path. */
  relayCount: number;
  /** Spine switches traversed. */
  spineCount: number;
}

export interface RouteOptions {
  /** ECMP hash seed (ADR §13). */
  seed: number;
}

/** UDP source port stand-in: one QP per (NIC pair, purpose) (ADR §13). */
export function pseudoSourcePort(srcNic: string, dstNic: string, purpose: Purpose): number {
  return 49152 + (hashFields(0, 'qp', srcNic, dstNic, purpose) % 16384);
}

function nvlinkSegment(topo: Topology, from: string, to: string): Segment {
  const a = gpuById(topo, from);
  const nvs = nvswitchId(a.node);
  const up = topo.nvlinkOfGpu.get(from)!;
  const down = topo.nvlinkOfGpu.get(to)!;
  return {
    kind: 'nvlink',
    from,
    to,
    hops: [
      { link: up, from, to: nvs },
      { link: down, from: nvs, to },
    ],
  };
}

function rdmaSegment(topo: Topology, from: string, to: string, purpose: Purpose, opts: RouteOptions): Segment {
  const a = gpuById(topo, from);
  const b = gpuById(topo, to);
  const leafA = topo.leafOfGpu.get(from)!;
  const leafB = topo.leafOfGpu.get(to)!;
  const hops: Hop[] = [{ link: topo.accessOfGpu.get(from)!, from, to: leafA }];

  if (leafA !== leafB) {
    const sport = pseudoSourcePort(a.nicId, b.nicId, purpose);
    const uplinks = [...topo.fabric.get(leafA)!.values()].flat();
    const up = uplinks[hashFields(opts.seed, 'up', a.nicId, b.nicId, sport) % uplinks.length]!;
    const spine = up.b;
    const downs = topo.fabric.get(leafB)!.get(spine)!;
    const down = downs[hashFields(opts.seed, 'down', spine, a.nicId, b.nicId, sport) % downs.length]!;
    hops.push({ link: up, from: leafA, to: spine }, { link: down, from: spine, to: leafB });
  }
  hops.push({ link: topo.accessOfGpu.get(to)!, from: leafB, to });
  return { kind: 'rdma', from, to, hops };
}

function summarize(src: string, dst: string, purpose: Purpose, method: Method, segments: Segment[]): Path {
  let switchCount = 0;
  let spineCount = 0;
  const interNode = segments.some((s) => s.kind === 'rdma');
  for (const seg of segments) {
    if (seg.kind !== 'rdma') continue;
    for (const hop of seg.hops) {
      // Count each switch once, on entry.
      if (hop.to.startsWith('leaf') || hop.to.startsWith('spine')) switchCount++;
      if (hop.to.startsWith('spine')) spineCount++;
    }
  }
  const relayCount = interNode ? segments.filter((s) => s.kind === 'nvlink').length : 0;
  return { src, dst, purpose, method, segments, switchCount, relayCount, spineCount };
}

export function route(
  topo: Topology,
  src: string,
  dst: string,
  purpose: Purpose,
  method: Method,
  opts: RouteOptions,
): Path {
  if (src === dst) return summarize(src, dst, purpose, method, []);
  const s = gpuById(topo, src);
  const d = gpuById(topo, dst);
  if (s.node === d.node) return summarize(src, dst, purpose, method, [nvlinkSegment(topo, src, dst)]);

  const segments: Segment[] = [];
  switch (method) {
    case 'direct':
      segments.push(rdmaSegment(topo, src, dst, purpose, opts));
      break;
    case 'relayRecv': {
      const mid = gpuId(d.node, s.local);
      segments.push(rdmaSegment(topo, src, mid, purpose, opts));
      if (mid !== dst) segments.push(nvlinkSegment(topo, mid, dst));
      break;
    }
    case 'relaySendAggregate': {
      const mid = gpuId(s.node, d.local);
      if (mid !== src) segments.push(nvlinkSegment(topo, src, mid));
      segments.push(rdmaSegment(topo, mid, dst, purpose, opts));
      break;
    }
  }
  return summarize(src, dst, purpose, method, segments);
}

/** Leaf/spine switches of a path, in order (for UI highlighting). */
export function pathSwitches(path: Path): string[] {
  const out: string[] = [];
  for (const seg of path.segments) {
    for (const hop of seg.hops) {
      if (hop.to.startsWith('leaf') || hop.to.startsWith('spine') || hop.to.startsWith('nvs')) out.push(hop.to);
    }
  }
  return out;
}
