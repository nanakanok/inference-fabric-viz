import { describe, expect, it } from 'vitest';
import { buildTopology, GPU_COUNT, gpuId, SPINES, type Topology, type Wiring } from '../src/model/topology';
import { route, type Method, type Path, type Purpose } from '../src/model/routing';

const WIRINGS: Wiring[] = ['rail', 'tor'];
const METHODS: Method[] = ['direct', 'relayRecv', 'relaySendAggregate'];
const opts = { seed: 1 };

function allPairs(topo: Topology): [string, string][] {
  const out: [string, string][] = [];
  for (const a of topo.gpus) for (const b of topo.gpus) if (a.id !== b.id) out.push([a.id, b.id]);
  return out;
}

/** Every hop uses a link of this topology between its endpoints, and hops chain from src to dst. */
function expectConnected(topo: Topology, path: Path) {
  if (path.src !== path.dst) expect(path.segments.length).toBeGreaterThan(0);
  const linkIds = new Set(topo.links.map((l) => l.id));
  let at = path.src;
  for (const seg of path.segments) {
    expect(seg.from).toBe(at);
    let cur = seg.from;
    for (const hop of seg.hops) {
      expect(hop.from).toBe(cur);
      expect(linkIds.has(hop.link.id)).toBe(true);
      const ends = [hop.link.a, hop.link.b];
      expect(ends).toContain(hop.from);
      expect(ends).toContain(hop.to);
      cur = hop.to;
    }
    expect(cur).toBe(seg.to);
    at = seg.to;
  }
  expect(at).toBe(path.dst);
}

describe('topology (ADR §4, §18)', () => {
  for (const wiring of WIRINGS) {
    it(`${wiring}: same NIC, uplink and spine port counts; 1:1 per leaf`, () => {
      const t = buildTopology(wiring);
      const access = t.links.filter((l) => l.kind === 'access');
      const fabric = t.links.filter((l) => l.kind === 'fabric');
      expect(access).toHaveLength(GPU_COUNT);
      expect(fabric).toHaveLength(32);
      for (let s = 0; s < SPINES; s++) {
        expect(fabric.filter((l) => l.b === `spine${s}`)).toHaveLength(8);
      }
      for (const leaf of t.switches.filter((x) => x.kind === 'leaf')) {
        const down = access.filter((l) => l.b === leaf.id).length;
        const up = fabric.filter((l) => l.a === leaf.id).length;
        expect(up).toBe(down);
      }
    });
  }

  it('rail: leaf i serves local GPU i of every node; tor: leaf n serves node n', () => {
    const rail = buildTopology('rail');
    const tor = buildTopology('tor');
    for (const g of rail.gpus) expect(rail.leafOfGpu.get(g.id)).toBe(`leaf${g.local}`);
    for (const g of tor.gpus) expect(tor.leafOfGpu.get(g.id)).toBe(`leaf${g.node}`);
    expect(rail.switches.filter((s) => s.kind === 'leaf')).toHaveLength(8);
    expect(tor.switches.filter((s) => s.kind === 'leaf')).toHaveLength(4);
  });
});

describe('routing (ADR §12, §15)', () => {
  const purposes: Purpose[] = ['kv', 'dispatch', 'combine', 'a2f', 'f2a'];

  for (const wiring of WIRINGS) {
    const topo = buildTopology(wiring);
    it(`${wiring}: every path walks only connected links`, () => {
      for (const [a, b] of allPairs(topo)) {
        for (const m of METHODS) expectConnected(topo, route(topo, a, b, 'dispatch', m, opts));
      }
    });

    it(`${wiring}: intra-node traffic is a single NVLink segment for every method`, () => {
      for (const m of METHODS) {
        const p = route(topo, gpuId(0, 1), gpuId(0, 6), 'dispatch', m, opts);
        expect(p.segments.map((s) => [s.kind, s.from, s.to])).toEqual([['nvlink', gpuId(0, 1), gpuId(0, 6)]]);
        expect(p.switchCount).toBe(0);
        expect(p.relayCount).toBe(0);
      }
    });
  }

  it('rail: same-rail inter-node transfer crosses exactly one leaf', () => {
    const t = buildTopology('rail');
    for (const purpose of purposes) {
      const p = route(t, gpuId(0, 3), gpuId(2, 3), purpose, 'direct', opts);
      expect(p.switchCount).toBe(1);
      expect(p.spineCount).toBe(0);
    }
  });

  it('tor: every inter-node transfer crosses leaf-spine-leaf regardless of method', () => {
    const t = buildTopology('tor');
    for (const [a, b] of allPairs(t)) {
      if (t.gpuMap.get(a)!.node === t.gpuMap.get(b)!.node) continue;
      for (const m of METHODS) {
        const p = route(t, a, b, 'dispatch', m, opts);
        expect(p.switchCount).toBe(3);
        expect(p.spineCount).toBe(1);
      }
    }
  });

  it('rail: cross-rail direct goes through a spine; relays stay on one leaf with one NVLink relay', () => {
    const t = buildTopology('rail');
    const src = gpuId(0, 0);
    const dst = gpuId(2, 5);
    const direct = route(t, src, dst, 'a2f', 'direct', opts);
    expect(direct.switchCount).toBe(3);
    expect(direct.relayCount).toBe(0);

    const recv = route(t, src, dst, 'dispatch', 'relayRecv', opts);
    expect(recv.switchCount).toBe(1);
    expect(recv.relayCount).toBe(1);
    // RDMA on the sender's rail to the same-index GPU of the destination node, then NVLink.
    expect(recv.segments.map((s) => [s.kind, s.from, s.to])).toEqual([
      ['rdma', src, gpuId(2, 0)],
      ['nvlink', gpuId(2, 0), dst],
    ]);

    const agg = route(t, dst, src, 'combine', 'relaySendAggregate', opts);
    expect(agg.switchCount).toBe(1);
    expect(agg.relayCount).toBe(1);
    // NVLink to the GPU matching the destination's rail, then RDMA on that rail.
    expect(agg.segments.map((s) => [s.kind, s.from, s.to])).toEqual([
      ['nvlink', dst, gpuId(2, 0)],
      ['rdma', gpuId(2, 0), src],
    ]);
  });

  it('rail: relay methods never touch a spine for any inter-node pair', () => {
    const t = buildTopology('rail');
    for (const [a, b] of allPairs(t)) {
      for (const m of ['relayRecv', 'relaySendAggregate'] as const) {
        expect(route(t, a, b, 'dispatch', m, opts).spineCount).toBe(0);
      }
    }
  });
});

describe('ECMP (ADR §13)', () => {
  it('is deterministic for the same seed', () => {
    for (const wiring of WIRINGS) {
      const t = buildTopology(wiring);
      for (const [a, b] of allPairs(t)) {
        const p1 = route(t, a, b, 'kv', 'direct', { seed: 7 });
        const p2 = route(t, a, b, 'kv', 'direct', { seed: 7 });
        expect(p1.segments.flatMap((s) => s.hops.map((h) => h.link.id))).toEqual(
          p2.segments.flatMap((s) => s.hops.map((h) => h.link.id)),
        );
      }
    }
  });

  it('a different seed re-rolls some spine choices', () => {
    const t = buildTopology('tor');
    let changed = 0;
    for (const [a, b] of allPairs(t)) {
      const ids = (seed: number) =>
        route(t, a, b, 'dispatch', 'direct', { seed }).segments.flatMap((s) => s.hops.map((h) => h.link.id)).join();
      if (ids(1) !== ids(2)) changed++;
    }
    expect(changed).toBeGreaterThan(0);
  });

  it('tor: both parallel uplinks and both parallel downlinks get used', () => {
    const t = buildTopology('tor');
    const ups = new Set<number>();
    const downs = new Set<number>();
    for (const [a, b] of allPairs(t)) {
      const p = route(t, a, b, 'dispatch', 'direct', opts);
      for (const seg of p.segments) {
        for (const h of seg.hops) {
          if (h.link.kind !== 'fabric') continue;
          (h.to.startsWith('spine') ? ups : downs).add(h.link.parallel);
        }
      }
    }
    expect([...ups].sort()).toEqual([0, 1]);
    expect([...downs].sort()).toEqual([0, 1]);
  });

  it('hashes on the NIC that actually sends after a relay', () => {
    const t = buildTopology('tor');
    // relaySendAggregate from g2.5 to g0.0 sends RDMA from g2.0's NIC; it must match
    // a direct RDMA from g2.0 to g0.0 for the same purpose.
    const relayed = route(t, gpuId(2, 5), gpuId(0, 0), 'combine', 'relaySendAggregate', opts);
    const direct = route(t, gpuId(2, 0), gpuId(0, 0), 'combine', 'direct', opts);
    const rdmaIds = (p: Path) =>
      p.segments.filter((s) => s.kind === 'rdma').flatMap((s) => s.hops.map((h) => h.link.id));
    expect(rdmaIds(relayed)).toEqual(rdmaIds(direct));
  });
});
