import { describe, expect, it } from 'vitest';
import {
  KV_BYTES_PER_TOKEN,
  RETURN_BYTES_PER_TOKEN,
  ROUTED_EXPERTS,
  SEND_BYTES_PER_TOKEN,
  MAX_NODES_PER_TOKEN,
} from '../src/model/constants';
import { buildRoles, expertOwner, methodFor, roleLabel, type Phase, type Scenario } from '../src/model/scenario';
import { kvTraffic, moeLayerTraffic, range, totals, type Traffic } from '../src/model/traffic';
import { assignDecode, buildWorkload, prefillGpuOf, tokenize, tokenRoute, TRACKED_REQUEST } from '../src/model/workload';
import { COMBOS, label, PROMPT } from './fixtures';

const workload = buildWorkload(PROMPT, 3);
const nodeOf = (g: string) => Number(g.slice(1).split('.')[0]);

function layer(s: Scenario, phase: Phase, direction: 'send' | 'return'): Traffic {
  const roles = buildRoles(s);
  const decodeOf = assignDecode(roles);
  return moeLayerTraffic({
    scenario: s,
    roles,
    workload,
    phase,
    direction,
    attnGpuOf: (r) => (phase === 'prefill' ? prefillGpuOf(roles, r) : decodeOf.get(r)!),
    tokensOf: (r) => (phase === 'prefill' ? range(workload.requests[r]!.promptTokens) : [0]),
  });
}

/** (request, src, dst, tokens) of every flow, order-independent. */
function logical(t: Traffic): string[] {
  return t.flows.map((f) => `${f.request}|${f.src}>${f.dst}|${f.tokens}`).sort();
}

describe('roles and placement (ADR §5, §17)', () => {
  it('assigns the mode table from the ADR', () => {
    const unified = buildRoles({ pd: false, af: false });
    expect(unified.prefill.attn).toHaveLength(32);
    expect(unified.prefill.experts).toHaveLength(32);
    expect(unified.kvTransfer).toBe(false);

    const pd = buildRoles({ pd: true, af: false });
    expect(pd.prefill.attn.every((g) => nodeOf(g) <= 1)).toBe(true);
    expect(pd.decode.attn.every((g) => nodeOf(g) >= 2)).toBe(true);
    expect(pd.decode.experts).toEqual(pd.decode.attn);

    const af = buildRoles({ pd: false, af: true });
    expect(af.prefill).toBe(af.decode);
    expect(af.prefill.attn.every((g) => nodeOf(g) <= 1)).toBe(true);
    expect(af.prefill.experts.every((g) => nodeOf(g) >= 2)).toBe(true);

    const both = buildRoles({ pd: true, af: true });
    expect(both.decode.attn.every((g) => nodeOf(g) === 2)).toBe(true);
    expect(both.decode.experts.every((g) => nodeOf(g) === 3)).toBe(true);
    expect(roleLabel(both, 'g0.0')).toBe('prefill');
    expect(roleLabel(both, 'g2.0')).toBe('decode-attention');
    expect(roleLabel(both, 'g3.0')).toBe('decode-expert');
  });

  it('places every routed expert on exactly one GPU of the group, evenly', () => {
    for (const s of COMBOS) {
      const roles = buildRoles(s);
      for (const phase of ['prefill', 'decode'] as const) {
        const g = roles[phase];
        const counts = new Map<string, number>();
        for (let e = 0; e < ROUTED_EXPERTS; e++) {
          const o = expertOwner(g, e);
          expect(g.experts).toContain(o);
          counts.set(o, (counts.get(o) ?? 0) + 1);
        }
        expect(new Set(counts.values()).size).toBe(1);
        expect(counts.size).toBe(g.experts.length);
      }
    }
  });

  it('uses Normal relay for prefill, direct for decode unless decodeRelay (ADR §12)', () => {
    expect(methodFor('prefill', 'send', { decodeRelay: false })).toBe('relayRecv');
    expect(methodFor('prefill', 'return', { decodeRelay: false })).toBe('relaySendAggregate');
    expect(methodFor('decode', 'send', { decodeRelay: false })).toBe('direct');
    expect(methodFor('decode', 'return', { decodeRelay: false })).toBe('direct');
    expect(methodFor('decode', 'send', { decodeRelay: true })).toBe('relayRecv');
    expect(methodFor('decode', 'return', { decodeRelay: true })).toBe('relaySendAggregate');
  });
});

describe('workload (ADR §7, §9, §20)', () => {
  it('is deterministic and independent of the scenario', () => {
    expect(buildWorkload(PROMPT, 3)).toEqual(workload);
    expect(tokenRoute(5, 'decode', 2)).toEqual(tokenRoute(5, 'decode', 2));
  });

  it('uses the workload seed for router choices in generated traffic', () => {
    const custom = buildWorkload(PROMPT, 1, 12345);
    const s = COMBOS.find((c) => c.pd && c.af)!;
    const roles = buildRoles(s);
    const decodeOf = assignDecode(roles);
    const t = moeLayerTraffic({
      scenario: s,
      roles,
      workload: custom,
      phase: 'decode',
      direction: 'send',
      attnGpuOf: (r) => decodeOf.get(r)!,
      tokensOf: () => [0],
    });
    const tracked = t.flows.filter((f) => f.tracked).map((f) => f.dst).sort();
    const expected = [...new Set(tokenRoute(TRACKED_REQUEST, 'decode', 0, 12345).experts.map((e) => expertOwner(roles.decode, e)))]
      .filter((g) => g !== decodeOf.get(TRACKED_REQUEST))
      .sort();
    expect(tracked).toEqual(expected);
    expect(tokenRoute(TRACKED_REQUEST, 'decode', 0, 12345)).not.toEqual(tokenRoute(TRACKED_REQUEST, 'decode', 0));
  });

  it('routes each token to 8 distinct experts with weights summing to 1', () => {
    for (let t = 0; t < 50; t++) {
      const r = tokenRoute(1, 'prefill', t);
      expect(new Set(r.experts).size).toBe(8);
      expect(r.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    }
  });

  it('never routes a token to more than 4 nodes', () => {
    for (const s of COMBOS) {
      const roles = buildRoles(s);
      for (let t = 0; t < 30; t++) {
        const nodes = new Set(tokenRoute(0, 'decode', t).experts.map((e) => nodeOf(expertOwner(roles.decode, e))));
        expect(nodes.size).toBeLessThanOrEqual(MAX_NODES_PER_TOKEN);
      }
    }
  });

  it('tokenizes CJK per character and ASCII words in chunks of 4', () => {
    expect(tokenize('推論とは')).toEqual(['推', '論', 'と', 'は']);
    expect(tokenize('inference fabric')).toEqual(['infe', 'renc', 'e', 'fabr', 'ic']);
  });

  it('reserves decode GPUs on the same local index, balanced, identical across wirings', () => {
    const both = buildRoles({ pd: true, af: true });
    const d = assignDecode(both);
    const load = new Map<string, number>();
    for (const [r, g] of d) {
      expect(g.split('.')[1]).toBe(prefillGpuOf(both, r).split('.')[1]);
      load.set(g, (load.get(g) ?? 0) + 1);
    }
    expect(new Set(load.values()).size).toBe(1);
  });
});

describe('MoE layer traffic (ADR §8, §15, §17)', () => {
  it('delivers each token to every selected expert GPU and returns to the same attention GPU', () => {
    for (const s of COMBOS) {
      const roles = buildRoles(s);
      for (const phase of ['prefill', 'decode'] as const) {
        const send = layer(s, phase, 'send');
        const ret = layer(s, phase, 'return');
        const decodeOf = assignDecode(roles);
        let expected = 0;
        for (const req of workload.requests) {
          const attn = phase === 'prefill' ? prefillGpuOf(roles, req.id) : decodeOf.get(req.id)!;
          const toks = phase === 'prefill' ? range(req.promptTokens) : [0];
          for (const t of toks) {
            const owners = new Set(tokenRoute(req.id, phase, t).experts.map((e) => expertOwner(roles[phase], e)));
            owners.delete(attn);
            expected += owners.size;
          }
        }
        const sent = send.flows.reduce((a, f) => a + f.tokens, 0);
        expect(sent, label(s)).toBe(expected);
        // Return flows mirror send flows.
        const mirrored = send.flows.map((f) => `${f.request}|${f.dst}>${f.src}|${f.tokens}`).sort();
        expect(logical(ret), label(s)).toEqual(mirrored);
      }
    }
  });

  it('AF modes send only from attention GPUs to disjoint expert GPUs', () => {
    for (const s of COMBOS.filter((c) => c.af)) {
      const roles = buildRoles(s);
      const send = layer(s, 'decode', 'send');
      for (const f of send.flows) {
        expect(f.purpose).toBe('a2f');
        expect(roles.decode.attn).toContain(f.src);
        expect(roles.decode.experts).toContain(f.dst);
      }
      if (s.pd) expect(send.flows.every((f) => nodeOf(f.src) === 2 && nodeOf(f.dst) === 3)).toBe(true);
    }
  });

  it('keeps the logical workload fixed across wiring and decode relay toggles', () => {
    for (const pd of [false, true]) {
      for (const af of [false, true]) {
        const variants = COMBOS.filter((c) => c.pd === pd && c.af === af);
        for (const phase of ['prefill', 'decode'] as const) {
          const ref = logical(layer(variants[0]!, phase, 'send'));
          for (const v of variants) expect(logical(layer(v, phase, 'send')), label(v)).toEqual(ref);
        }
      }
    }
  });

  it('charges 1 byte/elem on send and 2 bytes/elem on return', () => {
    const s = COMBOS.find((c) => c.pd && c.af && c.wiring === 'rail' && !c.decodeRelay)!;
    const send = layer(s, 'decode', 'send');
    const ret = layer(s, 'decode', 'return');
    for (const f of send.flows) expect(f.logicalBytes).toBe(f.tokens * SEND_BYTES_PER_TOKEN);
    for (const f of ret.flows) expect(f.logicalBytes).toBe(f.tokens * RETURN_BYTES_PER_TOKEN);
    expect(SEND_BYTES_PER_TOKEN).toBe(7168);
    expect(RETURN_BYTES_PER_TOKEN).toBe(14336);
  });

  it('relay dedups RDMA per (token, node); direct sends one RDMA per (token, GPU)', () => {
    const base = { pd: true, af: true, wiring: 'rail' as const, ecmpSeed: 1 };
    const direct = layer({ ...base, decodeRelay: false }, 'decode', 'send');
    const relay = layer({ ...base, decodeRelay: true }, 'decode', 'send');
    const roles = buildRoles(base);
    const decodeOf = assignDecode(roles);
    let gpuLegs = 0;
    let nodeLegs = 0;
    for (const req of workload.requests) {
      const owners = new Set(tokenRoute(req.id, 'decode', 0).experts.map((e) => expertOwner(roles.decode, e)));
      const attn = decodeOf.get(req.id)!;
      const remote = [...owners].filter((o) => nodeOf(o) !== nodeOf(attn));
      gpuLegs += remote.length;
      nodeLegs += new Set(remote.map(nodeOf)).size;
    }
    expect(totals(direct.transfers).rdmaBytes).toBe(gpuLegs * SEND_BYTES_PER_TOKEN);
    expect(totals(relay.transfers).rdmaBytes).toBe(nodeLegs * SEND_BYTES_PER_TOKEN);
    expect(totals(relay.transfers).nvlinkBytes).toBeGreaterThan(0);
  });

  it('rail: decode A->F crosses a spine by default and never with decode relay', () => {
    const base = { pd: true, af: true, wiring: 'rail' as const, ecmpSeed: 1 };
    const direct = layer({ ...base, decodeRelay: false }, 'decode', 'send');
    const relay = layer({ ...base, decodeRelay: true }, 'decode', 'send');
    const crossRail = direct.flows.filter((f) => f.src.split('.')[1] !== f.dst.split('.')[1]);
    expect(crossRail.length).toBeGreaterThan(0);
    for (const f of crossRail) expect(f.path.switchCount).toBe(3);
    for (const f of relay.flows) {
      expect(f.path.spineCount).toBe(0);
      expect(f.path.switchCount).toBe(1);
    }
    expect(totals(relay.transfers).spineBytes).toBe(0);
    expect(totals(direct.transfers).spineBytes).toBeGreaterThan(0);
  });

  it('marks only the tracked request as tracked', () => {
    const t = layer(COMBOS[0]!, 'decode', 'send');
    for (const f of t.flows) expect(f.tracked).toBe(f.request === TRACKED_REQUEST);
  });
});

describe('KV handoff (ADR §5, §8)', () => {
  it('moves every request\'s KV to its reserved decode GPU with the MLA formula', () => {
    for (const s of COMBOS) {
      const roles = buildRoles(s);
      const decodeOf = assignDecode(roles);
      const kv = kvTraffic(s, roles, workload, decodeOf);
      if (!roles.kvTransfer) {
        expect(kv.flows).toHaveLength(0);
        continue;
      }
      expect(kv.flows.filter((f) => f.tracked)).toHaveLength(1);
      const total = kv.flows.reduce((a, f) => a + f.logicalBytes, 0);
      const tokens = workload.requests.reduce((a, r) => a + r.promptTokens, 0);
      expect(total).toBe(tokens * KV_BYTES_PER_TOKEN);
      for (const f of kv.flows) {
        expect(f.method).toBe('direct');
        // Same local index is reserved, so rail wiring keeps KV on one leaf.
        expect(f.path.switchCount).toBe(s.wiring === 'rail' ? 1 : 3);
      }
    }
    expect(KV_BYTES_PER_TOKEN).toBe(70272);
  });

  it('scales linearly with prompt length', () => {
    const s = COMBOS.find((c) => c.pd)!;
    const roles = buildRoles(s);
    const d = assignDecode(roles);
    const kvOf = (p: string) =>
      kvTraffic(s, roles, buildWorkload(p, 1), d).flows.find((f) => f.tracked)!.logicalBytes;
    expect(kvOf('推論'.repeat(100))).toBe(2 * kvOf('推論'.repeat(50)));
  });
});
