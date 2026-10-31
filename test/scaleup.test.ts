import { describe, expect, it } from 'vitest';
import { QWEN3_30B_A3B } from '../src/model/constants';
import { buildPipeline } from '../src/model/pipeline';
import { buildRoles, expertOwner, expertsOn, modelFor, type Scenario } from '../src/model/scenario';
import { buildSteps } from '../src/model/steps';
import { kvTraffic, totals } from '../src/model/traffic';
import { assignDecode, buildWorkload, tokenRoute } from '../src/model/workload';
import { COMBOS, PROMPT } from './fixtures';

/** Scale-up only ignores PD / AF / wiring / relay, so every combination must behave the same. */
const UP: Scenario[] = COMBOS.map((c) => ({ ...c, fabric: 'scaleup' }));
const nodeOf = (g: string) => g.slice(1).split('.')[0];

describe('scale-up only (ADR §21)', () => {
  it('uses the lightweight reference model', () => {
    expect(modelFor(UP[0]!)).toBe(QWEN3_30B_A3B);
    expect(QWEN3_30B_A3B.kvBytesPerToken).toBe(98304);
    expect(modelFor(COMBOS[0]!).name).toBe('DeepSeek-V3');
  });

  it('runs every node as a replica: unified roles, no KV transfer, regardless of PD/AF toggles', () => {
    for (const s of UP) {
      const roles = buildRoles(s);
      expect(roles.kvTransfer).toBe(false);
      expect(roles.prefill.disaggregated).toBe(false);
      expect(roles.prefill.attn).toHaveLength(32);
      expect(kvTraffic(s, roles, buildWorkload(PROMPT, 2), assignDecode(roles)).flows).toHaveLength(0);
    }
  });

  it('places a full expert set on every node, evenly over its 8 GPUs', () => {
    const g = buildRoles(UP[0]!).prefill;
    for (const attn of ['g0.3', 'g2.5']) {
      const counts = new Map<string, number>();
      for (let e = 0; e < QWEN3_30B_A3B.experts; e++) {
        const o = expertOwner(g, e, QWEN3_30B_A3B, attn);
        expect(nodeOf(o)).toBe(nodeOf(attn));
        counts.set(o, (counts.get(o) ?? 0) + 1);
      }
      expect(counts.size).toBe(8);
      expect(new Set(counts.values())).toEqual(new Set([16]));
    }
    expect(expertsOn(g, 'g1.2', QWEN3_30B_A3B)).toEqual(Array.from({ length: 16 }, (_, i) => 32 + i));
  });

  it('keeps every flow on NVLink: no RDMA, no leaf or spine, in steps and in the overlap view', () => {
    for (const s of UP) {
      const plan = buildSteps(s, buildWorkload(PROMPT, 4));
      const slots = buildPipeline(s, PROMPT, 4).slots.map((x) => x.stage);
      for (const stage of [...plan.steps.flatMap((st) => st.stages), ...slots]) {
        if (!stage.traffic) continue;
        expect(totals(stage.traffic.transfers).rdmaBytes).toBe(0);
        for (const f of stage.traffic.flows) {
          expect(nodeOf(f.src)).toBe(nodeOf(f.dst));
          expect(f.path.switchCount).toBe(0);
        }
      }
    }
  });

  it('charges activation bytes from the model hidden size', () => {
    const plan = buildSteps(UP[0]!, buildWorkload(PROMPT, 2));
    const send = plan.steps.find((x) => x.kind === 'decode-send')!.stages[0]!.traffic!;
    const ret = plan.steps.find((x) => x.kind === 'decode-return')!.stages[0]!.traffic!;
    for (const f of send.flows) expect(f.logicalBytes).toBe(f.tokens * 2048);
    for (const f of ret.flows) expect(f.logicalBytes).toBe(f.tokens * 4096);
  });

  it('routes tokens over the model expert count and skips the dense-layer step', () => {
    for (let t = 0; t < 30; t++) {
      for (const e of tokenRoute(0, 'decode', t, undefined, 128, 8).experts) expect(e).toBeLessThan(128);
    }
    const kinds = buildSteps(UP[0]!, buildWorkload(PROMPT, 2)).steps.map((s) => s.kind);
    expect(kinds).not.toContain('prefill-dense');
    expect(kinds).not.toContain('kv');
    const text = buildSteps(UP[0]!, buildWorkload(PROMPT, 2)).steps.find((s) => s.kind === 'prefill-moe')!.body;
    expect(text.en).toMatch(/leaf-spine is unused/);
    expect(text.en).toMatch(/48 layers/);
  });
});

describe('scale-up only keeps no stale scale-out state', () => {
  it('the overlap view is sequential with no KV lane, even with PD left on', () => {
    for (const s of UP) {
      const p = buildPipeline(s, PROMPT, 4);
      expect(p.cyclic).toBe(false);
      for (const slot of p.slots) {
        expect(slot.lanes.some((l) => l.lane === 'kv')).toBe(false);
        for (const l of slot.lanes) expect(`${l.text.ja} ${l.text.en}`).not.toMatch(/61/);
      }
    }
  });

  it('notes mention only the active model', async () => {
    const { notesFor } = await import('../src/content/ui');
    const up = notesFor(true).map((n) => `${n.ja} ${n.en}`).join(' ');
    expect(up).toMatch(/Qwen3-30B-A3B/);
    expect(up).not.toMatch(/DeepSeek|61 層|61 layers|576/);
    const out = notesFor(false).map((n) => n.en).join(' ');
    expect(out).toMatch(/DeepSeek-V3/);
  });
});
