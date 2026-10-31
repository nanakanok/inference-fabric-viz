import { describe, expect, it } from 'vitest';
import { DEEPSEEK_V3, QWEN3_30B_A3B } from '../src/model/constants';
import { cardsForPipeline, cardsForStep, hbmScaleFor, weightSplit } from '../src/model/gpuview';
import { buildPipeline } from '../src/model/pipeline';
import { buildSteps } from '../src/model/steps';
import { buildWorkload, prefillGpuOf, TRACKED_REQUEST } from '../src/model/workload';
import { COMBOS, label, PROMPT } from './fixtures';

const workload = buildWorkload(PROMPT, 4);
const roleSet = (pd: boolean, af: boolean) =>
  pd && af ? ['P', 'DA', 'DF'] : pd ? ['P', 'D'] : af ? ['A', 'F'] : ['U'];

describe('inside-the-GPU cards (ADR §22)', () => {
  it('shows one representative per role: U, P/D, A/F or P/D-A/D-F', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, workload);
      const cards = cardsForStep(plan, s, workload, 0, 0);
      expect(cards.map((c) => c.role), label(s)).toEqual(roleSet(s.pd, s.af));
      expect(cards[0]!.gpu).toBe(prefillGpuOf(plan.roles, TRACKED_REQUEST));
    }
    const up = { ...COMBOS[5]!, fabric: 'scaleup' as const };
    expect(cardsForStep(buildSteps(up, workload), up, workload, 0, 0).map((c) => c.role)).toEqual(['U']);
  });

  it('lights only blocks the card draws, following the stage activity', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, workload);
      plan.steps.forEach((step, i) =>
        step.stages.forEach((_, j) => {
          for (const c of cardsForStep(plan, s, workload, i, j)) {
            for (const b of c.active) expect(c.blocks, `${label(s)} ${step.kind}`).toContain(b);
          }
        }),
      );
    }
  });

  it('F cards hold experts and no KV; A cards hold no experts', () => {
    for (const s of COMBOS.filter((c) => c.af)) {
      const plan = buildSteps(s, workload);
      const last = plan.steps.length - 1;
      for (const c of cardsForStep(plan, s, workload, last, 0)) {
        if (c.role === 'F' || c.role === 'DF') {
          expect(c.experts.length).toBeGreaterThan(0);
          expect(c.hbm.kv).toBe(0);
          expect(c.blocks).toEqual(['recv', 'experts', 'send']);
        }
        if (c.role === 'A' || c.role === 'DA') expect(c.experts).toHaveLength(0);
      }
    }
  });

  it('lit experts are a subset of the experts hosted on the card GPU', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, workload);
      plan.steps.forEach((_, i) => {
        for (const c of cardsForStep(plan, s, workload, i, 0)) for (const e of c.lit) expect(c.experts).toContain(e);
      });
      for (const slot of buildPipeline(s, PROMPT, 4).slots) {
        for (const c of cardsForPipeline(plan, s, workload, slot)) for (const e of c.lit) expect(c.experts).toContain(e);
      }
    }
  });

  it('KV leaves the P GPU and appears on the D GPU after the KV step', () => {
    const s = COMBOS.find((c) => c.pd && !c.af)!;
    const plan = buildSteps(s, workload);
    const at = (kind: string) => cardsForStep(plan, s, workload, plan.steps.findIndex((x) => x.kind === kind), 0);
    const done = at('prefill-done');
    expect(done.find((c) => c.role === 'P')!.hbm.kv).toBeGreaterThan(0);
    expect(done.find((c) => c.role === 'D')!.hbm.kv).toBe(0);
    const kv = at('kv');
    expect(kv.find((c) => c.role === 'P')!.hbm.kv).toBe(0);
    expect(kv.find((c) => c.role === 'D')!.hbm.kv).toBe(done.find((c) => c.role === 'P')!.hbm.kv);
    const dec = at('decode-done');
    expect(dec.find((c) => c.role === 'D')!.hbm.kv).toBeGreaterThanOrEqual(kv.find((c) => c.role === 'D')!.hbm.kv);
  });

  it('weight estimates follow the public parameter counts', () => {
    const v3 = weightSplit(DEEPSEEK_V3);
    expect(v3.expertBytes + v3.otherBytes).toBeCloseTo(671e9, -6);
    const q = weightSplit(QWEN3_30B_A3B);
    expect(q.expertBytes + q.otherBytes).toBeCloseTo(61e9, -6);
  });

  it('overlap view: non-P cards light experts exactly where the tracked token travels in that slot', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, workload);
      const attn = plan.decodeOf.get(TRACKED_REQUEST)!;
      for (const slot of buildPipeline(s, PROMPT, 4).slots) {
        // Without PD, prefill slots run on the same cards (the prompt here is under the 64-token highlight cap).
        const flows = (slot.stage.traffic?.flows ?? []).filter((f) => f.tracked && f.purpose !== 'kv' && (f.phase === 'decode' || !plan.roles.kvTransfer));
        // GPUs holding experts the tracked token uses now: destinations of sends, sources of returns.
        const touched = new Set(flows.map((f) => (f.purpose === 'dispatch' || f.purpose === 'a2f' ? f.dst : f.src)));
        for (const c of cardsForPipeline(plan, s, workload, slot)) {
          if (c.role === 'P' || !c.experts.length || c.gpu === attn) continue;
          expect(c.lit.size > 0, `${label(s)} slot ${slot.index} ${c.role}`).toBe(touched.has(c.gpu));
        }
      }
    }
  });

  it('every rendered card fits the HBM scale, including the stage-less repeat step', () => {
    const long = buildWorkload('短い', 4096);
    for (const s of COMBOS.filter((c) => c.pd)) {
      const plan = buildSteps(s, long);
      const slots = buildPipeline(s, '短い', 4096).slots;
      const scale = hbmScaleFor(plan, s, long, slots);
      expect(plan.steps.at(-1)!.kind).toBe('repeat');
      plan.steps.forEach((step, i) => {
        for (let j = 0; j < Math.max(1, step.stages.length); j++) {
          for (const c of cardsForStep(plan, s, long, i, j)) expect(c.hbm.weights + c.hbm.kv + c.hbm.activations).toBeLessThanOrEqual(scale);
        }
      });
    }
  });
});
