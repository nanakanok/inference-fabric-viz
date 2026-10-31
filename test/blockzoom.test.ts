import { describe, expect, it } from 'vitest';
import { blockDetail } from '../src/model/blockzoom';
import { cardsForStep } from '../src/model/gpuview';
import { buildSteps } from '../src/model/steps';
import { buildWorkload } from '../src/model/workload';
import { COMBOS, label, PROMPT } from './fixtures';

const workload = buildWorkload(PROMPT, 4);

function decodeCards(s: (typeof COMBOS)[number]) {
  const plan = buildSteps(s, workload);
  const i = plan.steps.findIndex((x) => x.kind === 'decode-send');
  return { plan, cards: cardsForStep(plan, s, workload, i, 0), generated: plan.steps[i]!.outputSoFar };
}

describe('block zoom (ADR §23)', () => {
  it('router lists top-k experts with weights summing to 1, sorted, with their owners', () => {
    for (const s of COMBOS) {
      const { plan, cards } = decodeCards(s);
      // The decode-side attention card (A / D-A in AF modes; the P card routes prefill tokens).
      const attnCard = cards.find((c) => c.role === 'A' || c.role === 'DA') ?? cards.find((c) => c.blocks.includes('router'))!;
      const d = blockDetail(attnCard, 'router', s, plan.roles, workload);
      if (!('experts' in d) || d.block !== 'router') throw new Error(label(s));
      expect(d.experts).toHaveLength(8);
      expect(d.experts.reduce((a, e) => a + e.weight, 0)).toBeCloseTo(1, 10);
      for (let k = 1; k < d.experts.length; k++) expect(d.experts[k - 1]!.weight).toBeGreaterThanOrEqual(d.experts[k]!.weight);
      // In AF modes every owner is on the F side, never the attention GPU itself.
      if (s.af) for (const e of d.experts) expect(e.where).not.toBe('here');
    }
  });

  it('experts zoom shows exactly the lit experts of the card', () => {
    for (const s of COMBOS) {
      const { plan, cards } = decodeCards(s);
      for (const c of cards.filter((x) => x.blocks.includes('experts'))) {
        const d = blockDetail(c, 'experts', s, plan.roles, workload);
        if (d.block !== 'experts' || !('hidden' in d)) continue;
        expect(new Set(d.experts.map((e) => e.expert)), label(s)).toEqual(c.lit);
      }
    }
  });

  it('combine partial sums cover all top-k weight, one row per owner GPU', () => {
    for (const s of COMBOS) {
      const { plan, cards } = decodeCards(s);
      const attnCard = cards.find((c) => c.blocks.includes('combine'))!;
      const d = blockDetail(attnCard, 'combine', s, plan.roles, workload);
      if (d.block !== 'combine' || !('partials' in d)) throw new Error(label(s));
      expect(d.partials.reduce((a, p) => a + p.weight, 0)).toBeCloseTo(1, 10);
      expect(new Set(d.partials.map((p) => p.owner)).size).toBe(d.partials.length);
      expect(d.sharedExpert).toBe(true);
    }
  });

  it('attention zoom reflects the model style and context length', () => {
    const s = COMBOS[0]!;
    const { plan, cards, generated } = decodeCards(s);
    const d = blockDetail(cards[0]!, 'attention', s, plan.roles, workload);
    if (d.block !== 'attention' || !('style' in d)) throw new Error();
    expect(d.style).toBe('mla');
    expect(d.kvBytesPerTokenPerLayer).toBe(576 * 2);
    expect(d.contextTokens).toBe(workload.requests[0]!.promptTokens + generated - 1);
    const up = { ...s, fabric: 'scaleup' as const };
    const p2 = buildSteps(up, workload);
    const c2 = cardsForStep(p2, up, workload, 0, 0)[0]!;
    const d2 = blockDetail(c2, 'kv', up, p2.roles, workload);
    if (!('style' in d2)) throw new Error();
    expect(d2.style).toBe('gqa');
    expect(d2.kvBytesPerTokenPerLayer).toBe(2 * 4 * 128 * 2);
  });

  it('decode-side cards have nothing to route during Prefill under PD', () => {
    for (const s of COMBOS.filter((c) => c.pd)) {
      const plan = buildSteps(s, workload);
      plan.steps.forEach((step, i) => {
        if (step.phase !== 'prefill') return;
        for (const c of cardsForStep(plan, s, workload, i, 0)) {
          if (c.role === 'P') continue;
          expect(c.routeRef, `${label(s)} ${step.kind} ${c.role}`).toBeNull();
          expect(c.lit.size).toBe(0);
          expect('idle' in blockDetail(c, 'router', s, plan.roles, workload)).toBe(true);
        }
      });
    }
  });

  it('experts zoom equals the card highlights in every step and overlap slot, Prefill included', async () => {
    const { buildPipeline } = await import('../src/model/pipeline');
    const { cardsForPipeline } = await import('../src/model/gpuview');
    for (const s of COMBOS) {
      const plan = buildSteps(s, workload);
      const all = [
        ...plan.steps.flatMap((_, i) => cardsForStep(plan, s, workload, i, 0)),
        ...buildPipeline(s, PROMPT, 4).slots.flatMap((slot) => cardsForPipeline(plan, s, workload, slot)),
      ];
      for (const c of all.filter((x) => x.experts.length && x.routeRef)) {
        const d = blockDetail(c, 'experts', s, plan.roles, workload);
        if (d.block !== 'experts' || !('hidden' in d)) throw new Error();
        expect(new Set(d.experts.map((e) => e.expert)), `${label(s)} ${c.role}`).toEqual(c.lit);
      }
    }
  });

  it('attention context follows the zoomed token: prompt in Prefill, prompt + i in decode iteration i', async () => {
    const { buildPipeline } = await import('../src/model/pipeline');
    const { cardsForPipeline } = await import('../src/model/gpuview');
    const s = COMBOS.find((c) => c.pd && !c.af)!;
    const plan = buildSteps(s, workload);
    const prompt = workload.requests[0]!.promptTokens;
    for (const slot of buildPipeline(s, PROMPT, 4).slots) {
      for (const c of cardsForPipeline(plan, s, workload, slot)) {
        const d = blockDetail(c, 'attention', s, plan.roles, workload);
        if (!('contextTokens' in d)) throw new Error();
        const expected = c.routeRef?.phase === 'decode' ? prompt + c.routeRef.tokenIdx : prompt;
        expect(d.contextTokens, `slot ${slot.index} ${c.role}`).toBe(expected);
      }
    }
    const iters = new Set(buildPipeline(s, PROMPT, 4).slots.flatMap((sl) => cardsForPipeline(plan, s, workload, sl)).filter((c) => c.role === 'D' && c.routeRef).map((c) => c.routeRef!.tokenIdx));
    expect(iters.size).toBeGreaterThan(1);
  });
});
