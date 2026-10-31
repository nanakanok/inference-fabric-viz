import { describe, expect, it } from 'vitest';
import { buildSteps, outputTokens, SHOWN_DECODE_ITERATIONS } from '../src/model/steps';
import { buildWorkload } from '../src/model/workload';
import { COMBOS, label, PROMPT } from './fixtures';

const kinds = (s: { kind: string }[]) => s.map((x) => x.kind);

describe('step sequence (ADR §7)', () => {
  it('follows the PD+AF sequence with three decode iterations then "and so on"', () => {
    const plan = buildSteps(COMBOS.find((c) => c.pd && c.af)!, buildWorkload(PROMPT, 10));
    const decode = ['decode-attn', 'decode-send', 'decode-return', 'decode-done'];
    expect(kinds(plan.steps)).toEqual([
      'arrive',
      'prefill-dense',
      'prefill-moe',
      'prefill-done',
      'kv',
      ...decode,
      ...decode,
      ...decode,
      'repeat',
    ]);
    expect(plan.steps.map((s) => s.index)).toEqual(plan.steps.map((_, i) => i));
  });

  it('has a KV step exactly when PD is on', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, buildWorkload(PROMPT, 4));
      expect(kinds(plan.steps).includes('kv'), label(s)).toBe(s.pd);
    }
  });

  it('dense layers and on-GPU steps carry no network traffic', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, buildWorkload(PROMPT, 3));
      for (const step of plan.steps) {
        if (['arrive', 'prefill-dense', 'prefill-done', 'decode-attn', 'decode-done', 'repeat'].includes(step.kind)) {
          expect(step.stages.every((st) => st.traffic === null), `${label(s)} ${step.kind}`).toBe(true);
        }
      }
    }
  });

  it('prefill MoE has send then return stages using the mode purposes', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, buildWorkload(PROMPT, 2));
      const moe = plan.steps.find((x) => x.kind === 'prefill-moe')!;
      expect(moe.stages).toHaveLength(3);
      expect(moe.stages[2]!.traffic).toBeNull();
      const [send, ret] = moe.stages.slice(0, 2).map((st) => new Set(st.traffic!.flows.map((f) => f.purpose)));
      const prefillAf = s.af && !s.pd;
      expect([...send!]).toEqual([prefillAf ? 'a2f' : 'dispatch']);
      expect([...ret!]).toEqual([prefillAf ? 'f2a' : 'combine']);
    }
  });

  it('decode purposes follow AF on the decode side', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, buildWorkload(PROMPT, 2));
      const send = plan.steps.find((x) => x.kind === 'decode-send')!;
      const purposes = new Set(send.stages[0]!.traffic!.flows.map((f) => f.purpose));
      expect([...purposes], label(s)).toEqual([s.af ? 'a2f' : 'dispatch']);
    }
  });

  it('keeps router, expert and aggregate activity in order even when roles share GPUs', () => {
    for (const s of COMBOS) {
      const plan = buildSteps(s, buildWorkload(PROMPT, 2));
      for (const kind of ['prefill-moe', 'decode-return'] as const) {
        const step = plan.steps.find((x) => x.kind === kind)!;
        const seq = step.stages.map((st) => new Set(st.activity.values()));
        const experts = kind === 'prefill-moe' ? plan.roles.prefill.experts : plan.roles.decode.experts;
        const expertStage = step.stages.find((st) => [...st.activity.values()].includes('experts'))!;
        expect(expertStage, `${label(s)} ${kind}`).toBeDefined();
        expect([...expertStage.activity.keys()].sort()).toEqual([...experts].sort());
        expect([...seq.at(-1)!]).toEqual(['aggregate']);
      }
      const moe = plan.steps.find((x) => x.kind === 'prefill-moe')!;
      expect([...moe.stages[0]!.activity.values()].every((a) => a === 'router')).toBe(true);
    }
  });

  it('explanations depend on wiring: ToR never claims to avoid the spine', () => {
    const base = { pd: true, af: true, ecmpSeed: 1 };
    for (const decodeRelay of [false, true]) {
      const rail = buildSteps({ ...base, wiring: 'rail', decodeRelay }, buildWorkload(PROMPT, 2)).steps;
      const tor = buildSteps({ ...base, wiring: 'tor', decodeRelay }, buildWorkload(PROMPT, 2)).steps;
      for (const kind of ['kv', 'decode-send']) {
        const r = rail.find((x) => x.kind === kind)!.body;
        const t = tor.find((x) => x.kind === kind)!.body;
        expect(r.ja).not.toBe(t.ja);
        expect(r.en).not.toBe(t.en);
        expect(t.en).not.toMatch(/avoids the spine|only one leaf/);
        expect(t.ja).not.toMatch(/Spine を通りません|Leaf 1 台で済み/);
      }
    }
  });

  it('the first token comes from prefill and each decode iteration adds one', () => {
    const plan = buildSteps(COMBOS[0]!, buildWorkload(PROMPT, 3));
    const done = plan.steps.find((s) => s.kind === 'prefill-done')!;
    expect(done.outputSoFar).toBe(1);
    const last = plan.steps.at(-1)!;
    expect(last.kind).toBe('decode-done');
    expect(last.outputSoFar).toBe(3);
  });

  it('a single output token needs no decode steps', () => {
    const plan = buildSteps(COMBOS[0]!, buildWorkload(PROMPT, 1));
    expect(kinds(plan.steps).some((k) => k.startsWith('decode'))).toBe(false);
  });

  it(`draws at most ${SHOWN_DECODE_ITERATIONS} decode iterations`, () => {
    const plan = buildSteps(COMBOS[0]!, buildWorkload(PROMPT, 50));
    expect(plan.steps.filter((s) => s.kind === 'decode-send')).toHaveLength(SHOWN_DECODE_ITERATIONS);
    expect(plan.steps.at(-1)!.outputSoFar).toBe(50);
  });

  it('every step has Japanese and English text', () => {
    for (const s of COMBOS) {
      for (const step of buildSteps(s, buildWorkload(PROMPT, 5)).steps) {
        expect(step.title.ja.length).toBeGreaterThan(0);
        expect(step.title.en.length).toBeGreaterThan(0);
        expect(step.body.ja.length).toBeGreaterThan(0);
        expect(step.body.en.length).toBeGreaterThan(0);
      }
    }
  });

  it('dummy output yields the requested number of tokens', () => {
    expect(outputTokens('ja', 7)).toHaveLength(7);
    expect(outputTokens('en', 7)).toHaveLength(7);
  });
});
