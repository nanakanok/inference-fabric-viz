// Step sequence for the step-through mode (ADR §7). One batch is followed;
// the tracked request is highlighted while the rest of the batch is drawn faintly.

import { DUMMY_OUTPUT, stepBody, stepTitle, type I18n, type StepKind, type StepTextContext } from '../content/text';
import { buildRoles, isScaleUpOnly, modelFor, type Phase, type Roles, type Scenario } from './scenario';
import { kvTraffic, moeLayerTraffic, range, type Traffic } from './traffic';
import { assignDecode, prefillGpuOf, tokenize, TRACKED_REQUEST, type Workload } from './workload';

/** Decode iterations drawn before "and so on" (ADR §7). */
export const SHOWN_DECODE_ITERATIONS = 3;

export type Activity = 'attention' | 'dense' | 'router' | 'experts' | 'aggregate' | 'head' | 'scheduler';

/**
 * One ordered sub-phase of a step: what each busy GPU computes, then the
 * network traffic it emits (if any). GPUs that host both attention and
 * experts appear in different stages, so no activity is lost.
 */
export interface Stage {
  activity: Map<string, Activity>;
  traffic: Traffic | null;
}

export interface Step {
  index: number;
  kind: StepKind;
  phase: Phase;
  title: I18n;
  body: I18n;
  /** Ordered compute/network stages, animated in order (e.g. router+dispatch, experts+combine, aggregate). */
  stages: Stage[];
  /** Output tokens of the tracked request available after this step. */
  outputSoFar: number;
  decodeIteration?: number;
}

export interface StepPlan {
  roles: Roles;
  decodeOf: Map<number, string>;
  steps: Step[];
}

function activityOn(gpus: Iterable<string>, a: Activity): Map<string, Activity> {
  return new Map([...gpus].map((g) => [g, a]));
}

function compute(gpus: Iterable<string>, a: Activity): Stage {
  return { activity: activityOn(gpus, a), traffic: null };
}

function emit(gpus: Iterable<string>, a: Activity, traffic: Traffic): Stage {
  return { activity: activityOn(gpus, a), traffic };
}

export function outputTokens(lang: 'ja' | 'en', count: number): string[] {
  const words = tokenize(DUMMY_OUTPUT[lang]);
  return Array.from({ length: count }, (_, i) => words[i % words.length]!);
}

export function buildSteps(scenario: Scenario, workload: Workload): StepPlan {
  const roles = buildRoles(scenario);
  const model = modelFor(scenario);
  const decodeOf = assignDecode(roles);
  const ctx: StepTextContext = {
    pd: scenario.pd,
    af: scenario.af,
    prefillAf: roles.prefill.disaggregated,
    decodeAf: roles.decode.disaggregated,
    decodeRelay: scenario.decodeRelay,
    wiring: scenario.wiring,
    scaleUp: isScaleUpOnly(scenario),
    modelName: model.name,
    layers: model.layers,
    moeLayers: model.layers - model.denseLayers,
    sharedExpert: model.sharedExpert,
  };
  const steps: Step[] = [];
  const push = (s: Omit<Step, 'index' | 'title' | 'body'>, c: StepTextContext = ctx) =>
    steps.push({ ...s, index: steps.length, title: stepTitle(s.kind, c), body: stepBody(s.kind, c) });

  const prefillAttn = workload.requests.map((r) => prefillGpuOf(roles, r.id));
  const prefillLayer = (direction: 'send' | 'return') =>
    moeLayerTraffic({
      scenario,
      roles,
      workload,
      phase: 'prefill',
      direction,
      attnGpuOf: (r) => prefillGpuOf(roles, r),
      tokensOf: (r) => range(workload.requests[r]!.promptTokens),
    });
  const decodeLayer = (direction: 'send' | 'return', iteration: number) =>
    moeLayerTraffic({
      scenario,
      roles,
      workload,
      phase: 'decode',
      direction,
      attnGpuOf: (r) => decodeOf.get(r)!,
      tokensOf: () => [iteration],
    });

  push({ kind: 'arrive', phase: 'prefill', stages: [compute(prefillAttn, 'scheduler')], outputSoFar: 0 });
  if (model.denseLayers > 0) {
    push({ kind: 'prefill-dense', phase: 'prefill', stages: [compute(prefillAttn, 'dense')], outputSoFar: 0 });
  }
  push({
    kind: 'prefill-moe',
    phase: 'prefill',
    stages: [
      emit(prefillAttn, 'router', prefillLayer('send')),
      emit(roles.prefill.experts, 'experts', prefillLayer('return')),
      compute(prefillAttn, 'aggregate'),
    ],
    outputSoFar: 0,
  });
  push({ kind: 'prefill-done', phase: 'prefill', stages: [compute(prefillAttn, 'head')], outputSoFar: 1 });
  if (roles.kvTransfer) {
    push({
      kind: 'kv',
      phase: 'prefill',
      stages: [emit(prefillAttn, 'attention', kvTraffic(scenario, roles, workload, decodeOf))],
      outputSoFar: 1,
    });
  }

  const decodeAttn = [...new Set(decodeOf.values())];
  const remaining = workload.outputTokens - 1;
  const shown = Math.min(remaining, SHOWN_DECODE_ITERATIONS);
  for (let i = 0; i < shown; i++) {
    const c = { ...ctx, iteration: i };
    const common = { phase: 'decode' as const, decodeIteration: i };
    push({ ...common, kind: 'decode-attn', stages: [compute(decodeAttn, 'attention')], outputSoFar: 1 + i }, c);
    push({ ...common, kind: 'decode-send', stages: [emit(decodeAttn, 'router', decodeLayer('send', i))], outputSoFar: 1 + i }, c);
    push(
      {
        ...common,
        kind: 'decode-return',
        stages: [emit(roles.decode.experts, 'experts', decodeLayer('return', i)), compute(decodeAttn, 'aggregate')],
        outputSoFar: 1 + i,
      },
      c,
    );
    push({ ...common, kind: 'decode-done', stages: [compute(decodeAttn, 'head')], outputSoFar: 2 + i }, c);
  }
  if (remaining > shown) {
    push({ kind: 'repeat', phase: 'decode', stages: [], outputSoFar: workload.outputTokens });
  }
  return { roles, decodeOf, steps };
}

/** GPU of the tracked request in a given phase. */
export function trackedGpu(plan: StepPlan, phase: Phase): string {
  return phase === 'prefill' ? prefillGpuOf(plan.roles, TRACKED_REQUEST) : plan.decodeOf.get(TRACKED_REQUEST)!;
}
