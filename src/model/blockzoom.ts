// Block zoom (ADR §23): what happens inside one compute block of a
// representative GPU, for the tracked request's current token. Schematic and
// derived from the deterministic workload and the public model config, not
// from running a model.

import { DENSE_LAYERS, type ModelSpec } from './constants';
import type { BlockId, GpuCard } from './gpuview';
import { expertOwner, modelFor, type Roles, type Scenario } from './scenario';
import { tokenRoute, TRACKED_REQUEST, type Workload } from './workload';

export interface RoutedExpert {
  expert: number;
  /** Router weight; for a set of prefill tokens, the mean over the tokens that picked it. */
  weight: number;
  /** Tokens (of those considered) that picked this expert. */
  tokens: number;
  owner: string;
  /** Owner relative to the card GPU. */
  where: 'here' | 'same-node' | 'remote';
}

export type BlockDetail =
  | { block: 'router'; experts: RoutedExpert[]; totalExperts: number; topK: number; scope: Scope }
  | { block: 'experts'; experts: RoutedExpert[]; hidden: number; inter: number; hostedCount: number; scope: Scope }
  | { block: 'combine'; partials: { owner: string; weight: number; experts: number[] }[]; sharedExpert: boolean; scope: Scope }
  | { block: 'attention' | 'kv'; style: ModelSpec['attention']; contextTokens: number; kvBytesPerTokenPerLayer: number; layers: number; kvOnGpu: number }
  | { block: Exclude<BlockId, 'router' | 'experts' | 'combine' | 'attention' | 'kv'>; denseLayers: number; hidden: number }
  | { block: BlockId; idle: true };

/**
 * Which tokens a zoom describes. Router and combine follow one token (the
 * first prompt token in Prefill, the current token in Decode). The experts
 * zoom in Prefill covers the same first-64-token set the card highlights.
 */
export interface Scope {
  phase: 'prefill' | 'decode';
  /** Token index for single-token views (prompt position or decode iteration). */
  tokenIdx: number;
  /** Number of prompt tokens aggregated (experts zoom in Prefill), else 1. */
  tokens: number;
}

/** Prompt tokens the card highlights aggregate over in Prefill (ADR §22). */
export const PREFILL_HIGHLIGHT_TOKENS = 64;

function nodeOf(gpu: string): string {
  return gpu.slice(1).split('.')[0]!;
}

export function blockDetail(card: GpuCard, block: BlockId, scenario: Scenario, roles: Roles, workload: Workload): BlockDetail {
  const m = modelFor(scenario);
  const ref = card.routeRef;
  if (!ref && (block === 'router' || block === 'experts' || block === 'combine')) return { block, idle: true };

  const prompt = workload.requests[TRACKED_REQUEST]!.promptTokens;
  /** Experts picked by `count` consecutive tokens starting at the scope token, with mean weights. */
  const routed = (count = 1): RoutedExpert[] => {
    const acc = new Map<number, { sum: number; tokens: number }>();
    for (let k = 0; k < count; k++) {
      const r = tokenRoute(TRACKED_REQUEST, ref!.phase, ref!.tokenIdx + k, ref!.seed, m.experts, m.topK);
      r.experts.forEach((e, i) => {
        const a = acc.get(e) ?? { sum: 0, tokens: 0 };
        a.sum += r.weights[i]!;
        a.tokens++;
        acc.set(e, a);
      });
    }
    return [...acc]
      .map(([expert, a]) => {
        const owner = expertOwner(roles[ref!.phase], expert, m, ref!.attnGpu);
        const where: RoutedExpert['where'] = owner === card.gpu ? 'here' : nodeOf(owner) === nodeOf(card.gpu) ? 'same-node' : 'remote';
        return { expert, weight: a.sum / a.tokens, tokens: a.tokens, owner, where };
      })
      .sort((a, b) => b.tokens - a.tokens || b.weight - a.weight);
  };
  const single: Scope | null = ref ? { phase: ref.phase, tokenIdx: ref.tokenIdx, tokens: 1 } : null;
  const prefillSet = ref?.phase === 'prefill' ? Math.min(prompt, PREFILL_HIGHLIGHT_TOKENS) : 1;

  switch (block) {
    case 'router':
      return { block, experts: routed(), totalExperts: m.experts, topK: m.topK, scope: single! };
    case 'experts':
      return {
        block,
        experts: routed(prefillSet).filter((e) => e.owner === card.gpu),
        hidden: m.hidden,
        inter: m.expertInter,
        hostedCount: card.experts.length,
        scope: { ...single!, tokens: prefillSet },
      };
    case 'combine': {
      const byOwner = new Map<string, { owner: string; weight: number; experts: number[] }>();
      for (const e of routed()) {
        const p = byOwner.get(e.owner) ?? { owner: e.owner, weight: 0, experts: [] };
        p.weight += e.weight;
        p.experts.push(e.expert);
        byOwner.set(e.owner, p);
      }
      return { block, partials: [...byOwner.values()].sort((a, b) => b.weight - a.weight), sharedExpert: m.sharedExpert, scope: single! };
    }
    case 'attention':
    case 'kv': {
      // Prefill attends over the prompt; decode iteration i attends over prompt + i tokens.
      const context = ref?.phase === 'decode' ? prompt + ref.tokenIdx : prompt;
      return {
        block,
        style: m.attention,
        contextTokens: context,
        kvBytesPerTokenPerLayer: m.kvBytesPerToken / m.layers,
        layers: m.layers,
        kvOnGpu: card.hbm.kv,
      };
    }
    default:
      return { block, denseLayers: m.id === 'deepseek-v3' ? DENSE_LAYERS : m.denseLayers, hidden: m.hidden };
  }
}
