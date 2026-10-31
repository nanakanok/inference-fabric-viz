import type { Scenario } from '../src/model/scenario';

/** All 16 combinations of PD x AF x wiring x decode relay (ADR §15). */
export const COMBOS: Scenario[] = [false, true].flatMap((pd) =>
  [false, true].flatMap((af) =>
    (['rail', 'tor'] as const).flatMap((wiring) =>
      [false, true].map((decodeRelay) => ({ pd, af, wiring, decodeRelay, ecmpSeed: 1 })),
    ),
  ),
);

export function label(s: Scenario): string {
  return `pd=${+s.pd} af=${+s.af} ${s.wiring} decodeRelay=${+s.decodeRelay}`;
}

export const PROMPT = 'Explain how prefill and decode are split across GPUs.';
