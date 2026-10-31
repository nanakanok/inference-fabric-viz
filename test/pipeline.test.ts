import { describe, expect, it } from 'vitest';
import { LAYERS } from '../src/model/constants';
import { buildPipeline, decodeSchedule, type DecodeOp, type Slot } from '../src/model/pipeline';
import { COMBOS, label, PROMPT } from './fixtures';

const purposes = (slot: Slot) => new Set(slot.stage.traffic?.flows.map((f) => f.purpose) ?? []);
const phases = (slot: Slot) => new Set(slot.stage.traffic?.flows.filter((f) => f.purpose !== 'kv').map((f) => f.phase) ?? []);
const opKey = (o: DecodeOp) => `${o.microBatch}|${o.iteration}`;

/** Every return must follow the matching send; cyclic schedules may wrap. */
function expectSendBeforeReturn(schedule: DecodeOp[][], cyclic: boolean) {
  const pending = new Set<string>();
  const rounds = cyclic ? 2 : 1;
  const sentEver = new Set(schedule.flat().filter((o) => o.direction === 'send').map(opKey));
  const returned: string[] = [];
  for (let r = 0; r < rounds; r++) {
    for (const ops of schedule) {
      for (const o of ops.filter((x) => x.direction === 'return')) {
        if (r === 0 && cyclic && !pending.has(opKey(o))) {
          // First pass of a cyclic schedule: a wrap-around return must match a send later in the cycle.
          expect(sentEver.has(opKey(o))).toBe(true);
        } else {
          expect(pending.has(opKey(o)), opKey(o)).toBe(true);
          pending.delete(opKey(o));
        }
        if (r === rounds - 1) returned.push(opKey(o));
      }
      for (const o of ops.filter((x) => x.direction === 'send')) pending.add(opKey(o));
    }
  }
  // Each send is returned exactly once per cycle (and the linear schedule drains fully).
  expect(returned.sort()).toEqual([...sentEver].sort());
  if (!cyclic) expect(pending.size).toBe(0);
}

describe('decode schedule', () => {
  for (const pingPong of [false, true]) {
    for (const cyclic of [false, true]) {
      it(`pingPong=${pingPong} cyclic=${cyclic}: returns follow their sends, across the wrap`, () => {
        expectSendBeforeReturn(decodeSchedule(cyclic || !pingPong ? 4 : 5, pingPong, cyclic), cyclic);
      });
    }
  }

  it('ping-pong overlaps A->F of one micro-batch with F->A of the other in steady state', () => {
    for (const ops of decodeSchedule(4, true, true)) {
      expect(ops.map((o) => o.direction).sort()).toEqual(['return', 'send']);
      expect(new Set(ops.map((o) => o.microBatch)).size).toBe(2);
    }
  });
});

describe('pipeline autoplay (ADR §7)', () => {
  it('PD on: every slot runs Prefill, layer-wise KV and Decode at the same time', () => {
    for (const s of COMBOS.filter((c) => c.pd)) {
      const p = buildPipeline(s, PROMPT, 8);
      expect(p.cyclic).toBe(true);
      for (const slot of p.slots) {
        expect([...phases(slot)].sort(), label(s)).toEqual(['decode', 'prefill']);
        expect(purposes(slot).has('kv')).toBe(true);
        expect(slot.lanes.map((l) => l.lane)).toEqual(['prefill', 'kv', 'decode']);
      }
    }
  });

  it('PD on: KV only carries layers already finished, and each batch sends all 61 layers once', () => {
    const p = buildPipeline(COMBOS.find((c) => c.pd)!, PROMPT, 8);
    let finishedBefore = 0; // layers of batch n finished before the current slot
    const sentByBatch = new Map<number, number>();
    for (const slot of p.slots) {
      const kv = slot.lanes.find((l) => l.lane === 'kv')!;
      const [from, to] = kv.layers!;
      if (kv.batch === 0) expect(to).toBeLessThanOrEqual(finishedBefore);
      sentByBatch.set(kv.batch, (sentByBatch.get(kv.batch) ?? 0) + (to - from + 1) / 2);
      finishedBefore = slot.lanes.find((l) => l.lane === 'prefill')!.layers![1];
      // Decode runs a batch whose KV is complete (two batches back).
      expect(slot.lanes.find((l) => l.lane === 'decode')!.batch).toBe(-2);
    }
    // One cycle sends n's head and n-1's tail; together that is one full KV.
    expect((sentByBatch.get(0) ?? 0) + (sentByBatch.get(-1) ?? 0)).toBe(LAYERS);
  });

  it('PD off: Prefill and Decode never share a slot, and Prefill comes first', () => {
    for (const s of COMBOS.filter((c) => !c.pd)) {
      const p = buildPipeline(s, PROMPT, 8);
      expect(p.cyclic).toBe(false);
      const seq = p.slots.map((sl) => [...phases(sl)]);
      for (const ph of seq) expect(ph.length, label(s)).toBe(1);
      expect(seq[0]).toEqual(['prefill']);
      for (const slot of p.slots) expect(purposes(slot).has('kv')).toBe(false);
    }
  });

  it('decode returns mirror the matching sends (same requests and expert GPUs)', () => {
    for (const s of COMBOS) {
      const p = buildPipeline(s, PROMPT, 8);
      const flowsOf = (dir: 'send' | 'return', key: string) =>
        p.slots
          .flatMap((sl) => sl.stage.traffic?.flows ?? [])
          .filter((f) => f.phase === 'decode' && f.id.startsWith(`d-${dir}-${key}:`))
          .map((f) => (dir === 'send' ? `${f.request}|${f.src}>${f.dst}|${f.tokens}` : `${f.request}|${f.dst}>${f.src}|${f.tokens}`))
          .sort();
      for (const key of ['0', '1', 'all']) expect(flowsOf('return', key), `${label(s)} ${key}`).toEqual(flowsOf('send', key));
    }
  });

  it('flow ids stay unique after merging lanes', () => {
    for (const s of COMBOS) {
      for (const slot of buildPipeline(s, PROMPT, 8).slots) {
        const ids = slot.stage.traffic?.flows.map((f) => f.id) ?? [];
        expect(new Set(ids).size).toBe(ids.length);
      }
    }
  });
});
