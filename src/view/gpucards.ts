// Inside-the-GPU cards drawn under the nodes (ADR §22): compute blocks,
// local experts, layer progress and an HBM breakdown for each representative GPU.

import type { Lang } from '../content/text';
import { formatBytes, tr, UI } from '../content/ui';
import type { BlockId, CardRole, GpuCard } from '../model/gpuview';
import { CARDS_H, CARDS_Y, VIEW_W, type Layout } from './layout';

const NS = 'http://www.w3.org/2000/svg';

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

function text(x: number, y: number, s: string, cls: string, anchor = 'start'): SVGTextElement {
  const e = el('text', { x, y, class: cls, 'text-anchor': anchor });
  e.textContent = s;
  return e;
}

/** Short labels that fit narrow blocks; the full name is in the tooltip. */
const BLOCK_LABEL: Record<BlockId, string> = {
  embed: 'Emb',
  attention: 'Attn',
  kv: 'KV',
  dense: 'Dense',
  router: 'Router',
  experts: 'Experts',
  combine: 'Comb',
  head: 'Head',
  send: 'Send',
  recv: 'Recv',
};

const BLOCK_TITLE: Record<BlockId, string> = {
  embed: 'Embedding',
  attention: 'Attention',
  kv: 'KV Cache',
  dense: 'Dense FFN',
  router: 'Router (top-k)',
  experts: 'Routed experts',
  combine: 'Combine + residual',
  head: 'Output head + sampling',
  send: 'Send (dispatch / A→F / combine)',
  recv: 'Receive',
};

const ROLE_TAG: Record<CardRole, string> = { U: 'U', P: 'P', D: 'D', A: 'A', F: 'F', DA: 'D-A', DF: 'D-F' };

export interface CardsOptions {
  lang: Lang;
  /** Byte value that maps to a full HBM bar (largest total in this scenario). */
  hbmScale: number;
}

export function drawCards(
  group: SVGGElement,
  layout: Layout,
  cards: GpuCard[],
  opts: CardsOptions,
  onBlock: (gpu: string, role: CardRole, block: BlockId) => void,
) {
  group.replaceChildren();
  if (!cards.length) return;
  const margin = 40;
  const gap = 20;
  const width = Math.min(520, (VIEW_W - 2 * margin - gap * (cards.length - 1)) / cards.length);
  const total = width * cards.length + gap * (cards.length - 1);
  const left = (VIEW_W - total) / 2;
  const L = (k: keyof typeof UI) => tr(UI[k], opts.lang);
  group.append(text(VIEW_W - 40, CARDS_Y + CARDS_H + 6, L('cardsNote'), 'card-sub', 'end'));

  cards.forEach((card, i) => {
    const x = left + i * (width + gap);
    const y = CARDS_Y;
    const g = el('g', { class: `gpu-card role-${card.role}` });

    // Connector from the card to its GPU in the topology.
    const gpuBox = layout.boxes.get(card.gpu)!;
    g.append(el('path', { d: `M${x + width / 2},${y} L${gpuBox.x + gpuBox.w / 2},${gpuBox.y + gpuBox.h}`, class: 'card-connector' }));
    g.append(el('rect', { x, y, width, height: CARDS_H, rx: 10, class: 'card-box' }));

    const [node, local] = card.gpu.slice(1).split('.');
    g.append(text(x + 12, y + 20, `${ROLE_TAG[card.role]} · Node${node} GPU${local}`, 'card-title'));
    g.append(text(x + width - 12, y + 20, tr(card.layer, opts.lang), 'card-sub', 'end'));

    // Compute blocks, left to right.
    const bx0 = x + 12;
    const bw = width - 24;
    const n = card.blocks.length;
    const bgap = 5;
    const cell = (bw - bgap * (n - 1)) / n;
    card.blocks.forEach((b, k) => {
      const bx = bx0 + k * (cell + bgap);
      const active = card.active.has(b);
      const blk = el('g', { class: `blk blk-${b}${active ? ' on' : ''}` });
      blk.append(el('rect', { x: bx, y: y + 32, width: cell, height: 30, rx: 5 }));
      blk.append(text(bx + cell / 2, y + 51, BLOCK_LABEL[b], 'blk-label', 'middle'));
      const title = el('title');
      title.textContent = BLOCK_TITLE[b];
      blk.append(title);
      // Click (or Enter/Space) zooms into the block in the right panel (ADR §23).
      blk.setAttribute('tabindex', '0');
      blk.setAttribute('role', 'button');
      blk.setAttribute('aria-label', BLOCK_TITLE[b]);
      blk.addEventListener('click', (e) => {
        e.stopPropagation();
        onBlock(card.gpu, card.role, b);
      });
      blk.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          e.stopPropagation();
          onBlock(card.gpu, card.role, b);
        }
      });
      g.append(blk);
      if (k < n - 1) g.append(el('path', { d: `M${bx + cell},${y + 47} l${bgap},0`, class: 'blk-arrow' }));
    });

    // Local experts grid; lit cells are experts the tracked request selected here.
    let gy = y + 74;
    if (card.experts.length) {
      g.append(text(x + 12, gy + 9, `${L('localExperts')} ${card.experts[0]}–${card.experts.at(-1)} (${card.lit.size} ${L('selected')})`, 'card-sub'));
      const perRow = Math.min(card.experts.length, 32);
      const sz = Math.min(11, (bw - (perRow - 1) * 2) / perRow);
      card.experts.forEach((e, k) => {
        const r = Math.floor(k / perRow);
        const c = k % perRow;
        const sq = el('rect', { x: bx0 + c * (sz + 2), y: gy + 14 + r * (sz + 2), width: sz, height: sz, rx: 2, class: `xp${card.lit.has(e) ? ' lit' : ''}` });
        const title = el('title');
        title.textContent = `Expert ${e}`;
        sq.append(title);
        g.append(sq);
      });
      gy += 14 + Math.ceil(card.experts.length / perRow) * (sz + 2) + 6;
    } else {
      g.append(text(x + 12, gy + 9, L('noLocalExperts'), 'card-sub'));
      gy += 18;
    }

    // HBM breakdown, relative to the largest total in this scenario.
    const barY = y + CARDS_H - 34;
    const scale = opts.hbmScale || 1;
    const parts: [keyof typeof card.hbm, string][] = [
      ['weights', 'hbm-w'],
      ['kv', 'hbm-kv'],
      ['activations', 'hbm-act'],
    ];
    g.append(el('rect', { x: bx0, y: barY, width: bw, height: 12, rx: 3, class: 'hbm-track' }));
    let cx = bx0;
    for (const [key, cls] of parts) {
      // Clamp so a bar never leaves its track even if the scale is stale.
      const w = Math.min((card.hbm[key] / scale) * bw, bx0 + bw - cx);
      if (w > 0.2) g.append(el('rect', { x: cx, y: barY, width: w, height: 12, class: cls }));
      cx += w;
    }
    const sum = card.hbm.weights + card.hbm.kv + card.hbm.activations;
    // Title above the bar, and a color-keyed legend below it.
    g.append(text(bx0, barY - 5, `${L('hbmTitle')} ≈ ${formatBytes(sum)}`, 'card-sub hbm-title'));
    const legend: [string, string, number][] = [
      ['hbm-w', L('weights'), card.hbm.weights],
      ['hbm-kv', 'KV Cache', card.hbm.kv],
      ['hbm-act', L('activations'), card.hbm.activations],
    ];
    let lx = bx0;
    for (const [cls, name, v] of legend) {
      g.append(el('rect', { x: lx, y: barY + 18, width: 9, height: 9, rx: 2, class: cls }));
      const label = text(lx + 13, barY + 26, `${name} ${formatBytes(v)}`, 'card-sub');
      g.append(label);
      lx += 13 + Math.max(90, (name.length + 9) * 6.2);
    }
    group.append(g);
  });
}
