// SVG coordinates for every topology element (ADR §11: topology center stage).

import type { Hop } from '../model/routing';
import { GPUS_PER_NODE, NODES, SPINES, type Topology } from '../model/topology';

export const VIEW_W = 1200;
export const VIEW_H = 842;
/** Top of the inside-the-GPU cards area, below the nodes (ADR §22). */
export const CARDS_Y = 628;
export const CARDS_H = 192;

const SPINE_Y = 50;
const LEAF_Y = 200;
const NODE_Y = 340;
const NODE_H = 270;
const GPU_Y = NODE_Y + 60;
const NVS_Y = NODE_Y + 190;
export const GPU_W = 26;
export const GPU_H = 40;
export const SWITCH_W = 88;
export const SWITCH_H = 30;
const MARGIN = 40;
const NODE_GAP = 24;
export const NODE_W = (VIEW_W - 2 * MARGIN - (NODES - 1) * NODE_GAP) / NODES;

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Layout {
  boxes: Map<string, Box>;
  nodes: Box[];
}

function spread(count: number, width: number, itemW: number, left = MARGIN): number[] {
  const usable = width - 2 * MARGIN;
  const step = usable / count;
  return Array.from({ length: count }, (_, i) => left + step * i + step / 2 - itemW / 2);
}

export function buildLayout(topo: Topology): Layout {
  const boxes = new Map<string, Box>();
  const nodes: Box[] = [];
  for (let n = 0; n < NODES; n++) {
    const x = MARGIN + n * (NODE_W + NODE_GAP);
    nodes.push({ x, y: NODE_Y, w: NODE_W, h: NODE_H });
    const inner = NODE_W - 24;
    const step = inner / GPUS_PER_NODE;
    for (let l = 0; l < GPUS_PER_NODE; l++) {
      boxes.set(`g${n}.${l}`, { x: x + 12 + step * l + (step - GPU_W) / 2, y: GPU_Y, w: GPU_W, h: GPU_H });
    }
    boxes.set(`nvs${n}`, { x: x + 12, y: NVS_Y, w: inner, h: 26 });
  }
  const leaves = topo.switches.filter((s) => s.kind === 'leaf');
  spread(leaves.length, VIEW_W, SWITCH_W).forEach((x, i) =>
    boxes.set(leaves[i]!.id, { x, y: LEAF_Y, w: SWITCH_W, h: SWITCH_H }),
  );
  spread(SPINES, VIEW_W, SWITCH_W * 1.3).forEach((x, i) =>
    boxes.set(`spine${i}`, { x, y: SPINE_Y, w: SWITCH_W * 1.3, h: SWITCH_H }),
  );
  return { boxes, nodes };
}

export interface Point {
  x: number;
  y: number;
}

/**
 * Where a link attaches to an element. Fabric links fan out across the switch
 * edge by spine/leaf index and parallel index so parallel links stay visible.
 */
export function anchor(layout: Layout, element: string, other: string, linkParallel = 0, linkKind = ''): Point {
  const b = layout.boxes.get(element)!;
  const o = layout.boxes.get(other)!;
  const cx = b.x + b.w / 2;
  if (element.startsWith('g')) {
    return linkKind === 'nvlink' ? { x: cx, y: b.y + b.h } : { x: cx, y: b.y };
  }
  if (element.startsWith('nvs')) {
    return { x: Math.min(Math.max(o.x + o.w / 2, b.x), b.x + b.w), y: b.y };
  }
  const below = o.y > b.y;
  const ox = o.x + o.w / 2;
  // Spread attachment points toward the other end, plus a small parallel offset.
  const t = Math.min(Math.max((ox - cx) / (VIEW_W / 2), -1), 1);
  const x = cx + t * (b.w / 2 - 6) + (linkParallel ? 5 : 0);
  return { x, y: below ? b.y + b.h : b.y };
}

export function hopPoints(layout: Layout, hop: Hop): [Point, Point] {
  return [
    anchor(layout, hop.from, hop.to, hop.link.parallel, hop.link.kind),
    anchor(layout, hop.to, hop.from, hop.link.parallel, hop.link.kind),
  ];
}
