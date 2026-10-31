// SVG renderer: static topology, per-stage link load, batch traffic as moving
// dashes and the tracked request's flows as particles along their paths.

import type { Path } from '../model/routing';
import { roleLabel, type Roles } from '../model/scenario';
import type { Stage } from '../model/steps';
import { NODES, type Link, type Topology } from '../model/topology';
import { linkLoad, type Flow } from '../model/traffic';
import { anchor, buildLayout, hopPoints, VIEW_H, VIEW_W, type Layout, type Point } from './layout';
import type { BlockId, CardRole, GpuCard } from '../model/gpuview';
import { drawCards, type CardsOptions } from './gpucards';

const NS = 'http://www.w3.org/2000/svg';

export type Target =
  | { type: 'gpu'; id: string }
  | { type: 'node'; node: number }
  | { type: 'switch'; id: string }
  | { type: 'link'; id: string }
  | { type: 'flow'; flow: Flow }
  | { type: 'block'; gpu: string; role: CardRole; block: BlockId };

export interface ViewHandlers {
  onHover(target: Target | null, event: MouseEvent): void;
  onClick(target: Target): void;
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

/** Make an SVG element focusable and activatable like a button. */
function interactive(e: SVGElement, label: string, target: () => Target, h: ViewHandlers) {
  e.setAttribute('tabindex', '0');
  e.setAttribute('role', 'button');
  e.setAttribute('aria-label', label);
  e.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      ev.stopPropagation();
      h.onClick(target());
    }
  });
  e.addEventListener('focus', () => {
    const r = e.getBoundingClientRect();
    h.onHover(target(), new MouseEvent('mousemove', { clientX: r.right, clientY: r.top }));
  });
  e.addEventListener('blur', (ev) => h.onHover(null, ev as unknown as MouseEvent));
}

function pathD(points: Point[]): string {
  return points.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
}

/** Points along a full path, merging consecutive hop endpoints. */
export function pathPoints(layout: Layout, path: Path): Point[] {
  const pts: Point[] = [];
  for (const seg of path.segments) {
    for (const hop of seg.hops) {
      const [a, b] = hopPoints(layout, hop);
      pts.push(a, b);
    }
  }
  return pts;
}

function interpolate(points: Point[], t: number): Point {
  if (points.length < 2) return points[0] ?? { x: 0, y: 0 };
  const lens: number[] = [];
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i]!.x - points[i - 1]!.x, points[i]!.y - points[i - 1]!.y);
    lens.push(d);
    total += d;
  }
  let target = t * total;
  for (let i = 0; i < lens.length; i++) {
    if (target <= lens[i]! || i === lens.length - 1) {
      const f = lens[i]! ? Math.min(target / lens[i]!, 1) : 1;
      const a = points[i]!;
      const b = points[i + 1]!;
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f };
    }
    target -= lens[i]!;
  }
  return points.at(-1)!;
}

export class FabricView {
  private layout!: Layout;
  private topo!: Topology;
  private linkEls = new Map<string, SVGPathElement>();
  private gpuEls = new Map<string, SVGGElement>();
  private switchEls = new Map<string, SVGGElement>();
  private nodeEls: SVGRectElement[] = [];
  private roleEls: SVGTextElement[] = [];
  private overlay!: SVGGElement;
  private cardsG!: SVGGElement;
  private raf = 0;
  /** Flows of the stage currently shown (for click/hover lookups). */
  stageFlows: Flow[] = [];
  /** Directed bytes of the current stage: key `${linkId}|${from}>${to}`. */
  stageLoad = new Map<string, number>();

  constructor(
    private svg: SVGSVGElement,
    private handlers: ViewHandlers,
  ) {
    svg.setAttribute('viewBox', `0 0 ${VIEW_W} ${VIEW_H}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  }

  setTopology(topo: Topology, roles: Roles, roleText: (gpu: string) => string, nodeText: (n: number) => string) {
    cancelAnimationFrame(this.raf);
    this.topo = topo;
    this.layout = buildLayout(topo);
    this.svg.replaceChildren();
    this.linkEls.clear();
    this.gpuEls.clear();
    this.switchEls.clear();
    this.nodeEls = [];
    this.roleEls = [];

    const nodesG = el('g');
    this.layout.nodes.forEach((b, n) => {
      const rect = el('rect', { x: b.x, y: b.y, width: b.w, height: b.h, rx: 10, class: 'node-box' });
      rect.addEventListener('mousemove', (e) => this.handlers.onHover({ type: 'node', node: n }, e));
      rect.addEventListener('mouseleave', (e) => this.handlers.onHover(null, e));
      rect.addEventListener('click', () => this.handlers.onClick({ type: 'node', node: n }));
      interactive(rect, `Node${n}`, () => ({ type: 'node', node: n }), this.handlers);
      nodesG.append(rect);
      const label = el('text', { x: b.x + 12, y: b.y + b.h - 30, class: 'node-label' });
      label.textContent = nodeText(n);
      const role = el('text', { x: b.x + 12, y: b.y + b.h - 12, class: 'role-label' });
      nodesG.append(label, role);
      this.nodeEls.push(rect);
      this.roleEls.push(role);
    });

    const linksG = el('g');
    const hitsG = el('g');
    for (const link of topo.links) {
      const a = anchor(this.layout, link.a, link.b, link.parallel, link.kind);
      const b = anchor(this.layout, link.b, link.a, link.parallel, link.kind);
      const d = pathD([a, b]);
      const line = el('path', { d, class: `link ${link.kind}` });
      linksG.append(line);
      this.linkEls.set(link.id, line);
      const hit = el('path', { d, class: 'link-hit' });
      hit.addEventListener('mousemove', (e) => this.handlers.onHover({ type: 'link', id: link.id }, e));
      hit.addEventListener('mouseleave', (e) => this.handlers.onHover(null, e));
      hit.addEventListener('click', () => this.handlers.onClick({ type: 'link', id: link.id }));
      if (link.kind !== 'nvlink') interactive(hit, `${link.a} – ${link.b}`, () => ({ type: 'link', id: link.id }), this.handlers);
      hitsG.append(hit);
    }

    const devG = el('g');
    for (const sw of topo.switches) {
      const b = this.layout.boxes.get(sw.id)!;
      const g = el('g', { class: `switch ${sw.kind}` });
      g.append(el('rect', { x: b.x, y: b.y, width: b.w, height: b.h }));
      const text = el('text', { x: b.x + b.w / 2, y: b.y + b.h / 2 + 4, 'text-anchor': 'middle' });
      text.textContent =
        sw.kind === 'nvswitch' ? 'Scale-up Switch' : sw.kind === 'spine' ? `Spine${sw.index}` : topo.wiring === 'rail' ? `Leaf${sw.index} (R${sw.index})` : `Leaf${sw.index} (ToR)`;
      g.append(text);
      g.addEventListener('mousemove', (e) => this.handlers.onHover({ type: 'switch', id: sw.id }, e));
      g.addEventListener('mouseleave', (e) => this.handlers.onHover(null, e));
      g.addEventListener('click', () => this.handlers.onClick({ type: 'switch', id: sw.id }));
      interactive(g, text.textContent, () => ({ type: 'switch', id: sw.id }), this.handlers);
      devG.append(g);
      this.switchEls.set(sw.id, g);
    }
    for (const gpu of topo.gpus) {
      const b = this.layout.boxes.get(gpu.id)!;
      const g = el('g', { class: 'gpu' });
      g.append(el('rect', { x: b.x, y: b.y, width: b.w, height: b.h }));
      const text = el('text', { x: b.x + b.w / 2, y: b.y + b.h / 2 + 4, 'text-anchor': 'middle' });
      text.textContent = String(gpu.local);
      g.append(text);
      g.addEventListener('mousemove', (e) => this.handlers.onHover({ type: 'gpu', id: gpu.id }, e));
      g.addEventListener('mouseleave', (e) => this.handlers.onHover(null, e));
      g.addEventListener('click', (e) => {
        e.stopPropagation();
        this.handlers.onClick({ type: 'gpu', id: gpu.id });
      });
      interactive(g, `Node${gpu.node} GPU${gpu.local}`, () => ({ type: 'gpu', id: gpu.id }), this.handlers);
      devG.append(g);
      this.gpuEls.set(gpu.id, g);
    }

    this.overlay = el('g');
    this.cardsG = el('g');
    this.svg.append(this.cardsG, nodesG, linksG, hitsG, devG, this.overlay);
    this.setRoles(roles, roleText);
  }

  setRoles(roles: Roles, roleText: (gpu: string) => string) {
    for (let n = 0; n < NODES; n++) {
      const sample = `g${n}.0`;
      const role = roleLabel(roles, sample);
      this.nodeEls[n]!.setAttribute('class', `node-box role-${role}`);
      this.roleEls[n]!.textContent = roleText(sample);
    }
  }

  setTracked(gpus: string[]) {
    for (const [id, g] of this.gpuEls) g.classList.toggle('tracked', gpus.includes(id));
  }

  /** Apply a stage: GPU activity, link heat and moving traffic. Resolves when particles arrive. */
  showStage(stage: Stage | null, durationMs: number, onDone?: () => void) {
    cancelAnimationFrame(this.raf);
    this.overlay.replaceChildren();
    for (const [, g] of this.gpuEls) {
      g.classList.forEach((c) => c.startsWith('act-') && g.classList.remove(c));
      g.classList.remove('busy');
    }
    for (const [, l] of this.linkEls) {
      l.classList.remove('flowing', 'reverse', 'heat-1', 'heat-2', 'heat-3', 'heat-4', 'heat-5');
    }
    this.stageFlows = [];
    this.stageLoad = new Map();
    if (!stage) {
      onDone?.();
      return;
    }
    for (const [gpu, act] of stage.activity) {
      const g = this.gpuEls.get(gpu);
      g?.classList.add('busy', `act-${act}`);
    }
    if (!stage.traffic) {
      onDone?.();
      return;
    }
    this.stageFlows = stage.traffic.flows;
    const load = linkLoad(stage.traffic.transfers);
    this.stageLoad = load;
    // Per link: max over directions, and which direction dominates.
    const perLink = new Map<string, { bytes: number; forward: boolean }>();
    for (const [key, bytes] of load) {
      const [linkId, dir] = key.split('|') as [string, string];
      const from = dir.split('>')[0]!;
      const link = this.linkById(linkId);
      const forward = link.a === from;
      const cur = perLink.get(linkId);
      if (!cur || bytes > cur.bytes) perLink.set(linkId, { bytes, forward });
    }
    let max = 0;
    for (const v of perLink.values()) max = Math.max(max, v.bytes);
    for (const [id, v] of perLink) {
      const line = this.linkEls.get(id)!;
      const level = Math.max(1, Math.ceil((v.bytes / max) * 5));
      line.classList.add(`heat-${level}`, 'flowing');
      if (!v.forward) line.classList.add('reverse');
    }
    this.animateTracked(stage.traffic.flows.filter((f) => f.tracked), durationMs, onDone);
  }

  private linkById(id: string): Link {
    return this.topo.linkMap.get(id)!;
  }

  private animateTracked(flows: Flow[], durationMs: number, onDone?: () => void) {
    const items = flows.map((flow) => {
      const pts = pathPoints(this.layout, flow.path);
      const trail = el('path', { d: pathD(pts), class: `trail p-${flow.purpose}` });
      trail.dataset.flow = flow.id;
      const dot = el('circle', { r: 6, class: `particle p-${flow.purpose}` });
      dot.dataset.flow = flow.id;
      dot.addEventListener('mousemove', (e) => this.handlers.onHover({ type: 'flow', flow }, e));
      dot.addEventListener('mouseleave', (e) => this.handlers.onHover(null, e));
      dot.addEventListener('click', (e) => {
        e.stopPropagation();
        this.handlers.onClick({ type: 'flow', flow });
      });
      this.overlay.append(trail, dot);
      return { pts, dot };
    });
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - start) / durationMs, 1);
      const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
      for (const { pts, dot } of items) {
        const p = interpolate(pts, eased);
        dot.setAttribute('cx', p.x.toFixed(1));
        dot.setAttribute('cy', p.y.toFixed(1));
      }
      if (t < 1) this.raf = requestAnimationFrame(tick);
      else onDone?.();
    };
    this.raf = requestAnimationFrame(tick);
  }

  showCards(cards: GpuCard[], opts: CardsOptions) {
    drawCards(this.cardsG, this.layout, cards, opts, (gpu, role, block) => this.handlers.onClick({ type: 'block', gpu, role, block }));
  }

  stop() {
    cancelAnimationFrame(this.raf);
  }

  /** Dim everything except the elements related to a target. */
  highlight(target: Target | null, related: { links: Set<string>; gpus: Set<string>; switches: Set<string>; flows: Set<string> }) {
    this.svg.classList.toggle('dimmed', !!target);
    for (const [id, e] of this.linkEls) e.classList.toggle('hl', related.links.has(id));
    for (const [id, e] of this.gpuEls) e.classList.toggle('hl', related.gpus.has(id));
    for (const [id, e] of this.switchEls) e.classList.toggle('hl', related.switches.has(id));
    for (const e of this.overlay.querySelectorAll<SVGElement>('[data-flow]')) {
      e.classList.toggle('hl', related.flows.has(e.dataset.flow!));
    }
  }
}
