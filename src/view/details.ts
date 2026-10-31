// Hover tooltips, pinned detail panels and highlight sets (ADR §12 of the UI:
// Decision 12 "element details").

import type { I18n, Lang } from '../content/text';
import { ACTIVITY_LABEL, formatBytes, METHOD_LABEL, PURPOSE_LABEL, ROLE_LABEL, UI, tr } from '../content/ui';
import type { ModelSpec } from '../model/constants';
import { expertsOn, roleLabel, type Roles } from '../model/scenario';
import type { Stage } from '../model/steps';
import type { Topology } from '../model/topology';
import type { Flow } from '../model/traffic';
import type { Target } from './render';

export interface Related {
  links: Set<string>;
  gpus: Set<string>;
  switches: Set<string>;
  flows: Set<string>;
}

export interface DetailContext {
  topo: Topology;
  roles: Roles;
  model: ModelSpec;
  /** Phase of the current step, to pick the expert placement; 'both' in the overlap view. */
  phase: 'prefill' | 'decode' | 'both';
  stage: Stage | null;
  flows: Flow[];
  load: Map<string, number>;
  lang: Lang;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const L = (text: I18n, lang: Lang) => esc(tr(text, lang));

function flowTouches(f: Flow, pred: (hop: { linkId: string; from: string; to: string }) => boolean): boolean {
  return f.path.segments.some((s) => s.hops.some((h) => pred({ linkId: h.link.id, from: h.from, to: h.to })));
}

function addFlow(r: Related, f: Flow) {
  r.flows.add(f.id);
  r.gpus.add(f.src);
  r.gpus.add(f.dst);
  for (const s of f.path.segments) {
    r.gpus.add(s.from);
    r.gpus.add(s.to);
    for (const h of s.hops) {
      r.links.add(h.link.id);
      for (const e of [h.from, h.to]) if (!e.startsWith('g')) r.switches.add(e);
    }
  }
}

export function relatedOf(target: Target, ctx: DetailContext): Related {
  const r: Related = { links: new Set(), gpus: new Set(), switches: new Set(), flows: new Set() };
  switch (target.type) {
    case 'flow':
      addFlow(r, target.flow);
      break;
    case 'gpu': {
      r.gpus.add(target.id);
      r.links.add(ctx.topo.accessOfGpu.get(target.id)!.id);
      r.links.add(ctx.topo.nvlinkOfGpu.get(target.id)!.id);
      for (const f of ctx.flows) if (f.src === target.id || f.dst === target.id) addFlow(r, f);
      break;
    }
    case 'node':
      for (const g of ctx.topo.gpus.filter((x) => x.node === target.node)) {
        r.gpus.add(g.id);
        r.links.add(ctx.topo.accessOfGpu.get(g.id)!.id);
        r.links.add(ctx.topo.nvlinkOfGpu.get(g.id)!.id);
      }
      r.switches.add(`nvs${target.node}`);
      break;
    case 'switch':
      r.switches.add(target.id);
      for (const l of ctx.topo.links) if (l.a === target.id || l.b === target.id) r.links.add(l.id);
      for (const f of ctx.flows) if (flowTouches(f, (h) => h.to === target.id)) addFlow(r, f);
      break;
    case 'block':
      r.gpus.add(target.gpu);
      break;
    case 'link':
      r.links.add(target.id);
      for (const f of ctx.flows) if (flowTouches(f, (h) => h.linkId === target.id)) addFlow(r, f);
      break;
  }
  return r;
}

/** Both directions of a link: [a -> b bytes, b -> a bytes]. */
function directed(ctx: DetailContext, linkId: string): [number, number] {
  const l = ctx.topo.linkMap.get(linkId)!;
  return [ctx.load.get(`${linkId}|${l.a}>${l.b}`) ?? 0, ctx.load.get(`${linkId}|${l.b}>${l.a}`) ?? 0];
}

function flowLine(f: Flow, lang: Lang): string {
  const who = f.tracked ? tr(UI.tracked, lang) : tr(UI.batch, lang);
  return `${L(PURPOSE_LABEL[f.purpose], lang)} ${esc(f.src)} → ${esc(f.dst)} · ${f.path.switchCount} sw · ${esc(who)}`;
}

function hopList(f: Flow): string {
  const out: string[] = [f.src];
  for (const s of f.path.segments) {
    for (const h of s.hops) out.push(s.kind === 'nvlink' && h.to.startsWith('nvs') ? `${h.to}` : h.to);
  }
  return out.map(esc).join(' → ');
}

function gpuRail(ctx: DetailContext, id: string): number {
  return ctx.topo.gpuMap.get(id)!.local;
}

function expertsText(ctx: DetailContext, gpu: string): string {
  const fmt = (e: number[]) => `${e[0]}–${e.at(-1)}`;
  if (ctx.phase !== 'both') {
    const e = expertsOn(ctx.roles[ctx.phase], gpu, ctx.model);
    return e.length ? fmt(e) : '—';
  }
  const p = expertsOn(ctx.roles.prefill, gpu, ctx.model);
  const d = expertsOn(ctx.roles.decode, gpu, ctx.model);
  if (ctx.roles.prefill === ctx.roles.decode || (p.join() === d.join())) return p.length ? fmt(p) : '—';
  const parts = [p.length ? `Prefill ${fmt(p)}` : '', d.length ? `Decode ${fmt(d)}` : ''].filter(Boolean);
  return parts.length ? parts.join(' / ') : '—';
}

export function tooltipHtml(target: Target, ctx: DetailContext): string {
  const { lang } = ctx;
  switch (target.type) {
    case 'flow': {
      const f = target.flow;
      return `<b>${L(PURPOSE_LABEL[f.purpose], lang)}</b> · ${L(METHOD_LABEL[f.method], lang)}<br>${esc(f.src)} → ${esc(f.dst)}<br>${L(UI.switches, lang)}: ${f.path.switchCount} · ${L(UI.relays, lang)}: ${f.path.relayCount}<br>${f.tokens} tok · ${formatBytes(f.logicalBytes)}`;
    }
    case 'gpu': {
      const g = ctx.topo.gpuMap.get(target.id)!;
      const act = ctx.stage?.activity.get(target.id);
      return `<b>Node${g.node} GPU${g.local}</b> (Rail ${g.local})<br>${L(ROLE_LABEL[roleLabel(ctx.roles, target.id)], lang)}<br>${act ? L(ACTIVITY_LABEL[act], lang) : L(UI.idle, lang)}`;
    }
    case 'node':
      return `<b>Node${target.node}</b><br>${L(ROLE_LABEL[roleLabel(ctx.roles, `g${target.node}.0`)], lang)}<br>8 GPU + Scale-up Switch`;
    case 'switch': {
      const n = ctx.flows.filter((f) => flowTouches(f, (h) => h.to === target.id)).length;
      return `<b>${esc(target.id)}</b><br>${n} flows`;
    }
    case 'block':
      return `<b>${esc(target.block)}</b> · ${esc(target.gpu)}`;
    case 'link': {
      const l = ctx.topo.linkMap.get(target.id)!;
      const [ab, ba] = directed(ctx, target.id);
      return `<b>${esc(l.a)} ↔ ${esc(l.b)}</b><br>${l.kind === 'nvlink' ? 'Scale-up' : 'Scale-out'}<br>${esc(l.a)} → ${esc(l.b)}: ${formatBytes(ab)}<br>${esc(l.b)} → ${esc(l.a)}: ${formatBytes(ba)}`;
    }
  }
}

export function detailHtml(target: Target, ctx: DetailContext): string {
  const { lang } = ctx;
  const flowsList = (flows: Flow[]) => {
    if (!flows.length) return '';
    const shown = [...flows].sort((a, b) => Number(b.tracked) - Number(a.tracked)).slice(0, 14);
    return `<ul class="flows">${shown.map((f) => `<li>${flowLine(f, lang)}</li>`).join('')}${flows.length > shown.length ? `<li>… +${flows.length - shown.length}</li>` : ''}</ul>`;
  };
  let body = '';
  switch (target.type) {
    case 'flow': {
      const f = target.flow;
      body = `<h2>${L(PURPOSE_LABEL[f.purpose], lang)}</h2>
        <dl>
          <dt>Src → Dst</dt><dd>${esc(f.src)} → ${esc(f.dst)}</dd>
          <dt>Method</dt><dd>${L(METHOD_LABEL[f.method], lang)}</dd>
          <dt>${L(UI.switches, lang)}</dt><dd>${f.path.switchCount}</dd>
          <dt>${L(UI.relays, lang)}</dt><dd>${f.path.relayCount}</dd>
          <dt>Tokens</dt><dd>${f.tokens}</dd>
          <dt>Bytes</dt><dd>${formatBytes(f.logicalBytes)}</dd>
          <dt>Request</dt><dd>${f.tracked ? L(UI.tracked, lang) : L(UI.batch, lang)}</dd>
        </dl>
        <h3>Path</h3><p>${hopList(f)}</p>`;
      break;
    }
    case 'gpu': {
      const g = ctx.topo.gpuMap.get(target.id)!;
      const act = ctx.stage?.activity.get(target.id);
      const flows = ctx.flows.filter((f) => f.src === target.id || f.dst === target.id);
      body = `<h2>Node${g.node} GPU${g.local}</h2>
        <dl>
          <dt>Rail / NIC</dt><dd>${gpuRail(ctx, target.id)} / ${esc(g.nicId)} → ${esc(ctx.topo.leafOfGpu.get(g.id)!)}</dd>
          <dt>Role</dt><dd>${L(ROLE_LABEL[roleLabel(ctx.roles, target.id)], lang)}</dd>
          <dt>Routed experts</dt><dd>${expertsText(ctx, target.id)}</dd>
          <dt>Now</dt><dd>${act ? L(ACTIVITY_LABEL[act], lang) : L(UI.idle, lang)}</dd>
        </dl>${flowsList(flows)}`;
      break;
    }
    case 'node': {
      const rows = ctx.topo.gpus
        .filter((x) => x.node === target.node)
        .map((x) => `<dt>GPU${x.local} ${esc(x.nicId)}</dt><dd>${esc(ctx.topo.leafOfGpu.get(x.id)!)}</dd>`)
        .join('');
      body = `<h2>Node${target.node}</h2><p>${L(ROLE_LABEL[roleLabel(ctx.roles, `g${target.node}.0`)], lang)} · 8 GPU + Scale-up Switch</p><dl>${rows}</dl>`;
      break;
    }
    case 'switch': {
      const sw = ctx.topo.switches.find((s) => s.id === target.id)!;
      const peers = ctx.topo.links.filter((l) => l.a === target.id || l.b === target.id).map((l) => (l.a === target.id ? l.b : l.a));
      const flows = ctx.flows.filter((f) => flowTouches(f, (h) => h.to === target.id));
      const rail = sw.rail !== undefined ? ` · Rail ${sw.rail}` : sw.kind === 'leaf' ? ` · ToR (Node${sw.node})` : '';
      body = `<h2>${esc(target.id)}</h2><p>${esc(sw.kind)}${rail}</p>
        <h3>Links</h3><p>${[...new Set(peers)].map(esc).join(', ')}</p>
        <h3>${flows.length} flows</h3>${flowsList(flows)}`;
      break;
    }
    case 'block':
      // Rendered by the block zoom (main.ts), which needs the card and workload.
      break;
    case 'link': {
      const l = ctx.topo.linkMap.get(target.id)!;
      const flows = ctx.flows.filter((f) => flowTouches(f, (h) => h.linkId === target.id));
      body = `<h2>${esc(l.a)} ↔ ${esc(l.b)}</h2>
        <dl>
          <dt>Type</dt><dd>${l.kind === 'nvlink' ? 'Scale-up' : `Scale-out (${esc(l.kind)})`}</dd>
          <dt>Parallel</dt><dd>#${l.parallel}</dd>
          <dt>${esc(l.a)} → ${esc(l.b)}</dt><dd>${formatBytes(directed(ctx, target.id)[0])}</dd>
          <dt>${esc(l.b)} → ${esc(l.a)}</dt><dd>${formatBytes(directed(ctx, target.id)[1])}</dd>
        </dl>
        <p class="kicker">${L(UI.linkLoad, lang)}: ${L(UI.heatNote, lang)}</p>
        <h3>${flows.length} flows</h3>${flowsList(flows)}`;
      break;
    }
  }
  return `<button type="button" data-action="unpin">← ${L(UI.close, lang)}</button>${body}`;
}
