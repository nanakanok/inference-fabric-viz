import './style.css';
import type { Lang } from './content/text';
import { formatBytes, notesFor, PRESETS, PURPOSE_LABEL, ROLE_LABEL, tr, UI } from './content/ui';
import { isScaleUpOnly, modelFor, roleLabel, topologyFor, type Scenario } from './model/scenario';
import { blockDetail } from './model/blockzoom';
import { cardsForPipeline, cardsForStep, hbmScaleFor, type GpuCard } from './model/gpuview';
import { blockZoomHtml } from './view/blockzoom';
import { buildPipeline, PIPELINE_NOTE, type Pipeline } from './model/pipeline';
import { buildSteps, outputTokens, trackedGpu, type StepPlan } from './model/steps';
import { linkLoad, totals } from './model/traffic';
import { buildWorkload, type Workload } from './model/workload';
import { detailHtml, relatedOf, tooltipHtml, type DetailContext } from './view/details';
import { FabricView, type Target } from './view/render';

const STAGE_MS = 1600;
const STEP_PAUSE_MS = 900;
const STAGE_GAP_MS = 400;

interface State {
  scenario: Scenario;
  lang: Lang;
  theme: 'auto' | 'light' | 'dark';
  preset: number;
  prompt: string;
  outputTokens: number;
  step: number;
  stage: number;
  playing: boolean;
  pinned: Target | null;
  /** Step-through explanation, or the looping overlapped-batches view (ADR §7). */
  mode: 'steps' | 'pipeline';
  slot: number;
}

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem('ifv.lang');
    if (saved === 'ja' || saved === 'en') return saved;
  } catch {
    /* storage unavailable */
  }
  return navigator.language.startsWith('ja') ? 'ja' : 'en';
}

const lang0 = initialLang();
const state: State = {
  scenario: { fabric: 'scaleout', pd: true, af: false, wiring: 'rail', decodeRelay: false, ecmpSeed: 1 },
  lang: lang0,
  theme: 'auto',
  preset: 1,
  prompt: PRESETS[1]!.text[lang0],
  outputTokens: PRESETS[1]!.output,
  step: 0,
  stage: 0,
  playing: false,
  pinned: null,
  mode: 'steps',
  slot: 0,
};

let workload: Workload = buildWorkload(state.prompt, state.outputTokens);
let plan: StepPlan = buildSteps(state.scenario, workload);
let pipeline: Pipeline | null = null;
/** Byte value for a full HBM bar: the largest card total in this scenario. */
let hbmScale = 1;
let timer = 0;
/** Bumped whenever pending transitions must be invalidated (rebuild, goto, pin, pause). */
let generation = 0;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const controls = $('controls');
const transport = $('transport');
const help = $('help');
const panel = $('panel');
const legend = $('legend');
const tooltip = $('tooltip');
const svg = document.getElementById('fabric') as unknown as SVGSVGElement;

const view = new FabricView(svg, {
  onHover(target, e) {
    if (!target) {
      tooltip.hidden = true;
      return;
    }
    tooltip.innerHTML = tooltipHtml(target, detailContext());
    tooltip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
    const y = Math.min(e.clientY + 14, window.innerHeight - tooltip.offsetHeight - 8);
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
  },
  onClick(target) {
    // Freeze the stage being inspected: no particle movement, no substage advance.
    pause();
    view.stop();
    state.pinned = target;
    applyHighlight();
    renderPanel();
  },
});

const t = (k: keyof typeof UI) => tr(UI[k], state.lang);

function currentStep() {
  return plan.steps[state.step]!;
}

function currentStage() {
  if (state.mode === 'pipeline' && pipeline) return pipeline.slots[state.slot]!.stage;
  return currentStep().stages[state.stage] ?? null;
}

function detailContext(): DetailContext {
  return {
    topo: topologyFor(state.scenario.wiring),
    roles: plan.roles,
    model: modelFor(state.scenario),
    phase: state.mode === 'pipeline' ? 'both' : currentStep().phase,
    stage: currentStage(),
    flows: view.stageFlows,
    load: view.stageLoad,
    lang: state.lang,
  };
}

function cancelTransitions() {
  generation++;
  clearTimeout(timer);
  view.stop();
}

function rebuild(keepStepKind = true) {
  cancelTransitions();
  const prevKind = plan.steps[state.step]?.kind;
  const prevIter = plan.steps[state.step]?.decodeIteration;
  workload = buildWorkload(state.prompt, state.outputTokens);
  plan = buildSteps(state.scenario, workload);
  pipeline = state.mode === 'pipeline' ? buildPipeline(state.scenario, state.prompt, state.outputTokens) : null;
  state.slot = 0;
  let idx = 0;
  if (keepStepKind && prevKind) {
    const same = plan.steps.findIndex((s) => s.kind === prevKind && s.decodeIteration === prevIter);
    idx = same >= 0 ? same : Math.min(state.step, plan.steps.length - 1);
  }
  state.step = idx;
  state.stage = 0;
  state.pinned = null;
  hbmScale = computeHbmScale();
  svg.classList.toggle('scaleup-only', isScaleUpOnly(state.scenario));
  view.setTopology(
    topologyFor(state.scenario.wiring),
    plan.roles,
    (g) => tr(ROLE_LABEL[roleLabel(plan.roles, g)], state.lang),
    (n) => `Node${n}`,
  );
  renderAll();
  showCurrent();
}

function currentCards(): GpuCard[] {
  if (state.mode === 'pipeline' && pipeline) return cardsForPipeline(plan, state.scenario, workload, pipeline.slots[state.slot]!);
  return cardsForStep(plan, state.scenario, workload, state.step, state.stage);
}

function computeHbmScale(): number {
  return hbmScaleFor(plan, state.scenario, workload, pipeline?.slots ?? []);
}

function showCards() {
  view.showCards(currentCards(), { lang: state.lang, hbmScale });
}

function showSlot() {
  cancelTransitions();
  const gen = generation;
  view.setTracked([trackedGpu(plan, 'prefill'), trackedGpu(plan, 'decode')]);
  showCards();
  view.showStage(currentStage(), STAGE_MS, () => {
    if (gen !== generation || !state.playing || !pipeline) return;
    timer = window.setTimeout(() => {
      if (gen !== generation || !pipeline) return;
      state.slot = (state.slot + 1) % pipeline.slots.length;
      showSlot();
    }, STAGE_GAP_MS);
  });
  applyHighlight();
  renderPanel();
  renderTransport();
}

function showCurrent() {
  if (state.mode === 'pipeline') {
    showSlot();
    return;
  }
  cancelTransitions();
  const gen = generation;
  const step = currentStep();
  view.setTracked([trackedGpu(plan, step.phase)]);
  showCards();
  view.showStage(currentStage(), STAGE_MS, () => {
    if (gen !== generation) return;
    renderPanel();
    // Stages within a step always run in order; steps advance only while playing.
    if (state.stage < step.stages.length - 1) {
      timer = window.setTimeout(() => {
        if (gen !== generation) return;
        state.stage++;
        showCurrent();
      }, STAGE_GAP_MS);
    } else if (state.playing) {
      timer = window.setTimeout(() => gen === generation && advance(), STEP_PAUSE_MS);
    }
  });
  applyHighlight();
  renderPanel();
  renderTransport();
}

function advance() {
  const step = currentStep();
  if (state.stage < step.stages.length - 1) {
    state.stage++;
  } else if (state.step < plan.steps.length - 1) {
    state.step++;
    state.stage = 0;
  } else {
    pause();
    return;
  }
  showCurrent();
}

/** Next/previous move through stages first, then steps. */
function stepForward() {
  if (state.stage < currentStep().stages.length - 1) {
    state.stage++;
    state.pinned = null;
    showCurrent();
  } else goto(state.step + 1);
}

function goto(step: number) {
  state.step = Math.max(0, Math.min(plan.steps.length - 1, step));
  state.stage = 0;
  state.pinned = null;
  showCurrent();
}

function pause() {
  state.playing = false;
  generation++;
  clearTimeout(timer);
  renderTransport();
}

function applyHighlight() {
  if (!state.pinned) {
    view.highlight(null, { links: new Set(), gpus: new Set(), switches: new Set(), flows: new Set() });
    return;
  }
  view.highlight(state.pinned, relatedOf(state.pinned, detailContext()));
}

function renderControls() {
  const s = state.scenario;
  const up = isScaleUpOnly(s);
  // Scale-up only: one node holds the whole model, so PD/AF/wiring do not apply (ADR §21).
  const off = up ? `disabled title="${t('scaleUpDisabled')}"` : '';
  const toggle = (key: 'pd' | 'af', label: string) =>
    `<button type="button" class="toggle" data-toggle="${key}" aria-pressed="${!up && s[key]}" ${off}>${label}</button>`;
  controls.innerHTML = `
    <h1>${t('title')}</h1>
    <span class="segmented" role="group" aria-label="${t('fabricLabel')}">
      <button type="button" data-fabric="scaleup" aria-pressed="${up}">${t('scaleUpOnly')}</button>
      <button type="button" data-fabric="scaleout" aria-pressed="${!up}">${t('scaleOut')}</button>
    </span>
    ${toggle('pd', t('pd'))}
    ${toggle('af', t('af'))}
    <button type="button" class="toggle" data-toggle="rail" aria-pressed="${!up && s.wiring === 'rail'}" ${off}>${t('rail')}</button>
    <span class="kicker">${t('model')}: ${modelFor(s).name}</span>
    <label>${t('prompt')}
      <select data-input="preset">
        ${PRESETS.map((p, i) => `<option value="${i}" ${i === state.preset ? 'selected' : ''}>${tr(p.name, state.lang)}</option>`).join('')}
        <option value="-1" ${state.preset < 0 ? 'selected' : ''}>${state.lang === 'ja' ? '自由入力' : 'Custom'}</option>
      </select>
    </label>
    <details class="prompt-editor"><summary>✎</summary><div>
      <textarea data-input="prompt"></textarea>
      <p class="kicker">${workload.trackedTokens.length} ${t('tokensApprox')}</p>
    </div></details>
    <label>${t('outputTokens')} <input type="number" min="1" max="4096" data-input="output" value="${state.outputTokens}" style="width:6em"></label>
    <span class="grow"></span>
    <button type="button" data-action="lang">${state.lang === 'ja' ? 'EN' : '日本語'}</button>
    <button type="button" data-action="theme" title="${t('theme')}">${state.theme === 'dark' ? '☾' : state.theme === 'light' ? '☀' : '◐'}</button>
    <details class="advanced"><summary>⚙ ${t('advanced')}</summary><div>
      <label><input type="checkbox" data-input="decodeRelay" ${s.decodeRelay ? 'checked' : ''} ${up ? 'disabled' : ''}> ${t('decodeRelay')}</label>
      <button type="button" data-action="reroll" ${up ? 'disabled' : ''}>${t('reroll')} (#${s.ecmpSeed})</button>
    </div></details>`;
  const ta = controls.querySelector<HTMLTextAreaElement>('textarea')!;
  ta.value = state.prompt;
}

/** Input tokens, then (in step-through mode) the dummy output generated so far. */
function tokenStrip(withOutput: boolean): string {
  const step = currentStep();
  const inTok = workload.trackedTokens.slice(0, 24).map((x) => `<span class="tok in">${escapeHtml(x)}</span>`);
  const more = workload.trackedTokens.length > 24 ? `<span class="more">… ${workload.trackedTokens.length} tok</span>` : '';
  const outs = outputTokens(state.lang, Math.min(step.outputSoFar, 40));
  const outTok = outs.map((x, i) => `<span class="tok out ${i === outs.length - 1 && step.outputSoFar > 0 ? 'new' : ''}">${escapeHtml(x)}</span>`);
  const outMore = step.outputSoFar > 40 ? `<span class="more">… ${step.outputSoFar} tok</span>` : '';
  const input = `<span class="kicker">${t('input')}</span>${inTok.join('')}${more}`;
  if (!withOutput) return input;
  return `${input}<span class="kicker">→ ${t('output')} (${t('dummy')})</span>${outTok.join('')}${outMore}`;
}

function renderTransport() {
  const modeBtns = `
    <button type="button" data-mode="steps" aria-pressed="${state.mode === 'steps'}">${t('modeSteps')}</button>
    <button type="button" data-mode="pipeline" aria-pressed="${state.mode === 'pipeline'}">${t('modePipeline')}</button>`;
  if (state.mode === 'pipeline') {
    transport.innerHTML = `<div class="row">${modeBtns}
    <button type="button" data-action="play" aria-pressed="${state.playing}">${state.playing ? '⏸ ' + t('pause') : '▶ ' + t('play')}</button>
    <span class="kicker">${t('slot')} ${state.slot + 1}/${pipeline?.slots.length ?? 0}</span></div>
    <div class="row tokens">${tokenStrip(false)}</div>`;
    return;
  }
  transport.innerHTML = `<div class="row">${modeBtns}
    <button type="button" data-action="first" title="${t('first')}">⏮</button>
    <button type="button" data-action="prev" title="${t('prev')}">◀</button>
    <button type="button" data-action="play" aria-pressed="${state.playing}">${state.playing ? '⏸ ' + t('pause') : '▶ ' + t('play')}</button>
    <button type="button" data-action="next" title="${t('next')}">▶</button>
    <button type="button" data-action="last" title="${t('last')}">⏭</button>
    <span class="kicker">${t('step')} ${state.step + 1}/${plan.steps.length}</span></div>
    <div class="row tokens">${tokenStrip(true)}</div>`;
}

function renderLegend() {
  const sw = (cls: string, dash = '') =>
    `<svg width="26" height="8"><line x1="0" y1="4" x2="26" y2="4" style="stroke:var(${cls});stroke-width:3" ${dash ? `stroke-dasharray="${dash}"` : ''}/></svg>`;
  legend.innerHTML = `
    <span>${sw('--kv', '10 4')} ${tr(PURPOSE_LABEL.kv, state.lang)}</span>
    <span>${sw('--send')} dispatch / A→F</span>
    <span>${sw('--ret', '4 3')} combine / F→A</span>
    <span>${sw('--heat-4')} ${t('linkLoad')}</span>`;
}

function trafficMetrics(traffic: ReturnType<typeof currentStage>[]): string {
  const parts = traffic.flatMap((s) => (s?.traffic ? [s.traffic] : []));
  if (!parts.length) return `<p class="kicker">${t('noTraffic')}</p>`;
  const all = parts.flatMap((x) => x.transfers);
  const sum = totals(all);
  const peak = Math.max(0, ...linkLoad(all).values());
  return `<dl>
      <dt>${t('rdma')}</dt><dd>${formatBytes(sum.rdmaBytes)}</dd>
      <dt>${t('nvlink')}</dt><dd>${formatBytes(sum.nvlinkBytes)}</dd>
      <dt>${t('spine')}</dt><dd>${formatBytes(sum.spineBytes)}</dd>
      <dt>${t('maxLink')}</dt><dd>${formatBytes(peak)}</dd>
    </dl>`;
}

function renderPanel() {
  if (state.pinned?.type === 'block') {
    const { gpu, role, block } = state.pinned;
    const card = currentCards().find((c) => c.gpu === gpu && c.role === role);
    if (card) {
      panel.innerHTML = blockZoomHtml(blockDetail(card, block, state.scenario, plan.roles, workload), gpu, role, state.lang);
      return;
    }
    state.pinned = null;
  }
  if (state.pinned) {
    panel.innerHTML = detailHtml(state.pinned, detailContext());
    return;
  }
  if (state.mode === 'pipeline' && pipeline) {
    const slot = pipeline.slots[state.slot]!;
    const lang = state.lang;
    panel.innerHTML = `
      <div class="kicker">${t('modePipeline')} · ${t('slot')} ${state.slot + 1}/${pipeline.slots.length}</div>
      <h2>${t(plan.roles.kvTransfer ? 'pipelinePd' : 'pipelineNoPd')}</h2>
      <ul class="lanes">${slot.lanes.map((l) => `<li class="lane-${l.lane}">${escapeHtml(l.text[lang])}</li>`).join('')}</ul>
      <h3>${t('metrics')}</h3>
      ${trafficMetrics([slot.stage])}
      <p class="kicker">${escapeHtml(PIPELINE_NOTE[lang])}</p>
      <details><summary>${t('details')}</summary><ul class="notes">${notes().map((n) => `<li>${escapeHtml(n[lang])}</li>`).join('')}</ul></details>`;
    return;
  }
  const step = currentStep();
  const lang = state.lang;
  const traffic = step.stages.flatMap((s) => (s.traffic ? [s.traffic] : []));
  let metrics = `<p class="kicker">${t('noTraffic')}</p>`;
  if (traffic.length) {
    const all = traffic.flatMap((x) => x.transfers);
    const sum = totals(all);
    const tracked = traffic.flatMap((x) => x.flows.filter((f) => f.tracked));
    const sws = tracked.map((f) => f.path.switchCount);
    const relays = tracked.map((f) => f.path.relayCount);
    const range = (v: number[]) => (v.length ? (Math.min(...v) === Math.max(...v) ? `${v[0]}` : `${Math.min(...v)}–${Math.max(...v)}`) : '—');
    const peak = Math.max(0, ...linkLoad(all).values());
    const kv = step.kind === 'kv' ? tracked.reduce((a, f) => a + f.logicalBytes, 0) : 0;
    metrics = `<dl>
      <dt>${t('switches')}</dt><dd>${range(sws)}</dd>
      <dt>${t('relays')}</dt><dd>${range(relays)}</dd>
      ${kv ? `<dt>${t('kvTracked')}</dt><dd>${formatBytes(kv)}</dd>` : ''}
      <dt>${t('rdma')}</dt><dd>${formatBytes(sum.rdmaBytes)}</dd>
      <dt>${t('nvlink')}</dt><dd>${formatBytes(sum.nvlinkBytes)}</dd>
      <dt>${t('spine')}</dt><dd>${formatBytes(sum.spineBytes)}</dd>
      <dt>${t('maxLink')}</dt><dd>${formatBytes(peak)}</dd>
    </dl>${step.kind === 'kv' ? '' : `<p class="kicker">${t('perLayer')}</p>`}`;
  }
  panel.innerHTML = `
    <div class="kicker">${t('step')} ${state.step + 1}/${plan.steps.length} · ${step.phase === 'prefill' ? 'Prefill' : 'Decode'}${step.stages.length > 1 ? ` · ${state.stage + 1}/${step.stages.length}` : ''}</div>
    <h2>${escapeHtml(step.title[lang])}</h2>
    <p>${escapeHtml(step.body[lang])}</p>
    <h3>${t('metrics')}</h3>
    ${metrics}
    <details><summary>${t('details')}</summary><ul class="notes">${notes().map((n) => `<li>${escapeHtml(n[lang])}</li>`).join('')}</ul></details>`;
}

function notes() {
  return notesFor(isScaleUpOnly(state.scenario));
}

function helpHidden(): boolean {
  try {
    return localStorage.getItem('ifv.helpHidden') === '1';
  } catch {
    return false;
  }
}

function renderHelp() {
  const hidden = helpHidden();
  help.classList.toggle('collapsed', hidden);
  // Help strings contain only static <kbd> markup from ui.ts, never user input.
  help.innerHTML = `<b>${t('helpTitle')}</b>
    <span>${t('helpConfig')}</span>
    <span>${t('helpStep')}</span>
    <span>${t('helpInspect')}</span>
    <span>${t('helpModes')}</span>
    <button type="button" data-action="help">${hidden ? t('helpShow') : t('helpHide')}</button>`;
}

function renderAll() {
  document.documentElement.lang = state.lang;
  renderHelp();
  renderControls();
  renderTransport();
  renderLegend();
  renderPanel();
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}

function applyTheme() {
  if (state.theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = state.theme;
}

controls.addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest<HTMLElement>('[data-toggle],[data-action],[data-fabric]');
  if (!btn || (btn as HTMLButtonElement).disabled) return;
  const s = state.scenario;
  if (btn.dataset.fabric) {
    if (btn.dataset.fabric === s.fabric) return;
    s.fabric = btn.dataset.fabric as Scenario['fabric'];
  } else if (btn.dataset.toggle === 'pd') s.pd = !s.pd;
  else if (btn.dataset.toggle === 'af') s.af = !s.af;
  else if (btn.dataset.toggle === 'rail') s.wiring = s.wiring === 'rail' ? 'tor' : 'rail';
  else if (btn.dataset.action === 'reroll') s.ecmpSeed++;
  else if (btn.dataset.action === 'lang') {
    state.lang = state.lang === 'ja' ? 'en' : 'ja';
    try {
      localStorage.setItem('ifv.lang', state.lang);
    } catch {
      /* storage unavailable */
    }
    if (state.preset >= 0) state.prompt = PRESETS[state.preset]!.text[state.lang];
  } else if (btn.dataset.action === 'theme') {
    state.theme = state.theme === 'auto' ? 'dark' : state.theme === 'dark' ? 'light' : 'auto';
    applyTheme();
    renderControls();
    return;
  } else return;
  rebuild();
});

let promptTimer = 0;
controls.addEventListener('input', (e) => {
  const input = e.target as HTMLInputElement;
  const kind = input.dataset.input;
  if (kind === 'preset') {
    state.preset = Number(input.value);
    if (state.preset >= 0) {
      state.prompt = PRESETS[state.preset]!.text[state.lang];
      state.outputTokens = PRESETS[state.preset]!.output;
    }
    rebuild();
  } else if (kind === 'prompt') {
    state.prompt = input.value;
    state.preset = -1;
    clearTimeout(promptTimer);
    promptTimer = window.setTimeout(() => {
      rebuild();
      controls.querySelector<HTMLDetailsElement>('.prompt-editor')!.open = true;
      const ta = controls.querySelector<HTMLTextAreaElement>('textarea')!;
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
    }, 500);
  } else if (kind === 'output') {
    const v = Math.max(1, Math.min(4096, Number(input.value) || 1));
    state.outputTokens = v;
    rebuild();
  } else if (kind === 'decodeRelay') {
    state.scenario.decodeRelay = input.checked;
    rebuild();
  }
});

transport.addEventListener('click', (e) => {
  const modeBtn = (e.target as HTMLElement).closest<HTMLElement>('[data-mode]');
  if (modeBtn) {
    const mode = modeBtn.dataset.mode as State['mode'];
    if (mode !== state.mode) {
      state.mode = mode;
      // The pipeline view is an autoplay view: start it running.
      state.playing = mode === 'pipeline';
      rebuild(false);
    }
    return;
  }
  const btn = (e.target as HTMLElement).closest<HTMLElement>('[data-action]');
  if (!btn) return;
  switch (btn.dataset.action) {
    case 'first':
      pause();
      goto(0);
      break;
    case 'prev':
      pause();
      goto(state.step - 1);
      break;
    case 'next':
      pause();
      stepForward();
      break;
    case 'last':
      pause();
      goto(plan.steps.length - 1);
      break;
    case 'play':
      if (state.playing) pause();
      else if (state.mode === 'pipeline') {
        state.playing = true;
        state.pinned = null;
        showSlot();
      } else {
        state.playing = true;
        state.pinned = null;
        if (state.step === plan.steps.length - 1 && state.stage === currentStep().stages.length - 1) goto(0);
        else advance();
      }
      renderTransport();
      break;
  }
});

help.addEventListener('click', (e) => {
  if (!(e.target as HTMLElement).closest('[data-action="help"]')) return;
  try {
    localStorage.setItem('ifv.helpHidden', helpHidden() ? '0' : '1');
  } catch {
    /* storage unavailable: toggle for this view only */
    help.classList.toggle('collapsed');
    return;
  }
  renderHelp();
});

panel.addEventListener('click', (e) => {
  if ((e.target as HTMLElement).closest('[data-action="unpin"]')) {
    state.pinned = null;
    applyHighlight();
    renderPanel();
  }
});

svg.addEventListener('click', (e) => {
  if (e.target === svg && state.pinned) {
    state.pinned = null;
    applyHighlight();
    renderPanel();
  }
});

document.addEventListener('keydown', (e) => {
  const focused = e.target as Element;
  if (focused.closest('input, textarea, select, button, summary, [tabindex]') && e.key !== 'Escape') return;
  if (e.key === 'Escape' && state.pinned) {
    state.pinned = null;
    applyHighlight();
    renderPanel();
  } else if (e.key === 'ArrowRight' && state.mode === 'steps') {
    pause();
    stepForward();
  } else if (e.key === 'ArrowLeft' && state.mode === 'steps') {
    pause();
    goto(state.step - 1);
  } else if (e.key === ' ') {
    e.preventDefault();
    transport.querySelector<HTMLElement>('[data-action="play"]')!.click();
  }
});

applyTheme();
rebuild(false);
