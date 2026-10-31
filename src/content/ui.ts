// UI chrome strings, presets and notes (ADR §8, §9, §10, §19).

import type { Activity } from '../model/steps';
import type { RoleLabel } from '../model/scenario';
import type { Method, Purpose } from '../model/routing';
import type { I18n, Lang } from './text';

const t = (ja: string, en: string): I18n => ({ ja, en });

export const UI = {
  title: t('推論ファブリックの可視化', 'Inference Fabric Visualizer'),
  pd: t('PD 分離', 'PD disagg.'),
  af: t('AF 分離', 'AF disagg.'),
  rail: t('レール最適化', 'Rail-optimized'),
  prompt: t('プロンプト', 'Prompt'),
  outputTokens: t('出力トークン数', 'Output tokens'),
  advanced: t('詳細設定', 'Advanced'),
  decodeRelay: t('Decode も Scale-up 中継する', 'Relay Decode via scale-up'),
  reroll: t('ECMP ハッシュを再抽選', 'Re-roll ECMP hash'),
  theme: t('テーマ', 'Theme'),
  first: t('最初へ', 'First'),
  prev: t('前へ', 'Previous'),
  next: t('次へ', 'Next'),
  last: t('最後へ', 'Last'),
  play: t('自動再生', 'Autoplay'),
  pause: t('一時停止', 'Pause'),
  replay: t('段階を再生', 'Replay stage'),
  step: t('ステップ', 'Step'),
  metrics: t('このステップの通信', 'Traffic in this step'),
  trackedPath: t('追跡中リクエストの経路', 'Tracked request path'),
  switches: t('Scale-out スイッチ通過数', 'Scale-out switches crossed'),
  relays: t('Scale-up 中継', 'Scale-up relays'),
  rdma: t('RDMA（ネットワーク）', 'RDMA (network)'),
  nvlink: t('Scale-up（ノード内）', 'Scale-up (in-node)'),
  spine: t('Spine を通るデータ量', 'Bytes through spines'),
  maxLink: t('最大リンク負荷（片方向）', 'Peak link load (one direction)'),
  kvTracked: t('追跡中リクエストの KV（論理量）', 'Tracked request KV (logical)'),
  perLayer: t('代表 1 層あたり', 'per representative layer'),
  noTraffic: t('このステップはネットワーク通信なし（GPU 内の処理）', 'No network traffic in this step (on-GPU work)'),
  input: t('入力', 'Input'),
  output: t('出力', 'Output'),
  dummy: t('ダミー文', 'dummy text'),
  legend: t('凡例', 'Legend'),
  fabricLabel: t('ネットワーク', 'Network'),
  scaleUpOnly: t('Scale-up のみ', 'Scale-up only'),
  scaleOut: t('Scale-up + Scale-out', 'Scale-up + Scale-out'),
  scaleUpDisabled: t('Scale-up のみでは 1 ノードにモデル全体が収まるため使いません', 'Not used: in scale-up only mode one node holds the whole model'),
  model: t('基準モデル', 'Reference model'),
  localExperts: t('この GPU の Expert', 'Experts on this GPU'),
  selected: t('個を選択中', 'selected'),
  noLocalExperts: t('Expert は持たない（F 側へ送る）', 'No experts here (sent to the F side)'),
  weights: t('重み', 'weights'),
  hbmTitle: t('HBM 使用量（この構成内の最大＝バー全幅）', 'HBM in use (full bar = largest in this setup)'),
  activations: t('activation', 'activations'),
  cardsNote: t('代表 GPU の内部（HBM はこの構成内の最大を 100% とした相対表示、概算）', 'Inside representative GPUs (HBM bar relative to the largest in this setup; estimates)'),
  helpTitle: t('操作方法', 'How to use'),
  helpConfig: t('上のボタンで Scale-up のみ / Scale-out あり、PD 分離・AF 分離・レール最適化を切り替え（⚙ に詳細設定）', 'Switch scale-up only / with scale-out, PD / AF / rail-optimized with the buttons above (⚙ for advanced)'),
  helpStep: t('<kbd>◀</kbd><kbd>▶</kbd> または <kbd>←</kbd><kbd>→</kbd> でステップ送り、<kbd>Space</kbd> で自動再生 / 一時停止', '<kbd>◀</kbd><kbd>▶</kbd> or <kbd>←</kbd><kbd>→</kbd> to step, <kbd>Space</kbd> to play / pause'),
  helpInspect: t('GPU・スイッチ・リンク・粒にマウスで概要、クリックで詳細を固定（<kbd>Esc</kbd> で解除）', 'Hover GPUs, switches, links or particles for a summary; click to pin details (<kbd>Esc</kbd> to unpin)'),
  helpModes: t('下の「並行動作」で複数バッチの同時進行を自動再生', '“Overlap” below autoplays several batches at once'),
  helpHide: t('隠す', 'Hide'),
  helpShow: t('操作方法を表示', 'Show help'),
  modeSteps: t('ステップ説明', 'Step-through'),
  modePipeline: t('並行動作（自動再生）', 'Overlap (autoplay)'),
  slot: t('スロット', 'Slot'),
  pipelinePd: t('PD 分離: Prefill・KV 転送・Decode が同時に進む', 'PD disaggregation: Prefill, KV transfer and Decode run at once'),
  pipelineNoPd: t('PD 分離なし: 同じ GPU で Prefill と Decode が交代する', 'No PD disaggregation: Prefill and Decode take turns on the same GPUs'),
  heatNote: t('色は方向ごとの転送量のうち大きい方を、このステップ内の最大値に対する相対値で示します', 'Color shows the larger of the two directions, relative to the busiest link in this stage'),
  linkLoad: t('リンク負荷（このステップの転送量、模式・相対）', 'Link load (bytes this step, schematic, relative)'),
  close: t('閉じる', 'Close'),
  details: t('詳細', 'Details'),
  tokensApprox: t('トークン数（模式的な推定）', 'tokens (schematic estimate)'),
  idle: t('待機', 'idle'),
  tracked: t('追跡中', 'tracked'),
  batch: t('バッチの他リクエスト', 'rest of batch'),
};

/** Vendor-neutral naming (ADR §4). */
export const TERMS_NOTE: I18n = t(
  '「Scale-up リンク」「Scale-up Switch」は、NVLink / NVSwitch、UALink、SUE（Scale-Up Ethernet）などの総称です。DeepEP や NCCL PXN の中継は、NVIDIA の環境では NVLink を使います。',
  '“Scale-up link” and “scale-up switch” stand for NVLink / NVSwitch, UALink, SUE (Scale-Up Ethernet) and similar. DeepEP and NCCL PXN relays use NVLink on NVIDIA systems.',
);

export const NOTES: I18n[] = [
  t(
    '32 GPU に縮小した模式図です。並列化も DeepSeek-V3 の実運用（Prefill: TP4×DP8・EP32、Decode: TP4×DP80・EP320）とは異なる模式配置で、この配置でモデルが収まることは保証しません。ただし V3 は FP8 公開のため、routed experts は約 0.65 TB と H200 / B200 世代の 8 GPU に収まる規模です。',
    'A schematic scaled down to 32 GPUs, with a simplified layout that differs from DeepSeek-V3 production (Prefill TP4×DP8/EP32, Decode TP4×DP80/EP320); model fit is not guaranteed. V3 ships in FP8, so routed experts are about 0.65 TB, within reach of 8 H200/B200-class GPUs.',
  ),
  t(
    'データ量は DeepSeek-V3 の公開 config に基づく概算です（KV: BF16・61 層×576、送信: FP8、返送: BF16、scale 等は除く）。時間・性能は表示しません。',
    'Volumes are estimates from the public DeepSeek-V3 config (KV: BF16, 61 layers × 576; send FP8; return BF16; scales excluded). No time or performance figures are shown.',
  ),
  t(
    'ECMP は RoCEv2・フローごとの静的ハッシュ・adaptive routing なし、QP は NIC の組と用途ごとに 1 本という説明用の仮定です。偏りはハッシュの一例で、トポロジの必然ではありません。',
    'ECMP assumes RoCEv2, static per-flow hashing, no adaptive routing, and one QP per NIC pair and purpose (a teaching assumption). Imbalance is one hash outcome, not a property of the topology.',
  ),
  t(
    'トークン分割は模式的で、出力文はダミーです。group-limited routing と MTP は再現していません。',
    'Tokenization is schematic and the output text is a dummy. Group-limited routing and MTP are not modeled.',
  ),
];

export const SCALEUP_NOTE: I18n = t(
  'Scale-up のみでは、基準モデルを Qwen3-30B-A3B 相当の軽量 MoE（hidden 2048、48 層、128 Expert 中 8 個、GQA の KV 4 ヘッド×128、BF16 で約 98 KB／トークン）に切り替えます。BF16 でも約 61 GB と 1 ノードに十分収まるため、4 ノードは独立したレプリカ（ノード間は DP）として動き、Scale-out のネットワークは使いません。',
  'In scale-up only mode the reference model switches to a lightweight MoE like Qwen3-30B-A3B (hidden 2048, 48 layers, top-8 of 128 experts, GQA KV with 4 heads × 128, about 98 KB per token in BF16). At about 61 GB in BF16 it fits easily in one node, so the four nodes run as independent replicas (DP across nodes) and the scale-out network is unused.',
);

const SCALEUP_EXTRA: I18n[] = [
  t(
    'データ量は Qwen3-30B-A3B の公開 config に基づく概算です（KV: BF16・48 層 × K/V × 4 ヘッド × 128、送信: FP8、返送: BF16、scale 等は除く）。時間・性能は表示しません。',
    'Volumes are estimates from the public Qwen3-30B-A3B config (KV: BF16, 48 layers × K/V × 4 heads × 128; send FP8; return BF16; scales excluded). No time or performance figures are shown.',
  ),
  t('トークン分割は模式的で、出力文はダミーです。', 'Tokenization is schematic and the output text is a dummy.'),
];

/** Notes for the active reference model; scale-up only must not mention DeepSeek-V3 dimensions. */
export function notesFor(scaleUp: boolean): I18n[] {
  return scaleUp ? [SCALEUP_NOTE, ...SCALEUP_EXTRA, TERMS_NOTE] : [...NOTES, TERMS_NOTE];
}

export interface Preset {
  name: I18n;
  text: I18n;
  output: number;
}

export const PRESETS: Preset[] = [
  {
    name: t('短い質問', 'Short question'),
    text: t('PD 分離とは何ですか？', 'What is PD disaggregation?'),
    output: 32,
  },
  {
    name: t('中くらいの依頼', 'Medium request'),
    text: t(
      'MoE モデルの推論で、Expert Parallelism の All-to-All 通信がネットワークにどんな負荷をかけるのか、Scale-up と Scale-out の違いに触れながら、エンジニア向けに 3 段落で説明してください。',
      'Explain to engineers, in three paragraphs, how the all-to-all traffic of expert parallelism in MoE inference loads the network, touching on the difference between scale-up and scale-out.',
    ),
    output: 256,
  },
  {
    name: t('長い文書の要約', 'Summarize a long document'),
    text: t(
      '次の設計文書を読み、論点と未決事項を箇条書きで要約してください。'.concat('（設計文書の本文がここに続きます。）'.repeat(60)),
      'Read the following design document and summarize the issues and open questions as bullet points. '.concat('(The body of the design document continues here.) '.repeat(60)),
    ),
    output: 512,
  },
];

export const PURPOSE_LABEL: Record<Purpose, I18n> = {
  kv: t('KV 転送', 'KV transfer'),
  dispatch: t('dispatch', 'dispatch'),
  combine: t('combine', 'combine'),
  a2f: t('A→F', 'A→F'),
  f2a: t('F→A', 'F→A'),
};

export const METHOD_LABEL: Record<Method, I18n> = {
  direct: t('direct', 'direct'),
  relayRecv: t('Scale-up 中継（受信側で配る）', 'Scale-up relay (fan out at receiver)'),
  relaySendAggregate: t('Scale-up 中継（送信側で集約）', 'Scale-up relay (aggregate at sender)'),
};

export const ROLE_LABEL: Record<RoleLabel, I18n> = {
  unified: t('統合（Prefill+Decode）', 'Unified (Prefill+Decode)'),
  prefill: t('Prefill', 'Prefill'),
  decode: t('Decode', 'Decode'),
  attention: t('Attention（A）', 'Attention (A)'),
  expert: t('Expert（F）', 'Expert (F)'),
  'decode-attention': t('Decode-Attention（A）', 'Decode-Attention (A)'),
  'decode-expert': t('Decode-Expert（F）', 'Decode-Expert (F)'),
};

export const ACTIVITY_LABEL: Record<Activity, I18n> = {
  scheduler: t('割り当て', 'scheduled'),
  attention: t('Attention', 'attention'),
  dense: t('dense FFN', 'dense FFN'),
  router: t('router → 送信', 'router → send'),
  experts: t('Expert 計算 → 返送', 'experts → return'),
  aggregate: t('合算 ＋ residual', 'combine + residual'),
  head: t('出力 head ＋ sampling', 'output head + sampling'),
};

export function tr(text: I18n, lang: Lang): string {
  return text[lang];
}

export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v >= 100 || i === 0 ? Math.round(v) : v.toFixed(1)} ${units[i]}`;
}
