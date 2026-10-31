// Step explanations in Japanese and English (ADR §10). Labels such as
// Prefill, Decode, Leaf, Spine, dispatch and KV Cache stay in English.

export type Lang = 'ja' | 'en';
export type I18n = Record<Lang, string>;

export interface StepTextContext {
  pd: boolean;
  af: boolean;
  /** Prefill MoE uses A->F / F->A (AF without PD). */
  prefillAf: boolean;
  decodeAf: boolean;
  decodeRelay: boolean;
  wiring: 'rail' | 'tor';
  /** Lightweight model, every node an independent replica; nothing leaves the scale-up domain. */
  scaleUp: boolean;
  modelName: string;
  layers: number;
  moeLayers: number;
  sharedExpert: boolean;
  iteration?: number;
}

export type StepKind =
  | 'arrive'
  | 'prefill-dense'
  | 'prefill-moe'
  | 'prefill-done'
  | 'kv'
  | 'decode-attn'
  | 'decode-send'
  | 'decode-return'
  | 'decode-done'
  | 'repeat';

const t = (ja: string, en: string): I18n => ({ ja, en });

export function stepTitle(kind: StepKind, c: StepTextContext): I18n {
  const it = c.iteration !== undefined ? ` #${c.iteration + 1}` : '';
  switch (kind) {
    case 'arrive':
      return t('バッチ到着と割り当て', 'Batch arrives and is scheduled');
    case 'prefill-dense':
      return t('Prefill: 先頭 3 層（dense）', 'Prefill: first 3 dense layers');
    case 'prefill-moe':
      return c.prefillAf
        ? t('Prefill: MoE 層（A→F → F→A）', 'Prefill: MoE layer (A→F → F→A)')
        : t('Prefill: MoE 層（dispatch → combine）', 'Prefill: MoE layer (dispatch → combine)');
    case 'prefill-done':
      return t('Prefill 完了: 最初のトークン', 'Prefill done: first token');
    case 'kv':
      return t('KV Cache 転送（Prefill → Decode）', 'KV Cache transfer (Prefill → Decode)');
    case 'decode-attn':
      return t(`Decode${it}: Attention と router`, `Decode${it}: attention and router`);
    case 'decode-send':
      return c.decodeAf ? t(`Decode${it}: A→F`, `Decode${it}: A→F`) : t(`Decode${it}: dispatch`, `Decode${it}: dispatch`);
    case 'decode-return':
      return c.decodeAf ? t(`Decode${it}: F→A と合算`, `Decode${it}: F→A and combine`) : t(`Decode${it}: combine`, `Decode${it}: combine`);
    case 'decode-done':
      return t(`Decode${it}: 次のトークン`, `Decode${it}: next token`);
    case 'repeat':
      return t('以下、繰り返し', 'And so on');
  }
}

export function stepBody(kind: StepKind, c: StepTextContext): I18n {
  switch (kind) {
    case 'arrive':
      if (c.scaleUp) {
        return t(
          `${c.modelName} 相当の軽量 MoE は 1 ノード（8 GPU）に収まるので、4 ノードはそれぞれ独立したレプリカとして動きます。スケジューラはバッチの各リクエストを GPU に割り当てます（DP attention）。強調色が追跡中のリクエストです。`,
          `A lightweight MoE like ${c.modelName} fits in one node (8 GPUs), so the four nodes run as independent replicas. The scheduler assigns each request in the batch to a GPU (DP attention). The highlighted request is the one we follow.`,
        );
      }
      return c.pd
        ? t(
            'スケジューラがバッチの各リクエストを Prefill 側の GPU に割り当てます。DP attention なので、1 リクエストの Attention と KV は 1 GPU が担当します。同時に、Decode 側の GPU も予約します（同じレール＝同じ GPU 番号を優先）。強調色が追跡中のリクエストです。',
            'The scheduler assigns each request in the batch to a Prefill GPU. With DP attention, one GPU owns a request’s attention and KV. It also reserves a Decode GPU now, preferring the same rail (same local GPU index). The highlighted request is the one we follow.',
          )
        : t(
            'スケジューラがバッチの各リクエストを GPU に割り当てます。DP attention なので、1 リクエストの Attention と KV は 1 GPU が担当します。強調色が追跡中のリクエストです。',
            'The scheduler assigns each request in the batch to a GPU. With DP attention, one GPU owns a request’s attention and KV. The highlighted request is the one we follow.',
          );
    case 'prefill-dense':
      return t(
        '入力トークンをまとめて処理します。先頭 3 層は MoE ではない dense FFN なので、各 GPU の中で完結し、ネットワーク通信は発生しません。',
        'All prompt tokens are processed at once. The first 3 layers use a dense FFN, so each GPU computes locally and no network traffic is needed.',
      );
    case 'prefill-moe':
      if (c.scaleUp) {
        return t(
          `router が各トークンの Expert を 8 個選び、同じノードの中でそれらを持つ GPU へ Scale-up リンクで送ります（dispatch）。Expert の計算後、GPU ごとの重み付き部分和を返します（combine）。Expert 一式が各ノードに揃っているので、通信はノード内の Scale-up Switch で完結し、Leaf-Spine は使いません。表示は代表の 1 層で、実際は ${c.moeLayers} 層を順に処理します。`,
          `The router picks 8 experts per token and sends tokens over scale-up links to the GPUs in the same node that own them (dispatch). After the experts run, each GPU returns one weighted partial sum (combine). Every node holds a full expert set, so traffic stays inside the node's scale-up switch and the leaf-spine is unused. One representative layer is shown; all ${c.moeLayers} layers run in turn.`,
        );
      }
      return c.prefillAf
        ? t(
            `Attention 側の router が各トークンの Expert を 8 個選び、それらを持つ FFN 側の GPU へ直接送ります（A→F、M:N）。FFN 側は GPU ごとに重み付きの部分和を作って返し（F→A）、Attention 側が shared expert と合わせて合算します。Prefill はスループット重視なので Scale-up 中継（DeepEP Normal）を使います。表示は代表の 1 層で、実際は MoE 層 ${c.moeLayers} 層を順に処理します。`,
            `The attention-side router picks 8 experts per token and sends tokens straight to the FFN GPUs that own them (A→F, M:N). Each FFN GPU returns one weighted partial sum (F→A); the attention GPU adds them to the shared expert output. Prefill favors throughput, so traffic is relayed over scale-up links (DeepEP Normal). One representative layer is shown; all ${c.moeLayers} MoE layers run in turn.`,
          )
        : t(
            `router が各トークンの Expert を 8 個選び、それらを持つ GPU へ送ります（dispatch）。Expert の計算後、GPU ごとの重み付き部分和を元の GPU へ返します（combine）。Prefill はスループット重視なので Scale-up 中継（DeepEP Normal）を使い、dispatch は受信ノード内で配り、combine は送信ノード内で集約してから送ります。表示は代表の 1 層で、実際は MoE 層 ${c.moeLayers} 層を順に処理します。`,
            `The router picks 8 experts per token and sends tokens to the GPUs that own them (dispatch). After the experts run, each GPU returns one weighted partial sum (combine). Prefill favors throughput, so traffic is relayed over scale-up links (DeepEP Normal): dispatch fans out inside the receiving node, combine aggregates inside the sending node. One representative layer is shown; all ${c.moeLayers} MoE layers run in turn.`,
          );
    case 'prefill-done':
      return t(
        `全 ${c.layers} 層を処理し終えると、出力 head と sampling で最初のトークンが決まります。最初のトークンは Prefill から生まれます。`,
        `After all ${c.layers} layers, the output head and sampling choose the first token. The first token comes out of Prefill.`,
      );
    case 'kv':
      return c.wiring === 'rail'
        ? t(
            'Prefill で作った KV Cache を、予約済みの Decode GPU へ RDMA で 1 対 1 に送ります（NIXL など）。同じレール（同じ GPU 番号）の GPU を予約しているので、スイッチは Leaf 1 台で済みます。この画面ではまとめて送っていますが、実際は層ごとに送り、Prefill の計算と重ねます。',
            'The KV Cache built during Prefill moves point-to-point over RDMA to the reserved Decode GPU (e.g. NIXL). The reserved GPU is on the same rail (same local index), so only one leaf is crossed. Shown here as one transfer; in practice KV is sent layer by layer, overlapped with Prefill compute.',
          )
        : t(
            'Prefill で作った KV Cache を、予約済みの Decode GPU へ RDMA で 1 対 1 に送ります（NIXL など）。ToR 型では Leaf がノードごとに分かれているので、ノード間の転送は必ず Leaf → Spine → Leaf を通ります。この画面ではまとめて送っていますが、実際は層ごとに送り、Prefill の計算と重ねます。',
            'The KV Cache built during Prefill moves point-to-point over RDMA to the reserved Decode GPU (e.g. NIXL). With ToR wiring each node has its own leaf, so any inter-node transfer crosses leaf → spine → leaf. Shown here as one transfer; in practice KV is sent layer by layer, overlapped with Prefill compute.',
          );
    case 'decode-attn':
      return t(
        '直前のトークンを入力に、Attention が KV Cache を読み、router が Expert を選びます。ここは GPU 内の処理です。生成は 1 トークンずつで、MTP（投機的デコード）は扱いません。',
        'Using the previous token as input, attention reads the KV Cache and the router picks experts. This happens inside the GPU. Tokens are generated one at a time; MTP (speculative decoding) is not shown.',
      );
    case 'decode-send':
      if (c.scaleUp) {
        return t(
          '選ばれた Expert を持つ同じノード内の GPU へ、トークンを Scale-up リンクで送ります。ノード間の通信はなく、Leaf-Spine は使いません。',
          'Tokens go over scale-up links to the GPUs in the same node that own the selected experts. No traffic crosses nodes, and the leaf-spine is unused.',
        );
      }
      if (c.decodeRelay) {
        return c.wiring === 'rail'
          ? t(
              '詳細設定で「Decode も Scale-up 中継する」が ON です。宛先ノードの同じレールの GPU へ RDMA で送り、そのノード内の Scale-up リンクで Expert を持つ GPU へ配ります。レール最適化ありなので Spine を通りません。そのぶん NVLink の中継が 1 段増えます。',
              '“Relay Decode via scale-up” is on. Tokens go over RDMA to the same-rail GPU on the destination node, then fan out over scale-up links to the expert GPUs. With rail-optimized wiring this avoids the spine, at the cost of one scale-up relay hop.',
            )
          : t(
              '詳細設定で「Decode も Scale-up 中継する」が ON です。宛先ノードへは RDMA で 1 回だけ送り、そのノード内の Scale-up リンクで配ります。ToR 型ではノード間の通信は Spine を避けられませんが、同じノードへの重複した送信がなくなるので、Spine を通るデータ量は減ります。',
              '“Relay Decode via scale-up” is on. Each destination node receives one RDMA transfer, then fans out over scale-up links. With ToR wiring inter-node traffic still crosses a spine, but duplicate sends to the same node disappear, so spine bytes drop.',
            );
      }
      return c.wiring === 'rail'
        ? t(
            '選ばれた Expert を持つ GPU へトークンを直接送ります。Decode はレイテンシ重視なので中継しません（DeepEP Low-Latency）。そのため、レール最適化ありでも、レールをまたぐ送信は Spine を通ります。',
            'Tokens go directly to the GPUs owning the selected experts. Decode favors latency, so there is no relay (DeepEP Low-Latency). Even with rail-optimized wiring, cross-rail traffic goes through a spine.',
          )
        : t(
            '選ばれた Expert を持つ GPU へトークンを直接送ります。Decode はレイテンシ重視なので中継しません（DeepEP Low-Latency）。ToR 型では、ノード間の送信はすべて Spine を通ります。',
            'Tokens go directly to the GPUs owning the selected experts. Decode favors latency, so there is no relay (DeepEP Low-Latency). With ToR wiring, every inter-node send goes through a spine.',
          );
    case 'decode-return':
      return t(
        c.sharedExpert
          ? 'Expert 側は GPU ごとに重み付きの部分和を作って返し、元の GPU が shared expert と合わせて合算し、residual を足します（代表の 1 層を表示）。'
          : 'Expert 側は GPU ごとに重み付きの部分和を作って返し、元の GPU が合算して residual を足します（代表の 1 層を表示）。',
        c.sharedExpert
          ? 'Each expert GPU returns one weighted partial sum; the original GPU adds it to the shared expert output and the residual (one representative layer shown).'
          : 'Each expert GPU returns one weighted partial sum; the original GPU adds them up with the residual (one representative layer shown).',
      );
    case 'decode-done':
      return t(
        '全層を処理し終えると、出力 head と sampling で次のトークンが決まります。',
        'After all layers, the output head and sampling choose the next token.',
      );
    case 'repeat':
      return t(
        '指定した出力トークン数に達するまで、Decode の 4 ステップを繰り返します。',
        'The four Decode steps repeat until the requested number of output tokens is reached.',
      );
  }
}

/** Dummy output text; the page never runs a real model (ADR §9). */
export const DUMMY_OUTPUT: I18n = t(
  'Prefill は入力全体を一度に処理し、Decode は 1 トークンずつ生成します。そのため、両者を別の GPU に分けると、それぞれに合った資源で並行に処理できます。',
  'Prefill processes the whole prompt at once while Decode generates one token at a time, so splitting them lets each run in parallel on hardware suited to it.',
);
