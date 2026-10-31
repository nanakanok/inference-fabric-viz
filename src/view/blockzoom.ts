// Right-panel rendering of the block zoom (ADR §23).

import type { Lang } from '../content/text';
import { formatBytes } from '../content/ui';
import type { BlockDetail, RoutedExpert, Scope } from '../model/blockzoom';
import type { BlockId, CardRole } from '../model/gpuview';

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const J = (lang: Lang, ja: string, en: string) => (lang === 'ja' ? ja : en);

export const BLOCK_NAME: Record<BlockId, [string, string]> = {
  embed: ['Embedding', 'Embedding'],
  attention: ['Attention', 'Attention'],
  kv: ['KV Cache', 'KV Cache'],
  dense: ['Dense FFN', 'Dense FFN'],
  router: ['Router（top-k 選択）', 'Router (top-k)'],
  experts: ['Routed experts', 'Routed experts'],
  combine: ['Combine ＋ residual', 'Combine + residual'],
  head: ['出力 head ＋ sampling', 'Output head + sampling'],
  send: ['送信', 'Send'],
  recv: ['受信', 'Receive'],
};

function whereLabel(e: RoutedExpert, lang: Lang): string {
  if (e.where === 'here') return J(lang, 'この GPU', 'this GPU');
  if (e.where === 'same-node') return J(lang, `同じノード（${e.owner}）`, `same node (${e.owner})`);
  return J(lang, `別ノード（${e.owner}）`, `other node (${e.owner})`);
}

function scopeLabel(s: Scope, lang: Lang): string {
  if (s.tokens > 1) return J(lang, `対象: プロンプトの先頭 ${s.tokens} トークン（カードの点灯と同じ）`, `Scope: first ${s.tokens} prompt tokens (same as the card highlights)`);
  if (s.phase === 'prefill') return J(lang, '対象: プロンプトの先頭トークン（Prefill では全トークンが同時に流れます）', 'Scope: the first prompt token (in Prefill every token flows at once)');
  return J(lang, `対象: Decode の ${s.tokenIdx + 1} 番目のトークン`, `Scope: decode token ${s.tokenIdx + 1}`);
}

function bar(weight: number, max: number, cls: string): string {
  return `<span class="zbar"><span class="${cls}" style="width:${Math.max(2, (weight / max) * 100).toFixed(1)}%"></span></span>`;
}

export function blockZoomHtml(d: BlockDetail, gpu: string, role: CardRole, lang: Lang): string {
  const [ja, en] = BLOCK_NAME[d.block];
  const head = `<button type="button" data-action="unpin">← ${J(lang, '閉じる', 'Close')}</button>
    <div class="kicker">${esc(role)} · ${esc(gpu)}</div><h2>${esc(J(lang, ja, en))}</h2>`;
  const note = `<p class="kicker">${J(lang, '模式図です。数値は公開 config と決定的なワークロードからの概算で、実モデルは動かしていません。', 'Schematic. Numbers come from the public config and the deterministic workload; no model is run.')}</p>`;
  if ('idle' in d) return `${head}<p>${J(lang, 'この段階では、追跡中のリクエストはこの GPU を使っていません。', 'The tracked request does not use this GPU at this point.')}</p>`;

  switch (d.block) {
    case 'router': {
      const max = Math.max(...d.experts.map((e) => e.weight), 1e-9);
      return `${head}<p class="kicker">${esc(scopeLabel(d.scope, lang))}</p>
        <p>${J(
          lang,
          `トークンの hidden state にゲートの重みを掛けて ${d.totalExperts} 個の Expert それぞれのスコアを出し、上位 ${d.topK} 個を選んで重みを正規化します。選ばれた Expert を持つ GPU にトークンを送ります。`,
          `The token's hidden state is scored against all ${d.totalExperts} experts by the gate; the top ${d.topK} are kept and their weights renormalized. The token is then sent to the GPUs that own them.`,
        )}</p>
        <table class="ztable"><thead><tr><th>Expert</th><th>${J(lang, '重み', 'Weight')}</th><th>${J(lang, '場所', 'Where')}</th></tr></thead><tbody>
        ${d.experts.map((e) => `<tr class="w-${e.where}"><td>#${e.expert}</td><td>${bar(e.weight, max, 'zfill')} ${(e.weight * 100).toFixed(1)}%</td><td>${esc(whereLabel(e, lang))}</td></tr>`).join('')}
        </tbody></table>${note}`;
    }
    case 'experts': {
      const rows = d.experts.length
        ? d.experts
            .map(
              (e) => `<li><b>Expert #${e.expert}</b>（${d.scope.tokens > 1 ? J(lang, `${e.tokens} トークンが選択、平均の重み`, `picked by ${e.tokens} tokens, mean weight`) : J(lang, '重み', 'weight')} ${(e.weight * 100).toFixed(1)}%）
                <div class="ffn"><span>x [${d.hidden}]</span><span>→ gate / up [${d.inter}]</span><span>→ SiLU ⊙</span><span>→ down [${d.hidden}]</span><span>× ${(e.weight * 100).toFixed(1)}%</span></div></li>`,
            )
            .join('')
        : `<li>${J(lang, 'このトークンが選んだ Expert は、この GPU にはありません。', 'None of the experts this token picked live on this GPU.')}</li>`;
      return `${head}<p class="kicker">${esc(scopeLabel(d.scope, lang))}</p>
        <p>${J(
          lang,
          `この GPU は ${d.hostedCount} 個の Expert を持ちます。届いたトークンについて、選ばれた Expert だけが FFN（gate と up の射影 → SiLU で掛け合わせ → down の射影）を並列に計算し、router の重みを掛けます。`,
          `This GPU hosts ${d.hostedCount} experts. For each arriving token only the selected experts run their FFN (gate and up projections → SiLU product → down projection) in parallel, scaled by the router weight.`,
        )}</p><ul class="zlist">${rows}</ul>${note}`;
    }
    case 'combine': {
      const max = d.partials[0]?.weight ?? 1;
      return `${head}<p class="kicker">${esc(scopeLabel(d.scope, lang))}</p>
        <p>${J(
          lang,
          `Expert を持つ GPU ごとに重み付きの部分和が 1 本ずつ返ってきます。それらを足し合わせ${d.sharedExpert ? '、shared expert の出力を加え' : ''}、最後に residual（層の入力）を足して次の層へ渡します。`,
          `Each expert GPU returns one weighted partial sum. They are added together${d.sharedExpert ? ', the shared expert output is added' : ''}, and finally the residual (the layer input) is added before the next layer.`,
        )}</p>
        <table class="ztable"><thead><tr><th>${J(lang, '部分和の送り元', 'Partial sum from')}</th><th>Experts</th><th>${J(lang, '重みの合計', 'Weight')}</th></tr></thead><tbody>
        ${d.partials.map((p) => `<tr><td>${esc(p.owner)}</td><td>${p.experts.map((e) => `#${e}`).join(' ')}</td><td>${bar(p.weight, max, 'zfill')} ${(p.weight * 100).toFixed(1)}%</td></tr>`).join('')}
        ${d.sharedExpert ? `<tr><td>${J(lang, 'この GPU', 'this GPU')}</td><td>shared expert</td><td>${J(lang, '常に加算', 'always added')}</td></tr>` : ''}
        <tr><td colspan="2">residual</td><td>${J(lang, '常に加算', 'always added')}</td></tr>
        </tbody></table>${note}`;
    }
    case 'attention':
    case 'kv': {
      const style =
        d.style === 'mla'
          ? J(
              lang,
              'MLA（Multi-head Latent Attention）: KV を 512 次元の latent と 64 次元の RoPE key に圧縮して保存し、計算時にヘッドごとに展開します。',
              'MLA (multi-head latent attention): KV is stored compressed as a 512-dim latent plus a 64-dim RoPE key and expanded per head at compute time.',
            )
          : J(lang, 'GQA: 32 個の Query ヘッドが 4 組の K / V（各 128 次元）を共有します。', 'GQA: 32 query heads share 4 K/V heads of 128 dims each.');
      const readPerLayer = d.contextTokens * d.kvBytesPerTokenPerLayer;
      return `${head}
        <p>${style}</p>
        <p>${J(
          lang,
          'Decode では、新しいトークンの Query が、これまでの全トークンの K / V を読み、softmax で重み付けして混ぜます。読み込む量は文脈の長さに比例し、Decode がメモリ帯域に縛られる主な理由です。',
          'In Decode the new token’s query reads the K/V of every earlier token and mixes them with softmax weights. The amount read grows with context length, which is why Decode is bound by memory bandwidth.',
        )}</p>
        <dl>
          <dt>${J(lang, '文脈の長さ', 'Context length')}</dt><dd>${d.contextTokens} tok</dd>
          <dt>${J(lang, '1 トークン・1 層あたりの KV', 'KV per token per layer')}</dt><dd>${formatBytes(d.kvBytesPerTokenPerLayer)}</dd>
          <dt>${J(lang, '1 層で読む KV', 'KV read per layer')}</dt><dd>${formatBytes(readPerLayer)}</dd>
          <dt>${J(lang, `全 ${d.layers} 層で読む KV`, `KV read over ${d.layers} layers`)}</dt><dd>${formatBytes(readPerLayer * d.layers)}</dd>
          <dt>${J(lang, 'この GPU の KV（担当リクエスト全体）', 'KV on this GPU (all its requests)')}</dt><dd>${formatBytes(d.kvOnGpu)}</dd>
        </dl>${note}`;
    }
    default: {
      const text: Record<string, [string, string]> = {
        embed: ['トークン ID を、埋め込み表から hidden 次元のベクトルに変換します。', 'Token IDs are looked up in the embedding table to get hidden-size vectors.'],
        dense: [
          `先頭 ${d.denseLayers} 層は MoE ではない通常の FFN で、この GPU の中だけで計算します。`,
          `The first ${d.denseLayers} layers use an ordinary FFN computed entirely on this GPU.`,
        ],
        head: ['最終層の出力を語彙サイズに射影し、確率分布から次のトークンを 1 つ選びます（sampling）。', 'The final hidden state is projected to the vocabulary and one next token is sampled from the distribution.'],
        send: ['トークン（送る側）または部分和（返す側）を、相手の GPU へ送ります。経路はトポロジ上の粒で表示しています。', 'Tokens (send side) or partial sums (return side) go to the peer GPU; the path is shown by the particles above.'],
        recv: ['相手の GPU からトークンや部分和を受け取ります。', 'Receives tokens or partial sums from the peer GPU.'],
      };
      const [tja, ten] = text[d.block] ?? ['', ''];
      return `${head}<p>${J(lang, tja, ten)}</p><p class="kicker">hidden = ${d.hidden}</p>${note}`;
    }
  }
}
