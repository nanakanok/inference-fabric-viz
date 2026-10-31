---
type: Web Visualization
title: inference-fabric-viz
description: Animated step-through explainer of MoE LLM inference traffic (PD / AF disaggregation) over scale-up links and scale-out leaf-spine, rail-optimized vs ToR
tags:
  - llm-inference
  - moe
  - disaggregation
  - rail-optimized
  - leaf-spine
timestamp: "2026-10-31"
---

# inference-fabric-viz

An animated, step-through web page that shows how MoE LLM inference traffic moves across a GPU fabric:

- **PD disaggregation**: Prefill and Decode on separate GPUs, with a KV Cache handoff.
- **AF disaggregation**: Attention and FFN experts on separate GPUs, with M:N token routing.
- **Scale-up** (GPU-to-GPU links through a scale-up switch inside an 8-GPU node; NVSwitch, UALink and SUE switches are examples) and **scale-out** (two-tier leaf-spine).
- **Rail-optimized** vs **ToR** wiring, scale-up relay (PXN / DeepEP-style forwarding) and ECMP hashing.

"Scale-up link" and "scale-up switch" are vendor-neutral names for NVLink / NVSwitch, UALink, SUE (Scale-Up Ethernet) and similar. DeepEP and NCCL PXN relays use NVLink on NVIDIA systems.

It is a teaching aid, not a simulator. Paths, switch counts and byte volumes are computed from an explicit topology graph, but no latency or throughput figures are shown.

The design decisions and their rationale are in [docs/adr/0001-inference-animation-design.md](docs/adr/0001-inference-animation-design.md) (Japanese).

## What you can do

- Toggle **PD**, **AF** and **rail-optimized** wiring independently. An advanced **"relay Decode over scale-up"** switch brings the total to 16 combinations.
- Step through one batch: arrival → dense layers → MoE dispatch / combine → first token → KV transfer → decode iterations. One request is highlighted and followed.
- Switch to the **overlap (autoplay)** view. With PD, Prefill, layer-wise KV transfer and Decode run at the same time. With AF, two micro-batches ping-pong between A and F.
- Hover or click any GPU, node, switch, link or moving particle to see what it is and what flows through it. Clicking pauses and pins the details; `Esc` unpins.
- Enter your own prompt or pick a preset. The (schematic) token count drives the KV and activation volumes.
- Japanese / English and light / dark themes. Keyboard: `←` `→` to step, `Space` to play or pause, `Tab` + `Enter` to inspect elements.

## Model

| Item | Value |
| :--- | :--- |
| Topology | 4 nodes × 8 GPUs, one NIC per GPU, 4 spines. Rail: 8 leaves; ToR: 4 leaves. Both are 1:1 non-blocking with the same NIC, uplink and spine-port counts. |
| Attention | DP attention (TP=1): one GPU owns a request's attention and KV. |
| Experts | 256 routed experts, top-8, spread evenly (EP) over the expert GPUs of each phase. |
| AF split | The attention side owns the router, shared expert, dense layers, final sum and residual. The FFN side owns the routed experts and returns one weighted partial sum per GPU. |
| Transfer methods | Prefill dispatch / combine: scale-up relay (DeepEP Normal). Decode dispatch / combine and A↔F: direct (DeepEP Low-Latency). KV: direct RDMA. |
| ECMP | RoCEv2, static per-flow hash on the NICs that actually send and receive plus a per-QP source port. One QP per NIC pair and purpose (a teaching assumption). The seed can be re-rolled. |
| Volumes | KV = tokens × 61 × (512 + 64) × 2 B (BF16). Send = 7168 × 1 B (FP8). Return = 7168 × 2 B (BF16). Per representative layer. |

The 32-GPU layout is a scaled-down schematic. It does not match DeepSeek-V3's production deployment, and model fit is not guaranteed.

## Usage

```bash
npm install
npm run dev          # local dev server
npm test             # vitest: topology, routing, traffic, steps, pipeline (all 16 combinations)
npm run build        # static site in dist/
npm run build:single # one self-contained HTML in dist-single/index.html (works from file://)
```

Requires Node.js 20 or later. No runtime dependencies: Vite + TypeScript + plain SVG.

## Files

| Path | Contents |
| :--- | :--- |
| `src/model/` | Topology, routing (transfer methods, ECMP), roles, workload, traffic, step sequence, overlap pipeline |
| `src/content/` | Japanese / English text, presets, notes |
| `src/view/` | SVG layout and renderer, tooltips and detail panels |
| `test/` | Vitest suites |
| `docs/adr/` | Architecture decision records |

## Sources

- DeepSeek-V3 Technical Report (architecture, node-limited routing, deployment): <https://arxiv.org/abs/2412.19437>
- DeepSeek-V3 model config (hidden size, layers, MLA dimensions, experts): <https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json>
- DeepEP (Normal and Low-Latency kernels, FP8 dispatch / BF16 combine). Reference version **v1.2.1** (`9af0e0d`): <https://github.com/deepseek-ai/DeepEP/tree/v1.2.1>
- NCCL PXN: <https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/>
- MegaScale-Infer (disaggregated expert parallelism, M2N communication): <https://arxiv.org/abs/2504.02263>
- Mooncake (KV-cache-centric disaggregated serving, layer-wise KV transfer): <https://arxiv.org/abs/2407.00079>
- RoCEv2 ECMP entropy via the UDP source port: <https://networking-docs.nvidia.com/mlnxofedswum/24010331/rdma-over-converged-ethernet-roce>

## License

[MIT](LICENSE).

<details>
<summary>日本語</summary>

# inference-fabric-viz

MoE LLM の推論で、通信が GPU ファブリックをどう流れるかを、アニメーションとステップ送りで見せる Web ページです。

- **PD 分離**: Prefill と Decode を別の GPU で処理し、KV Cache を受け渡します。
- **AF 分離**: Attention と FFN（Expert）を別の GPU に分け、トークンを M:N で送ります。
- **Scale-up**（8 GPU ノード内の Scale-up Switch を介した GPU 間リンク。NVSwitch、UALink、SUE のスイッチなどが該当）と **Scale-out**（2 段の Leaf-Spine）。
- **レール最適化**と **ToR 型**の配線、Scale-up 中継（PXN / DeepEP 方式）、ECMP ハッシュ。

「Scale-up リンク」「Scale-up Switch」は、NVLink / NVSwitch、UALink、SUE（Scale-Up Ethernet）などの総称です。DeepEP や NCCL PXN の中継は、NVIDIA の環境では NVLink を使います。

説明用の教材で、シミュレータではありません。経路、スイッチの通過数、データ量はトポロジのグラフから計算しますが、レイテンシやスループットの数値は出しません。

設計の判断とその理由は [docs/adr/0001-inference-animation-design.md](docs/adr/0001-inference-animation-design.md) にあります。

## できること

- **PD 分離**、**AF 分離**、**レール最適化**を独立に切り替えられます。詳細設定の「**Decode も Scale-up 中継する**」を合わせると 16 通りです。
- 1 つのバッチをステップ送りで追えます。到着 → dense 層 → MoE の dispatch / combine → 最初のトークン → KV 転送 → Decode の繰り返し、の順です。1 つのリクエストを強調して追跡します。
- **並行動作（自動再生）**に切り替えられます。PD 分離では Prefill、層ごとの KV 転送、Decode が同時に進みます。AF 分離では 2 つのマイクロバッチが A と F の間でピンポンします。
- GPU、ノード、スイッチ、リンク、動いている粒にマウスを載せたりクリックしたりすると、それが何で、何が流れているかを表示します。クリックすると一時停止して詳細を固定し、`Esc` で解除します。
- プロンプトは自由に入力するか、プリセットから選べます。（模式的に数えた）トークン数が KV と activation のデータ量に反映されます。
- 日本語 / 英語、ライト / ダークを切り替えられます。キーボード操作: `←` `→` でステップ送り、`Space` で再生と一時停止、`Tab` と `Enter` で要素の詳細を開きます。

## モデル

| 項目 | 内容 |
| :--- | :--- |
| トポロジ | 4 ノード × 8 GPU、GPU ごとに NIC 1 枚、Spine 4 台。レール最適化: Leaf 8 台、ToR 型: Leaf 4 台。どちらも 1:1 のノンブロッキングで、NIC 数・uplink 数・Spine のポート数は同じです。 |
| Attention | DP attention（TP=1）。1 リクエストの Attention と KV は 1 GPU が担当します。 |
| Expert | routed expert 256 個、top-8。各フェーズの Expert 用 GPU に均等に配置します（EP）。 |
| AF 分離 | A 側が router、shared expert、dense 層、最終的な合算、residual を担当します。F 側は routed expert を担当し、GPU ごとに重み付きの部分和を 1 本返します。 |
| 転送方式 | Prefill の dispatch / combine: Scale-up 中継（DeepEP Normal）。Decode の dispatch / combine と A↔F: direct（DeepEP Low-Latency）。KV: direct の RDMA。 |
| ECMP | RoCEv2、フローごとの静的ハッシュ。実際に送受信する NIC と、QP ごとの送信元ポートでハッシュします。QP は NIC の組と用途ごとに 1 本（説明用の仮定）。シード値は再抽選できます。 |
| データ量 | KV = トークン数 × 61 × (512 + 64) × 2 B（BF16）。送信 = 7168 × 1 B（FP8）。返送 = 7168 × 2 B（BF16）。代表 1 層あたりの値です。 |

32 GPU の配置は縮小した模式図です。DeepSeek-V3 の実運用の配置とは異なり、この配置でモデルが収まることは保証しません。

## 使い方

```bash
npm install
npm run dev          # 開発用サーバ
npm test             # vitest: トポロジ、経路、通信、ステップ、並行動作（16 通りすべて）
npm run build        # 静的サイトを dist/ に出力
npm run build:single # 単一の HTML を dist-single/index.html に出力（file:// でも動作）
```

Node.js 20 以降が必要です。実行時の依存はありません（Vite + TypeScript + SVG）。

## ファイル

| パス | 内容 |
| :--- | :--- |
| `src/model/` | トポロジ、経路（転送方式、ECMP）、役割、ワークロード、通信、ステップ列、並行動作 |
| `src/content/` | 日本語 / 英語の文言、プリセット、注記 |
| `src/view/` | SVG のレイアウトと描画、ツールチップ、詳細パネル |
| `test/` | Vitest のテスト |
| `docs/adr/` | 設計判断の記録（ADR） |

## 出典

- DeepSeek-V3 技術報告（アーキテクチャ、ノード制限ルーティング、運用構成）: <https://arxiv.org/abs/2412.19437>
- DeepSeek-V3 のモデル config（hidden size、層数、MLA の次元、Expert 数）: <https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json>
- DeepEP（Normal / Low-Latency カーネル、FP8 の dispatch と BF16 の combine）。参照した版は **v1.2.1**（`9af0e0d`）: <https://github.com/deepseek-ai/DeepEP/tree/v1.2.1>
- NCCL PXN: <https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/>
- MegaScale-Infer（分離型の Expert 並列、M2N 通信）: <https://arxiv.org/abs/2504.02263>
- Mooncake（KV Cache 中心の分離型サービング、層ごとの KV 転送）: <https://arxiv.org/abs/2407.00079>
- RoCEv2 の UDP 送信元ポートによる ECMP のエントロピー: <https://networking-docs.nvidia.com/mlnxofedswum/24010331/rdma-over-converged-ethernet-roce>

## ライセンス

[MIT](LICENSE)。

</details>
