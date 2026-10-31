// DeepSeek-V3-like reference model (public config). Used only for order-of-
// magnitude volume estimates (ADR §8, §19).

export const HIDDEN = 7168;
export const LAYERS = 61;
export const DENSE_LAYERS = 3;
export const MOE_LAYERS = LAYERS - DENSE_LAYERS; // 58
export const ROUTED_EXPERTS = 256;
export const TOP_K = 8;
/** Compressed latent (512) + decoupled RoPE key (64) per layer per token. */
export const KV_ELEMS_PER_LAYER = 512 + 64;
export const KV_BYTES_PER_ELEM = 2; // BF16
/** Dispatch / A->F activations: FP8. */
export const SEND_BYTES_PER_ELEM = 1;
/** Combine / F->A partial sums: BF16. */
export const RETURN_BYTES_PER_ELEM = 2;
/** DeepSeek-V3 node-limited routing: a token goes to at most 4 nodes. */
export const MAX_NODES_PER_TOKEN = 4;

export const KV_BYTES_PER_TOKEN = LAYERS * KV_ELEMS_PER_LAYER * KV_BYTES_PER_ELEM; // 70,272

export function kvBytes(tokens: number): number {
  return tokens * KV_BYTES_PER_TOKEN;
}

/** Bytes for one token, one MoE layer, one direction, one destination. */
export const SEND_BYTES_PER_TOKEN = HIDDEN * SEND_BYTES_PER_ELEM;
export const RETURN_BYTES_PER_TOKEN = HIDDEN * RETURN_BYTES_PER_ELEM;

/** Per-model dimensions used for volume estimates and routing. */
export interface ModelSpec {
  id: 'deepseek-v3' | 'qwen3-30b-a3b';
  name: string;
  hidden: number;
  layers: number;
  denseLayers: number;
  experts: number;
  topK: number;
  sharedExpert: boolean;
  /** Per-expert FFN intermediate size (moe_intermediate_size). */
  expertInter: number;
  /** Attention style, for the block zoom. */
  attention: 'mla' | 'gqa';
  /** KV bytes per token over all layers (BF16). */
  kvBytesPerToken: number;
}

export const DEEPSEEK_V3: ModelSpec = {
  id: 'deepseek-v3',
  name: 'DeepSeek-V3',
  hidden: HIDDEN,
  layers: LAYERS,
  denseLayers: DENSE_LAYERS,
  experts: ROUTED_EXPERTS,
  topK: TOP_K,
  sharedExpert: true,
  expertInter: 2048,
  attention: 'mla',
  kvBytesPerToken: KV_BYTES_PER_TOKEN,
};

/**
 * Lightweight MoE for the scale-up-only mode (public config): hidden 2048,
 * 48 layers (all MoE), 128 experts / top-8, no shared expert, GQA with
 * 4 KV heads x 128 dims, so KV = 48 x 2 (K,V) x 4 x 128 x 2 B per token.
 */
export const QWEN3_30B_A3B: ModelSpec = {
  id: 'qwen3-30b-a3b',
  name: 'Qwen3-30B-A3B',
  hidden: 2048,
  layers: 48,
  denseLayers: 0,
  experts: 128,
  topK: 8,
  sharedExpert: false,
  expertInter: 768,
  attention: 'gqa',
  kvBytesPerToken: 48 * 2 * 4 * 128 * KV_BYTES_PER_ELEM, // 98,304
};

export function sendBytesPerToken(m: ModelSpec): number {
  return m.hidden * SEND_BYTES_PER_ELEM;
}

export function returnBytesPerToken(m: ModelSpec): number {
  return m.hidden * RETURN_BYTES_PER_ELEM;
}
