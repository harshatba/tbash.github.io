/* ============================================================
   Part 2 — "How LLMs Run Fast": shared constants + helpers.
   Formula widgets (FLOPs, bytes, tokens/sec) driven by sliders.
   Loaded AFTER course-data.js. All headline numbers hand-verified
   — see _course-dev/spec-part2.md, "Hand-verification (DONE)".
   ============================================================ */

/* ---- A 7B-class toy model (Llama-2-7B-shaped, MHA) ---- */
const MODEL = {
  layers:     32,
  d_model:    4096,
  n_heads:    32,
  head_dim:   128,     // n_heads * head_dim === d_model
  n_kv_heads: 32,      // MHA: KV heads == query heads. GQA would shrink this.
  d_ff:       11008,   // SwiGLU feed-forward inner size
  vocab:      32000,
  bytes_fp16: 2
};

/* bytes per parameter, by precision */
const BYTES = { fp16: 2, int8: 1, int4: 0.5 };

/* ---- A reference data-center GPU (A100-80GB SXM) ---- */
const GPU = {
  name:      'A100-80GB',
  flops:     312e12,   // dense fp16 tensor-core FLOP/s
  bandwidth: 2.0e12,   // HBM bytes/s
  vram:      80e9      // bytes
};
const GPU_RIDGE = GPU.flops / GPU.bandwidth;   // = 156 FLOP/byte

/* ---- parameter count (weights only; unembedding tied to embedding) ---- */
function paramCount(m){
  m = m || MODEL;
  const embed = m.vocab * m.d_model;              // 131,072,000
  const attn  = 4 * m.d_model * m.d_model;        // per layer: Wq,Wk,Wv,Wo
  const ffn   = 3 * m.d_model * m.d_ff;           // per layer: gate,up,down
  return embed + m.layers * (attn + ffn);         // 6,607,077,376
}

/* ---- weight memory (bytes) at a given precision ---- */
function weightBytes(m, bytesPerParam){ return paramCount(m) * bytesPerParam; }

/* ---- KV-cache bytes: 2(K&V) × layers × n_kv_heads × head_dim × seq × batch × bytes ---- */
function kvCacheBytes(m, seq, batch, bytesPerElem){
  m = m || MODEL;
  return 2 * m.layers * m.n_kv_heads * m.head_dim * seq * batch * bytesPerElem;
}
// n_kv_heads*head_dim === d_model, so per token per layer = 2*d_model elems = 512 KiB/token total.

/* ---- decode: FLOPs per generated token ≈ 2 × params (one multiply-add = 2 FLOP) ---- */
function flopsPerToken(m){ return 2 * paramCount(m); }        // 1.32e10

/* ---- no-cache recompute: total token-K/V computations to generate n tokens ---- */
function recomputeWork(n){ return n * (n + 1) / 2; }          // triangular
// with a cache it is just n; ratio = (n+1)/2.

/* ---- arithmetic intensity of a batched decode step (GEMV → GEMM) ----
   Each weight is read once and used for 2 FLOP per request in the batch:
   AI = (2 × params × batch) / (params × bytesPerParam) = 2·batch / bytesPerParam  */
function arithmeticIntensity(batch, bytesPerParam){ return 2 * batch / bytesPerParam; }
function isComputeBound(batch, bytesPerParam){ return arithmeticIntensity(batch,bytesPerParam) >= GPU_RIDGE; }

/* ---- speculative decoding: expected tokens per target verify pass ----
   draft length k, per-token acceptance prob a (independent approx):
   E[tokens] = (1 - a^(k+1)) / (1 - a)                                   */
function specExpectedTokens(a, k){
  if(a >= 1) return k + 1;
  return (1 - Math.pow(a, k + 1)) / (1 - a);
}
// net speedup, with draft-pass cost fraction c (= draft size / target size):
function specSpeedup(a, k, c){ return specExpectedTokens(a, k) / (1 + k * c); }

/* ---- symmetric absmax quantization of a weight vector to `bits` ---- */
function quantizeSym(w, bits){
  const absmax = Math.max(...w.map(Math.abs));
  const qmax   = Math.pow(2, bits - 1) - 1;                 // int4 -> 7
  const scale  = absmax / qmax;
  const q      = w.map(x => Math.max(-qmax, Math.min(qmax, Math.round(x / scale))));
  const deq    = q.map(k => k * scale);
  const mae    = deq.reduce((s,d,i)=> s + Math.abs(d - w[i]), 0) / w.length;
  return { scale, q, deq, mae, levels: Math.pow(2, bits) };
}
/* small readable weight vector for the quant worked-example (page 4 hood) */
const QUANT_W = [0.14,-0.42,0.35,-0.14,0.07,0.28,-0.35,0.21];

/* ---- paging: round a length up to whole pages ---- */
function pagesFor(len, pageSize){ return Math.ceil(len / pageSize) * pageSize; }
const PAGE_REQS   = [512, 37, 1900, 128, 4];   // example live sequence lengths
const PAGE_MAXLEN = 2048;                       // reserve-max slot size
const PAGE_SIZE   = 16;                          // tokens per page

/* ---- continuous batching: decode lengths across B=4 slots ---- */
const BATCH_LENS = [20, 8, 12, 6];
// static utilization = sum / (N × max);  continuous keeps every slot full.
