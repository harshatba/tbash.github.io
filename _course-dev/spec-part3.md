# Build Spec — Part 3
## "Beyond the Basics" — systems internals & capability patterns, 8 compute-live lessons

**For:** coding agents building directly from this spec.
**Design system (do not restyle):** `tools/course/course.css`, `tools/course/course.js`, clone `tools/course/_template.html` for every new page. All 8 files live in `tools/course/` beside the Part-1 and Part-2 pages so `prev`/`next` relative links resolve.
**Load order on every Part-3 page:** `<script src="course-data.js"></script>` → `<script src="course-data-p2.js"></script>` → `<script src="course-data-p3.js"></script>` → `<script src="course.js" defer></script>` → the page's own `<script>`. (Part-3 reuses `E / TOKENS / cosine / dot / softmax / matvec` from p1 **and** `MODEL / GPU / GPU_RIDGE / BYTES / paramCount / weightBytes / kvCacheBytes / arithmeticIntensity` from p2, so all three data files must load.)
**The bar to hit (unchanged from Parts 1–2):** every widget must **COMPUTE its result live from the shared constants on each interaction — recompute, never replay.** No headline number may be typed as a constant in the HTML; it must fall out of a formula or a step-logic the reader can trace, driven by the controls. The reference bar for "a widget that computes" is still `attention-scoring.html` (staged, traceable arithmetic) and `roofline.html` (a live formula widget). **The four less-numeric pages (tool-use, reasoning, rag, multimodal) must still be genuinely INTERACTIVE and mechanism-revealing** — a stepper, a live cosine retrieval, a live patch→vector projection — never prose with a static diagram. No page may be prose-only.

**Color canon (reuse consistently; from course.css):** amber = tokens/**structure** · cyan = vectors/embeddings/**memory** · violet = attention/**compute** · rose = **weights**/parameters · green = training/probabilities/**throughput**.

**Owner's rule:** ONE core idea + ONE widget per page. Do not merge pages. Do not add pages. Filenames, order, and eyebrow numbers are LOCKED (below). This is a **new Part 3 sub-track** — do NOT renumber Part 1 or Part 2.

**Locked sequence (eyebrow reads `Part 3 · Lesson N/8`):**
```
1. gpu-anatomy.html             prev: index.html
2. grouped-query-attention.html
3. flash-attention.html
4. mixture-of-experts.html
5. tool-use.html
6. reasoning.html
7. rag.html
8. multimodal.html              next: index.html
```
Page 1 `prev` → `index.html`; page 8 `next` → `index.html`. Internal pages chain the sequence above. Crumbs on every page: `Course / Part 3 · Beyond the Basics / NN · title`, with the middle crumb linking `index.html#part3`.

---

# A. CONCEPT CANON — Part 3 additions

Plain language, define each term the first time a page needs it, then reuse the phrasing. These continue the Part-1 and Part-2 canon; **every page must tie back to an earlier lesson by name** (the tie is called out per-page below).

- **GPU memory hierarchy — VRAM vs SRAM vs registers, and the SM** *(pg 1)* — a GPU has three tiers of memory, fast-and-tiny to slow-and-huge. **Registers** (a few KB per thread, instant). **SRAM / shared memory** (on-chip, ~192 KB per compute unit on an A100, very fast, tiny). **VRAM / HBM** (the 80 GB the model lives in, ~2 TB/s — huge but "far away"). The compute units themselves are **SMs (Streaming Multiprocessors)** — the ~108 little processors that do the multiply-adds. The physical fact that runs all of Part 2: the SMs are fast, but a decode step must drag every weight up from VRAM through this hierarchy, and **VRAM bandwidth is the wall** — that's *why* decode is bandwidth-bound (Part-2 lesson 1 / the roofline, lesson 8). Weights, the KV cache, and activations all live in VRAM; only the sliver being worked on right now fits in SRAM.

- **grouped-query attention (GQA) / multi-query attention (MQA)** *(pg 2)* — in the attention of Part-1 lesson 6, every **query head** had its own **Key** and **Value** head (that's **MHA**, multi-head attention). The KV cache (Part-2 lesson 3) stores one K and one V per token *per head*, so its size scales with the number of KV heads. **GQA** lets several query heads **share** one K/V head: with 32 query heads and 8 KV heads, four query heads share each K/V, and the cache shrinks by `n_query_heads / n_kv_heads = 32/8 = 4×`. **MQA** is the extreme — one K/V head for all 32 query heads, a 32× shrink. The queries still differ per head; only the K/V are shared. This is the exact lever `kv-memory.html` named ("cut n_kv_heads 32→8, cache shrinks 4×").

- **FlashAttention — tiling + online softmax + never materialize n×n** *(pg 3)* — naive attention builds the full `n×n` score matrix (every token's Query dotted with every token's Key, Part-1 lesson 6) and **writes it to VRAM**, then reads it back to softmax and blend. That matrix is `O(n²)` and dominates the memory traffic. **FlashAttention** never writes it: it cuts Q, K, V into **tiles** small enough to fit in on-chip **SRAM**, computes attention one tile-pair at a time, and uses an **online softmax** (a running max and running sum that let each new tile correct the partial result without ever seeing the whole row). The answer is bit-identical; the win is **memory movement, not FLOPs** — the same insight as the roofline (Part-2 lesson 8): attention was bandwidth-bound on the score matrix, and Flash keeps that matrix on-chip. It is **IO-aware**: designed around the memory hierarchy of lesson 1.

- **mixture-of-experts (MoE) — router, top-k, sparse activation, active vs total params** *(pg 4)* — a dense model runs **every** weight for every token. An MoE replaces the single feed-forward network (the FFN from Part-1 lesson 9) with **N separate expert FFNs** and a small **router** (a gate: a tiny weight matrix that scores the token's vector against each expert, softmax, Part-1 lesson 6's dot-product-then-softmax again). Each token is sent to only its **top-k** experts (**sparse activation**). So **total params** grow with N (more experts stored), but **active params** — and therefore **FLOPs per token** (Part-2 roofline) — stay near a small-k dense model. **Load balancing** matters: if every token picked the same expert, the rest would be dead weight, so training adds a term that spreads tokens across experts. Different tokens fire different experts (verifiable on the Part-1 tokens).

- **tool use / function calling** *(pg 5)* — the model is still just the next-token predictor of Part-1 lesson 1. To "use a tool" it **emits a structured tool call as ordinary tokens** (a JSON blob it learned to produce). A wrapper around the model watches the stream; when a complete call appears, **generation pauses**, the wrapper **actually runs** the function (the model never runs anything itself), and the **result is injected back into the context as more tokens**. Then generation **resumes** with the result now visible. The model never "calls" anything — it predicts call-shaped tokens, and the harness does the rest. (Ties to Part-1 lesson 14 hallucination: a tool injects *ground truth* the frozen weights don't contain.)

- **chain-of-thought / test-time compute** *(pg 6)* — Part-1's myth-buster said the math for **one** token is trivial. Reasoning models exploit exactly that: spend **more tokens** "thinking out loud" (a scratchpad of intermediate tokens) **before** committing to an answer, and accuracy rises — because each extra token is another cheap forward pass that can condition the next. This is **test-time compute**: buying accuracy with generation length instead of a bigger model. Returns **diminish** (accuracy climbs then plateaus), and every thinking token costs one **decode step** (Part-2 lesson 1), so it trades latency for correctness. Reasoning helps **hard** problems far more than easy ones.

- **RAG vs fine-tuning vs prompting — three ways to add knowledge** *(pg 7)* — the weights are **frozen** after training (Part-1: "train vs run"), so a running model can't learn a new fact by itself. Three ways to get new knowledge in: **prompting** (paste it into the context — instant, but limited by the context window and you must have the text); **fine-tuning** (actually retrain the weights on new data — permanent, expensive, and it re-freezes); **RAG** (retrieval-augmented generation: keep a library of documents as embedding vectors, and at query time **retrieve** the most relevant ones by **embedding similarity — the cosine of Part-1 lesson 2** — and prepend them to the prompt). RAG = automated prompting: retrieval finds the text, prompting injects it. When to use which: prompting for one-off facts, RAG for a changing knowledge base, fine-tuning for changing *behavior/format*.

- **multimodal tokens — patches/frames become embeddings in the same stream** *(pg 8)* — the transformer only ever ate a **list of embedding vectors** (Part-1 lesson 1). Nothing about that list is text-specific. An image is cut into fixed **patches** (e.g. 16×16 pixels); each patch is flattened and **linearly projected** (a weight matrix, one matvec) into an embedding vector the same size as a text-token embedding — then a **position encoding** is added (Part-1 lesson 13) and the patch-vectors are **appended to the token stream**. Attention (Part-1 lessons 5–7) then treats image-patches and text-tokens identically. Audio becomes frames the same way. **Tokens aren't only text** — they're whatever you can turn into a vector in the shared space.

---

# B. SHARED DATA / HELPERS — new file `tools/course/course-data-p3.js`

Create this new file (do **not** append to `course-data.js` or `course-data-p2.js`; Part-3 pages load all three). JS-ready and hand-verified. It **reuses** `MODEL / GPU / paramCount / weightBytes / kvCacheBytes / BYTES` from `course-data-p2.js` and `E / TOKENS / matvec / dot / cosine / softmax` from `course-data.js`, and adds only what is new.

```js
/* ============================================================
   Part 3 — "Beyond the Basics": shared constants + helpers.
   Loaded AFTER course-data.js AND course-data-p2.js.
   Reuses: E, TOKENS, matvec, dot, cosine, softmax (p1);
           MODEL, GPU, paramCount, weightBytes, kvCacheBytes, BYTES (p2).
   All headline numbers hand-verified — see "Hand-verification (DONE)".
   ============================================================ */

/* ------------------------------------------------------------------
   Page 1 · GPU anatomy — the memory hierarchy (reference figures).
   These annotate a diagram; the widget computes "does X fit in tier Y?"
   ------------------------------------------------------------------ */
const HIER = {
  // one A100-80GB SXM (matches GPU in course-data-p2.js)
  sms:            108,          // Streaming Multiprocessors (compute units)
  registers_kb:   256,          // register file per SM (KB)
  sram_kb:        192,          // on-chip SRAM / shared+L1 per SM (KB)
  vram_gb:        80,           // HBM (GB)
  vram_bw_tbs:    2.0,          // HBM bandwidth (TB/s)   == GPU.bandwidth
  sram_bw_tbs:    19            // on-chip SRAM bandwidth (TB/s, ~10× HBM) — illustrative
};
// what lives where, for the diagram (bytes from the shared helpers):
//   weights (fp16)   = weightBytes(MODEL, 2)            = 13.2 GB   -> VRAM
//   kv cache (4k,b1) = kvCacheBytes(MODEL,4096,1,2)     = 2.15 GB   -> VRAM
//   one attention tile (Part-3 pg3 FLASH default)       = 56 KiB    -> fits SRAM
function fitsInSram(bytes){ return bytes <= HIER.sram_kb * 1024; }
function fitsInVram(bytes){ return bytes <= HIER.vram_gb * 1e9; }

/* ------------------------------------------------------------------
   Page 2 · Grouped-query attention — KV cache vs n_kv_heads.
   Reuses MODEL (n_heads 32, head_dim 128, layers 32).
   ------------------------------------------------------------------ */
const GQA = {
  configs: [
    { name:'MHA',   n_kv:32, blurb:'every query head has its own K/V (Part-1 lesson 6)' },
    { name:'GQA-8', n_kv:8,  blurb:'8 K/V heads shared across 32 query heads (Llama-3 8B)' },
    { name:'GQA-4', n_kv:4,  blurb:'4 K/V groups' },
    { name:'MQA',   n_kv:1,  blurb:'one K/V head for all 32 query heads (extreme)' }
  ]
};
// bytes of KV cache PER TOKEN (all layers) for a given number of KV heads:
function gqaKvBytesPerToken(m, n_kv, bytes){ m=m||MODEL; return 2*m.layers*n_kv*m.head_dim*bytes; }
// full KV cache bytes at a sequence length + batch:
function gqaKvCacheBytes(m, n_kv, seq, batch, bytes){ return gqaKvBytesPerToken(m,n_kv,bytes)*seq*batch; }
// cache shrink factor vs full MHA:
function gqaShrink(m, n_kv){ m=m||MODEL; return m.n_heads / n_kv; }

/* ------------------------------------------------------------------
   Page 3 · FlashAttention — tiling model (per attention head).
   Naive materializes the n×n score matrix in VRAM; Flash keeps one
   Br×Bc tile in SRAM. head_dim d = 128, fp16.
   ------------------------------------------------------------------ */
const FLASH = { d: 128, bytes: 2, sram_budget: HIER.sram_kb*1024 /* 196608 B */ };
// naive per-head score matrix bytes = n*n*bytes  (the thing Flash never writes):
function scoreMatrixBytes(n, bytes){ bytes=bytes||FLASH.bytes; return n*n*bytes; }
// SRAM a single Br×Bc tile needs: Q block + K block + V block + scores block.
// with square tiles Br=Bc=B: (3*B*d + B*B) * bytes
function flashTileBytes(B, d, bytes){ d=d||FLASH.d; bytes=bytes||FLASH.bytes; return (3*B*d + B*B)*bytes; }
function flashTilesPerAxis(n, B){ return Math.ceil(n/B); }                 // Q-blocks (and K-blocks)
function flashMemReduction(n, B){ return (n/B)*(n/B); }                    // score-matrix VRAM shrink
// K/V is re-streamed once per Q-block ⇒ passes over K/V = tiles per axis:
function flashKvPasses(n, B){ return flashTilesPerAxis(n, B); }

/* ------------------------------------------------------------------
   Page 4 · Mixture-of-Experts — router over a token's embedding, and
   active-vs-total params. Router reuses Part-1 E (4 toy dims:
   [animacy, action, state, function]) + matvec + softmax.
   Param preset is Mixtral-8x7B-shaped (its own d_ff, NOT MODEL.d_ff).
   ------------------------------------------------------------------ */
const MOE = {
  n_experts: 8,
  top_k: 2,
  // ---- param preset (its own dims; Mixtral-8x7B-like) ----
  d_model: 4096, d_ff: 14336, layers: 32, vocab: 32000,
  // ---- toy router gate: 8 experts × 4 dims (rows read the Part-1 E dims) ----
  gate: [
    [ 3.0, 0.0, 0.0, 0.0],   // e0  animacy / entity
    [ 0.0, 3.0, 0.0, 0.0],   // e1  action / verb
    [ 0.0, 0.0, 3.0, 0.0],   // e2  state / adjective
    [ 0.0, 0.0, 0.0, 3.0],   // e3  function / glue
    [ 1.5, 1.5, 0.0, 0.0],   // e4  animate + action
    [ 1.5, 0.0, 1.5, 0.0],   // e5  animate + state
    [ 0.5, 0.5, 0.5, 0.5],   // e6  generalist
    [-1.0,-1.0,-1.0,-1.0]    // e7  cold / rarely picked
  ],
  labels: ['animacy','action','state','function','animate+action','animate+state','generalist','cold']
};
// route ONE token embedding to its top-k experts (renormalized gate over the top-k):
function moeRoute(embed, k){
  k = k || MOE.top_k;
  const logits = matvec(MOE.gate, embed);                       // reuse Part-1 matvec
  const order  = logits.map((z,i)=>({i,z})).sort((a,b)=>b.z-a.z);
  const top    = order.slice(0,k);
  const w      = softmax(top.map(o=>o.z));                      // reuse Part-1 softmax
  return { logits, top: top.map((o,j)=>({expert:o.i, label:MOE.labels[o.i], logit:o.z, weight:w[j]})) };
}
// total vs active params:
function moeParams(p){
  p = p || MOE;
  const embed  = p.vocab * p.d_model;                           // 131,072,000
  const attn   = 4 * p.d_model * p.d_model;                     // per layer (dense attn)
  const router = p.n_experts * p.d_model;                       // per layer (tiny)
  const expert = 3 * p.d_model * p.d_ff;                        // one expert FFN (gate,up,down)
  const total  = embed + p.layers * (attn + router + p.n_experts * expert);
  const active = embed + p.layers * (attn + router + p.top_k   * expert);
  return { embed, attn, router, expert, total, active, activeFrac: active/total };
}

/* ------------------------------------------------------------------
   Page 5 · Tool use — a scripted token stream with a JSON call, and a
   MOCK tool the widget actually runs (result templated live, not replayed).
   ------------------------------------------------------------------ */
const TOOLUSE = {
  userPrompt: "What's the weather in Paris right now?",
  // model emits these one token at a time (pre-tokenized for the toy):
  reason:      ['I','need','live','data','—','calling','a','tool','.'],
  callTokens:  ['{','"tool"',':','"get_weather"',',','"args"',':','{','"city"',':','"<CITY>"','}','}'],
  // the harness runs THIS (the model never does) — real lookup, editable city:
  weatherDB: {
    Paris:  { tempC:18, sky:'cloudy' },
    Tokyo:  { tempC:24, sky:'clear'  },
    Cairo:  { tempC:34, sky:'sunny'  },
    Oslo:   { tempC: 6, sky:'rain'   }
  },
  cities: ['Paris','Tokyo','Cairo','Oslo']
};
function runWeather(city){ return TOOLUSE.weatherDB[city] || null; }
// injected-result tokens + resumed answer are BUILT from the tool result (recompute):
function toolResultTokens(city){
  const r = runWeather(city);
  return ['{','"tempC"',':', String(r.tempC), ',','"sky"',':', '"'+r.sky+'"','}'];
}
function toolAnswer(city){
  const r = runWeather(city);
  return `It's ${r.tempC}°C and ${r.sky} in ${city} right now.`;
}

/* ------------------------------------------------------------------
   Page 6 · Reasoning — accuracy vs thinking-tokens (saturating curve).
   acc(t) = floor + (cap - floor) * (1 - e^(-t/tau))
   ------------------------------------------------------------------ */
const REASON = {
  presets: {
    typical: { floor:0.30, cap:0.85, tau:120, label:'a typical problem' },
    hard:    { floor:0.15, cap:0.80, tau:200, label:'a hard problem'    },
    easy:    { floor:0.70, cap:0.95, tau:60 , label:'an easy problem'   }
  },
  maxTokens: 600
};
function reasonAcc(t, p){ p = p || REASON.presets.typical; return p.floor + (p.cap-p.floor)*(1 - Math.exp(-t/p.tau)); }
// marginal gain of the last `step` thinking tokens (percentage points):
function reasonMarginal(t, step, p){ return 100*(reasonAcc(t,p) - reasonAcc(Math.max(0,t-step),p)); }

/* ------------------------------------------------------------------
   Page 7 · RAG — a tiny corpus + queries as 4-dim vectors (E-style).
   Retrieval reuses Part-1 cosine. dims: [animals, weather, cooking, space]
   ------------------------------------------------------------------ */
const RAG = {
  dims: ['animals','weather','cooking','space'],
  passages: [
    { id:'P0', text:'Cats purr when content and sleep about 15 hours a day.',      vec:[1.0,0.0,0.1,0.0] },
    { id:'P1', text:'A thunderstorm forms when warm, moist air rises quickly.',    vec:[0.0,1.0,0.0,0.1] },
    { id:'P2', text:'Caramelize onions by cooking them slowly in butter ~40 min.', vec:[0.0,0.1,1.0,0.0] },
    { id:'P3', text:'Jupiter is the largest planet; its Red Spot is a giant storm.',vec:[0.0,0.2,0.0,1.0] },
    { id:'P4', text:'Penguins are flightless birds that huddle for warmth.',       vec:[0.9,0.3,0.0,0.1] },
    { id:'P5', text:'Sourdough rises from wild yeast fermenting the dough.',       vec:[0.1,0.0,0.9,0.0] }
  ],
  queries: [
    { q:'How long do cats sleep?',          vec:[1.0,0.0,0.0,0.0] },
    { q:'What temperature to cook onions?', vec:[0.0,0.1,1.0,0.0] },
    { q:'Which is the biggest planet?',     vec:[0.0,0.1,0.0,1.0] }
  ]
};
function ragRetrieve(qvec, topN){
  topN = topN || 2;
  return RAG.passages
    .map(p => ({ id:p.id, text:p.text, vec:p.vec, cos: cosine(qvec, p.vec) }))   // reuse Part-1 cosine
    .sort((a,b) => b.cos - a.cos)
    .slice(0, topN);
}

/* ------------------------------------------------------------------
   Page 8 · Multimodal — a 4×4 toy "image" → 2×2 patches → each patch a
   4-dim embedding via a projection matrix (reuse matvec). Ties to Part-1
   lesson 1 (embeddings) + lesson 13 (position added to patches).
   ------------------------------------------------------------------ */
const IMG = [                                 // 4×4 grayscale, bright diagonal
  [0.9, 0.1, 0.1, 0.1],
  [0.1, 0.9, 0.1, 0.1],
  [0.1, 0.1, 0.9, 0.1],
  [0.1, 0.1, 0.1, 0.9]
];
const PATCH = 2;                              // 2×2-pixel patches → 4 patches
// patch-projection: 4 flattened pixels → 4-dim embedding. Rows are interpretable:
const WPATCH = [
  [ 0.25, 0.25, 0.25, 0.25],   // e0 brightness (mean)
  [ 0.5 ,-0.5 , 0.5 ,-0.5 ],   // e1 left-right edge
  [ 0.5 , 0.5 ,-0.5 ,-0.5 ],   // e2 top-bottom edge
  [ 0.5 ,-0.5 ,-0.5 , 0.5 ]    // e3 diagonal
];
const PATCH_DIMS = ['brightness','LR-edge','TB-edge','diagonal'];
// patch position encodings (added after projection — Part-1 lesson 13), tiny:
const PATCH_POS = [ [0,0,0,0.1],[0,0,0,0.2],[0,0,0,0.3],[0,0,0,0.4] ];
// cut the image into row-major patches; each patch flattened row-major:
function patchesOf(img, p){
  const out = []; const g = img.length;
  for(let pr=0; pr<g; pr+=p){ for(let pc=0; pc<g; pc+=p){
    const px = [];
    for(let r=0;r<p;r++) for(let c=0;c<p;c++) px.push(img[pr+r][pc+c]);
    out.push({ row:pr/p, col:pc/p, pixels:px });
  }}
  return out;                                  // 4 patches for a 4×4 image, p=2
}
function patchEmbed(pixels){ return matvec(WPATCH, pixels); }   // reuse Part-1 matvec
```

### Hand-verification (DONE — numbers below must reproduce; if a build differs, the build is wrong)

**Page 1 · GPU anatomy.** Reference figures only; the widget's job is "does it fit?": `weightBytes(MODEL,2) = 13.2 GB` → fits 80 GB VRAM ✓, does NOT fit 192 KB SRAM ✓; `kvCacheBytes(MODEL,4096,1,2) = 2.15 GB` → VRAM ✓; one Flash tile `flashTileBytes(64) = 57,344 B = 56 KiB` → `fitsInSram` **true** ✓. SRAM bandwidth ~19 TB/s vs HBM 2.0 TB/s ⇒ on-chip is ~**10×** faster, the reason Flash (pg 3) wants the score matrix on-chip.

**Page 2 · GQA.** `gqaKvBytesPerToken`, fp16, all 32 layers, head_dim 128:
- MHA `n_kv=32`: `2·32·32·128·2 = 524,288 B = 512 KiB/token` (matches Part-2 kv-memory) ✓
- GQA-8 `n_kv=8`: `2·32·8·128·2 = 131,072 B = 128 KiB/token`; shrink `32/8 = 4×` ✓
- GQA-4 `n_kv=4`: `65,536 B = 64 KiB`; shrink `8×` ✓
- MQA `n_kv=1`: `16,384 B = 16 KiB`; shrink `32×` ✓
Full cache at seq 8192, batch 1, fp16: MHA `524288·8192 = 4,294,967,296 B = 4.0 GiB (4.29 GB)`; GQA-8 `= 1.0 GiB (1.07 GB)`; MQA `= 128 MiB (134 MB)`. Groups: 32 query heads / `n_kv` KV heads ⇒ each K/V shared by `32/n_kv` query heads (4 for GQA-8, 32 for MQA). ✓

**Page 3 · FlashAttention.** head_dim d=128, fp16, tile Br=Bc=B.
- naive score matrix, `n=4096`: `scoreMatrixBytes(4096) = 4096²·2 = 33,554,432 B = 32 MiB` (per head) ✓; `n=8192` → `128 MiB` ✓.
- one tile, `B=64`: `flashTileBytes(64) = (3·64·128 + 64²)·2 = (24,576 + 4,096)·2 = 28,672·2 = 57,344 B = 56 KiB` ≤ 192 KiB SRAM ✓.
- `flashTilesPerAxis(4096,64) = 64` per axis ⇒ `64² = 4096` tile-pairs; `flashMemReduction(4096,64) = (4096/64)² = 64² = 4096×` smaller peak score-memory ✓.
- K/V passes `flashKvPasses(4096,64) = 64`. Headline: the `n×n` matrix (32 MiB/head) is **never written to VRAM**; only a **56 KiB** tile lives on-chip at a time. ✓

**Page 4 · MoE.** Router on the Part-1 tokens (top-2), `logits = matvec(MOE.gate, E[tok])`:
- `cat [1.0,0.2,0.1,0]` → logits `[3.0, 0.6, 0.3, 0.0, 1.8, 1.65, 0.65, −1.3]` → top-2 **e0 (3.0), e4 (1.8)**; gate `softmax([3.0,1.8]) = [0.769, 0.231]` → **e0 76.9%, e4 23.1%** ✓
- `sat [0.2,1.0,0.1,0]` → logits `[0.6, 3.0, 0.3, 0.0, 1.8, 0.45, 0.65, −1.3]` → top-2 **e1 (3.0), e4 (1.8)** ✓
- `tired [0.3,0.1,1.0,0]` → logits `[0.9, 0.3, 3.0, 0.0, 0.6, 1.95, 0.7, −1.4]` → top-2 **e2 (3.0), e5 (1.95)** ✓
- `it [0.8,0.1,0.2,0.3]` → logits `[2.4, 0.3, 0.6, 0.9, 1.35, 1.5, 0.7, −1.4]` → top-2 **e0 (2.4), e5 (1.5)** ✓
Four tokens, four different top-expert pairs — sparse routing is real, not staged.
Params (`moeParams()`, Mixtral-8x7B-shaped): `embed = 32000·4096 = 131,072,000`; `attn = 4·4096² = 67,108,864`; `router = 8·4096 = 32,768`; one `expert = 3·4096·14336 = 176,160,768`.
- **total** `= 131,072,000 + 32·(67,108,864 + 32,768 + 8·176,160,768) = 131,072,000 + 32·1,476,427,776 = 47,376,760,832 ≈ 47.4B`
- **active** `= 131,072,000 + 32·(67,108,864 + 32,768 + 2·176,160,768) = 131,072,000 + 32·419,463,168 = 13,553,893,376 ≈ 13.6B`
- `activeFrac = 13.55B / 47.38B = 0.286` → **only ~29% of params run per token** (FLOPs like a 13.6B dense model, storage like a 47B one). ✓ (Raw expert fraction `k/N = 2/8 = 25%`; overall is a bit higher because attention+embeddings are always active.)

**Page 5 · tool use.** No headline number — verify the *flow* recomputes: pick `Tokyo` → `runWeather('Tokyo') = {tempC:24, sky:'clear'}` → injected tokens `['{','"tempC"',':','24',',','"sky"',':','"clear"','}']` → resumed answer `toolAnswer('Tokyo') = "It's 24°C and clear in Tokyo right now."` Changing the city changes the injected result AND the resumed sentence — built from the tool result, never replayed. ✓

**Page 6 · reasoning.** `acc(t) = floor + (cap−floor)(1−e^(−t/τ))`, typical `{0.30, 0.85, 120}`:
- `t=0` → 30.0% · `t=60` → 51.6% · `t=120` → 64.8% · `t=240` → 77.6% · `t=480` → 84.0% · `t=600` → 84.6% (cap 85%). ✓
- Diminishing returns: `reasonMarginal(60,60)` = +21.6 pts (first 60), `reasonMarginal(180,60)` = +7.9 pts, `reasonMarginal(540,60)` = +0.4 pts. ✓
- `hard {0.15,0.80,200}`: `t=0` 15.0% → `t=200` 56.1% → `t=600` 76.0% (lifts 15→76). `easy {0.70,0.95,60}`: `t=0` 70.0% → `t=60` 85.8% (starts high, gains little). Reasoning helps hard ≫ easy. ✓

**Page 7 · RAG.** `ragRetrieve` (reuse `cosine`):
- Q "cats sleep" `[1,0,0,0]` → **P0** cos `1/(1·√1.01)=0.995` (top), then P4 `0.9/√0.91=0.944`. ✓
- Q "cook onions" `[0,0.1,1,0]` → **P2** cos `1.01/(√1.01·√1.01)=1.00` (top), then P5 `0.9/(√1.01·√0.82)=0.989`. ✓
- Q "biggest planet" `[0,0.1,0,1]` → **P3** cos `1.02/(√1.01·√1.04)=0.995` (top), then P1 `0.2/(√1.01·√1.01)=0.198`. ✓
Every query cleanly retrieves its intended passage; a query far from all passages yields a low top-cosine ("nothing relevant → inject nothing → back to guessing," ties to Part-1 lesson 14).

**Page 8 · multimodal.** `patchesOf(IMG,2)` → 4 patches (row-major, each flattened row-major):
- patch(0,0) pixels `[0.9,0.1,0.1,0.9]` → `patchEmbed = [0.25·2.0, 0, 0, 0.5·0.9−0.5·0.1−0.5·0.1+0.5·0.9] = [0.5, 0.0, 0.0, 0.8]` (bright + diagonal) ✓
- patch(0,1) pixels `[0.1,0.1,0.1,0.1]` → `[0.1, 0.0, 0.0, 0.0]` (dark) ✓
- patch(1,0) pixels `[0.1,0.1,0.1,0.1]` → `[0.1, 0.0, 0.0, 0.0]` ✓
- patch(1,1) pixels `[0.9,0.1,0.1,0.9]` → `[0.5, 0.0, 0.0, 0.8]` ✓
The two diagonal patches light up **brightness** and **diagonal**; the off-diagonal patches are dark. Each 4-number vector then gets `PATCH_POS[i]` added and joins the token stream — same shape as a text embedding. Real anchor (hood): a 224×224 image at 16×16 patches → `(224/16)² = 14² = 196` patches → 196 image-tokens, each projected to `d_model`. ✓

---

# C. PER-PAGE SPEC (8 pages)

> Eyebrow reads `Part 3 · Lesson N/8`. Every widget carries a `.stage-note` with a `TRY →` line naming the exact interaction. Every page: prose (2–4 beats) → widget (recompute/step, not replay) → `.hood` with exact numbers → myth/reality + `.takeaway` → self-check → pager. Meta pills: `⏱ ~X min` · a middle pill naming the compute/interaction · `math: some` (or `math: none` for tool-use/multimodal steppers). Reuse the shared classes already in Parts 1–2 (`.stage`, `.stage-label`, `.stage-note`, `.reads`/`.read`/`.rnum`/`.rlab`, `.bneck`/`.bn-band`/`.bn-comp`, `.metric-row` + `.bar-track`/`.bar-fill`, `.tank`, `.seg` toggle, `.ctl`/`.ctl-top`/`.ctl-lab`, `.formula`, `.hood table`, `.tok` chips, the `.stg`/`.stg-h`/`.stg-num`/`.stg-b` staged-reveal shells). Bottleneck badge language stays: **violet = compute**, **cyan = bandwidth/memory**, **rose = capacity**.

---

## 1 · gpu-anatomy.html — What a GPU is: VRAM ↔ SMs ↔ tiny on-chip SRAM

- **title (h1):** What a <span class="cyan">GPU</span> actually is: memory, all the way down
- **eyebrow:** `Part 3 · Lesson 1/8`
- **objective:** Ground the whole of Part 2 in one picture — the memory hierarchy (registers → SRAM → VRAM), the SMs that compute, and why VRAM bandwidth is the wall that makes decode bandwidth-bound.
- **prereq:** Part-2 lesson 1 (decode is bandwidth-bound) and lesson 8 (the roofline) — this page shows the *physical stage* those lessons play on.
- **prose beats:**
  1. A GPU is a few hundred tiny processors (**SMs**, ~108 on an A100) wired to a big slow pool of memory (**VRAM / HBM**, 80 GB at 2 TB/s). Between them sit two tiny-but-fast tiers: **on-chip SRAM** (~192 KB per SM) and **registers**.
  2. Everything the model *is* lives in VRAM: the frozen weights (13.2 GB), the KV cache (Part-2 lesson 3), the activations. The SMs can only compute on what's been dragged up into SRAM/registers first.
  3. Here's the wall: SRAM is ~10× faster than VRAM but holds only kilobytes, so a decode step must stream all 13.2 GB of weights *up from VRAM* to do a little math — **VRAM bandwidth is the bottleneck**, exactly the "bandwidth-bound" of Part-2. Every trick in Part 2 (and FlashAttention next page) is a move against this hierarchy.
- **THE WIDGET — computational spec (a "what lives where / does it fit" hierarchy explorer):**
  - **Inputs:** a set of chips for the things a model holds — **weights (fp16/int4 toggle)**, **KV cache (seq slider 512…131072)**, **one attention score-matrix (seq)**, **one Flash tile** — and the reader clicks one to "place" it in the hierarchy.
  - **Computed live** from the shared helpers: `weightBytes(MODEL, BYTES[prec])`, `kvCacheBytes(MODEL, seq, 1, 2)`, `scoreMatrixBytes(seq)`, `flashTileBytes(64)`; then `fitsInSram(bytes)` and `fitsInVram(bytes)` decide which tier lights up. **Recompute, not replay** — moving the seq slider re-sizes the KV/score bars and can flip "fits in VRAM → won't fit."
  - **What updates on screen:** a three-tier diagram (registers · SRAM 192 KB · VRAM 80 GB) drawn to a log scale, with the selected item rendered as a bar and dropped into the smallest tier it fits in (SRAM badge cyan if it fits, else VRAM; if > 80 GB, a rose "won't fit" capacity badge). A `.reads` row: **item size**, **tier it lands in**, **that tier's bandwidth** (2 TB/s VRAM vs ~19 TB/s SRAM), and **"× slower than SRAM."** A caption: "the 56 KiB Flash tile fits on-chip; the 32 MiB score matrix and 13.2 GB weights do not — they live in slow VRAM."
  - **Formula on screen:** `fits SRAM ⇔ bytes ≤ 192 KiB` · `weights = paramCount × bytes/param` · `KV = 2·layers·heads·dim·seq·2`.
- **under-the-hood (.hood):** A100: 108 SMs, 192 KB SRAM each, 80 GB HBM at 2.0 TB/s (~19 TB/s on-chip). weights fp16 = 13.2 GB (VRAM), KV at 4k = 2.15 GB (VRAM), one head's score matrix at n=4096 = 32 MiB (VRAM), one 64×64 Flash tile = 56 KiB (SRAM ✓). "A decode step reads all 13.2 GB up from VRAM to do ~13 GFLOP — 2 TB/s means ~6.6 ms just to move the weights (Part-2 lesson 1's exact number). The math is trivial; the *fetch* is the cost. That is the whole roofline (Part-2 lesson 8) in hardware."
- **myth / reality:** *Myth* — "A GPU is fast because it has lots of fast memory." / *Reality* — "It has a *little* fast memory (SRAM) and a *lot* of slow memory (VRAM). The art of fast inference is keeping the SMs fed across that gap."
- **self-check:** *"Why is a batch-1 decode step 'bandwidth-bound'?"* — (a) the SMs are too slow to do the math *(wrong)*; **(b) the weights live in slow VRAM and must be streamed up to the SMs each step — memory movement, not math, is the limit** *(correct, `data-correct="1"`)*; (c) the SRAM is too big to search *(wrong)*. **Why:** SRAM is tiny and fast, VRAM is huge and slow; a decode step drags every weight up from VRAM, so bandwidth is the wall.
- **prev / next:** prev: `index.html` ("Back to the course map") · next: `grouped-query-attention.html` ("Shrink the KV cache: GQA")

---

## 2 · grouped-query-attention.html — Share K/V across query heads to shrink the cache

- **title (h1):** <span class="cyan">Grouped-query attention</span>: share the Keys and Values
- **eyebrow:** `Part 3 · Lesson 2/8`
- **objective:** Show that letting several query heads share one K/V head shrinks the KV cache by `n_heads / n_kv_heads`, walking MHA → GQA → MQA — the exact lever `kv-memory.html` named.
- **prereq:** Part-1 lesson 6 (attention: Q·K then blend V, per head) and Part-2 lesson 3 (the KV cache stores K/V per token *per head*).
- **prose beats:**
  1. In the attention you built (Part-1 lesson 6), every **query head** had its own **Key** and **Value** — that's MHA. The KV cache (Part-2 lesson 3) stores one K and one V per token *per head*, so its size scales with the number of KV heads.
  2. **GQA** keeps all 32 query heads but lets groups of them **share** a smaller set of K/V heads. 32 query heads over 8 K/V heads → 4 queries share each K/V → the cache is **4× smaller**. The queries still differ; only the K/V are pooled.
  3. **MQA** is the limit: one K/V head for all 32 query heads → **32×** smaller cache, a little more quality loss. Modern models (Llama-3) sit at GQA-8: most of the memory win, almost none of the quality cost.
- **THE WIDGET — computational spec (dial the KV heads, recompute the cache):**
  - **Inputs:** a segmented toggle over the four `GQA.configs` (**MHA 32 · GQA-8 · GQA-4 · MQA 1**) and a **sequence-length** slider (512…131072, stepped like Part-2 kv-memory). Optional batch slider.
  - **Computed live:** `perTok = gqaKvBytesPerToken(MODEL, n_kv, 2)`; `total = gqaKvCacheBytes(MODEL, n_kv, seq, batch, 2)`; `shrink = gqaShrink(MODEL, n_kv)`; `sharedBy = MODEL.n_heads / n_kv`. **Recompute, not replay** — every toggle/slider re-runs the formula.
  - **What updates on screen:** (1) a small head-grouping diagram — 32 query-head dots bracketed into `n_kv` groups, each group pointing at one K/V head — that regroups as you toggle; (2) a `.tank`/bar comparing the KV cache at the current config against the MHA baseline (cyan memory badge), with the 80 GB `.gpu-line` when seq is large; (3) `.reads`: **KB/token**, **total cache GB**, **shrink vs MHA (×)**, **query heads per K/V**. At GQA-8, seq 8192: **128 KiB/token, 1.07 GB, 4×, 4 queries/KV.**
  - **Formula on screen:** `kv/token = 2 × layers × n_kv_heads × head_dim × bytes` · `shrink = n_heads / n_kv_heads`.
- **under-the-hood (.hood):** per-token KV (fp16, 32 layers, head_dim 128): MHA `2·32·32·128·2 = 512 KiB`; GQA-8 `128 KiB` (÷4); GQA-4 `64 KiB` (÷8); MQA `16 KiB` (÷32). At seq 8192, batch 1: 4.0 / 1.0 / 0.5 / 0.13 GiB. "Only the K and V are shared — each query head still computes its own Query and its own attention weights, so the model keeps most of its expressiveness. This is why GQA is nearly free memory: the thing that grows the cache (K/V per head, Part-2 lesson 3) is exactly the thing GQA pools."
- **myth / reality:** *Myth* — "Fewer KV heads means fewer attention heads — a smaller, weaker model." / *Reality* — "The **query** heads are unchanged; only the **Key/Value** heads are shared. You lose a little precision in what's attended to, and shrink the cache up to 32×."
- **self-check:** *"GQA with 32 query heads and 8 KV heads shrinks the KV cache by…"* — (a) 8× *(wrong)*; **(b) 4× — the cache scales with KV heads, and 32/8 = 4** *(correct)*; (c) not at all *(wrong)*. **Why:** the KV cache stores K/V per KV head; sharing 32 query heads over 8 KV heads stores 4× fewer K/V.
- **prev / next:** prev: `gpu-anatomy.html` · next: `flash-attention.html` ("Never write the n×n matrix")

---

## 3 · flash-attention.html — Tile attention so the n×n matrix never touches VRAM

- **title (h1):** <span class="violet">FlashAttention</span>: keep the score matrix on-chip
- **eyebrow:** `Part 3 · Lesson 3/8`
- **objective:** Show that naive attention writes an `n×n` score matrix to VRAM, and FlashAttention tiles the work so K/V blocks stay in on-chip SRAM and that matrix is never materialized — the win is memory movement, not FLOPs.
- **prereq:** Part-1 lesson 6 (the `q·k` scores over every token — the n×n matrix), lesson 1 here (SRAM vs VRAM), and Part-2 lesson 8 (bandwidth, not math, is the wall).
- **prose beats:**
  1. Attention (Part-1 lesson 6) scores every token's Query against every token's Key — for `n` tokens that's an `n×n` matrix of scores. Naive attention **writes that whole matrix to VRAM**, reads it back to softmax, reads it again to blend V. At n=8192 that's a 128 MiB matrix moved several times — *per head, per layer*.
  2. FlashAttention never builds it. It cuts Q, K, V into **tiles** small enough to fit in on-chip **SRAM** (lesson 1), and computes attention one tile-pair at a time, keeping a **running max and running sum** (an **online softmax**) so each new tile corrects the partial answer without ever seeing the whole row.
  3. The output is **bit-identical** to naive attention. The only thing that changed is *where the numbers live* — the huge matrix stays on-chip and is thrown away tile by tile. It's the roofline lesson applied to attention: stop moving `O(n²)` bytes through VRAM.
- **THE WIDGET — computational spec (tile the matrix, watch VRAM traffic vanish):**
  - **Inputs:** a **sequence-length** slider `n` (256…16384) and a **tile size** slider `B` (16…256, default 64). Optional fp16/int8 toggle for `bytes`.
  - **Computed live:** `matrixBytes = scoreMatrixBytes(n)`; `tileBytes = flashTileBytes(B)`; `fits = fitsInSram(tileBytes)`; `tilesAxis = flashTilesPerAxis(n,B)`; `reduction = flashMemReduction(n,B)`; `kvPasses = flashKvPasses(n,B)`. **Recompute, not replay** — sliders re-run every figure; if `B` gets too big, `tileBytes` exceeds 192 KiB and a rose "tile won't fit SRAM" badge appears.
  - **What updates on screen:** (1) an `n×n` grid schematic with a single `B×B` tile highlighted, sweeping across it (the tile is the only thing "in SRAM"; the rest is greyed "never in VRAM"); (2) two bars — **naive VRAM for scores** (cyan, huge) vs **Flash SRAM tile** (tiny) — with the reduction factor between them; (3) `.reads`: **score matrix (naive, VRAM)**, **one tile (Flash, SRAM)**, **memory reduction ×**, **K/V passes**. At n=4096, B=64: **32 MiB → 56 KiB, 4096× smaller, 64 passes, fits SRAM ✓.**
  - **Formula on screen:** `naive scores = n² × bytes  (→ VRAM)` · `Flash tile = (3·B·d + B²) × bytes  (→ SRAM)` · `reduction = (n/B)²`.
- **under-the-hood (.hood):** head_dim d=128, fp16. n=4096: naive `4096²·2 = 32 MiB` per head written to VRAM; Flash tile B=64 `(3·64·128 + 64²)·2 = 56 KiB` in SRAM, `(4096/64)² = 4096` tile-pairs, K/V streamed 64 times. "The online softmax is the trick that makes tiling exact: keep a running max `m` and running denominator `ℓ`; when a new tile arrives, rescale the old partial by `e^(m_old − m_new)` and add the new contribution. No tile ever needs the full row, so the `n×n` matrix is never assembled. Same output as Part-1 lesson 6 — only the memory schedule changed (Part-2 Q18)."
- **myth / reality:** *Myth* — "FlashAttention is a faster attention *algorithm* — fewer multiplications." / *Reality* — "It does the **same** FLOPs and gives the **same** answer. It just never writes the `n×n` score matrix to VRAM — it's an IO trick, memory movement not math."
- **self-check:** *"What does FlashAttention avoid?"* — (a) computing the Q·K dot products *(wrong)*; **(b) writing the full n×n score matrix to VRAM — it tiles the work in on-chip SRAM instead** *(correct)*; (c) the softmax step *(wrong)*. **Why:** the scores are still computed and softmaxed, but tile by tile in SRAM; the quadratic matrix never lands in slow VRAM.
- **prev / next:** prev: `grouped-query-attention.html` · next: `mixture-of-experts.html` ("Route each token to a few experts")

---

## 4 · mixture-of-experts.html — A router sends each token to top-k of N experts

- **title (h1):** <span class="rose">Mixture-of-experts</span>: many experts, a few per token
- **eyebrow:** `Part 3 · Lesson 4/8`
- **objective:** Show a learned router sending each token to its top-k of N expert FFNs, so total params grow with N but FLOPs/token stay near a small dense model — active vs total params.
- **prereq:** Part-1 lesson 9 (the FFN — one expert *is* an FFN), lesson 6 (router = dot-product + softmax again), Part-2 lesson 8 (FLOPs track active params).
- **prose beats:**
  1. A dense model runs its one feed-forward network (Part-1 lesson 9) for **every** token. An MoE keeps **N** separate expert FFNs and adds a tiny **router** — a gate that scores the token's vector against each expert (a dot product then softmax, Part-1 lesson 6 again) and picks the **top-k**.
  2. Only those k experts run — **sparse activation**. So you *store* N experts (total params grow with N) but only *compute* k of them per token (FLOPs, and the roofline of Part-2, track the **active** params).
  3. With 8 experts and top-2, a model can hold **47B** parameters yet run each token through only **~13.6B** of them — the memory of a big model at the speed of a small one. Different tokens fire different experts, and training **load-balances** so no expert goes dead.
- **THE WIDGET — computational spec (route a real token, then size the model):**
  - **Inputs:** a **token picker** (the Part-1 chips: `The cat sat because it was tired`) and a **top-k** control (1/2/3, default 2). Optional **N experts** control for the param math.
  - **Computed live:** `moeRoute(E[token], k)` → the 8 router logits, the top-k experts and their renormalized gate weights (reuse `matvec` + `softmax`); `moeParams()` → total vs active params and `activeFrac`. **Recompute, not replay** — pick a different token and the lit experts change (`cat`→e0+e4, `sat`→e1+e4, `tired`→e2+e5, `it`→e0+e5); change k and both the routing and the active-param bar move.
  - **What updates on screen:** (1) a row of 8 expert tiles (labeled animacy/action/state/… ) with the router logit on each; the top-k light up rose with their gate %, the rest dim; (2) a two-bar **active vs total params** meter (rose) with the % active; (3) `.reads`: **experts fired**, **active params**, **total params**, **% active**. Default `it`, k=2: **e0 (animacy) + e5 (animate+state), 13.6B active / 47.4B total, 29% active.**
  - **Formula on screen:** `router = softmax(top-k of  gate · embedding)` · `active = shared + k·expert` · `total = shared + N·expert`.
- **under-the-hood (.hood):** router for `cat [1,0.2,0.1,0]`: logits `[3.0, 0.6, 0.3, 0.0, 1.8, 1.65, 0.65, −1.3]` → top-2 e0 (animacy, 3.0) and e4 (animate+action, 1.8) → gate `softmax([3.0,1.8]) = 76.9% / 23.1%`. Params (8 experts, top-2, Mixtral-shaped): one expert FFN `= 3·4096·14336 = 176.2M`; total `= embed 131M + 32·(attn 67.1M + router 0.03M + 8·176.2M) = 47.4B`; active `= 131M + 32·(67.1M + 0.03M + 2·176.2M) = 13.6B` → **29% active** (raw expert fraction 2/8 = 25%, plus the always-on attention/embeddings). "The router is a *learned* matrix — it decides routing, and a load-balancing loss during training keeps tokens spread so every expert earns its keep. FLOPs/token ≈ 2·active params (Part-2 lesson 8), so the roofline sees a 13.6B model even though 47.4B sit in VRAM."
- **myth / reality:** *Myth* — "An 8-expert model is 8× more compute per token." / *Reality* — "It's ~2 experts' worth of compute per token (top-2), but 8 experts' worth of *storage*. Total params scale with N; **FLOPs scale with k**."
- **self-check:** *"In an 8-expert, top-2 MoE, per token the model…"* — (a) runs all 8 experts and averages them *(wrong)*; **(b) runs only the 2 experts the router picks — sparse activation — so FLOPs stay near a small dense model** *(correct)*; (c) runs a different single expert each layer at random *(wrong)*. **Why:** the router selects top-k experts; only those compute, so active params (and FLOPs) are far below the total stored.
- **prev / next:** prev: `flash-attention.html` · next: `tool-use.html` ("A predictor that calls an API")

---

## 5 · tool-use.html — A next-token predictor emits a call; the result is injected; generation resumes

- **title (h1):** <span class="amber">Tool use</span>: it's still just predicting tokens
- **eyebrow:** `Part 3 · Lesson 5/8`
- **objective:** Show that "calling a tool" is the model emitting a structured call as ordinary tokens; the harness pauses, actually runs the function, injects the result as tokens, and generation resumes. The model never runs anything itself.
- **prereq:** Part-1 lesson 1 (next-token prediction, looped) and lesson 14 (hallucination — a tool injects the ground truth frozen weights lack).
- **prose beats:**
  1. The model can't fetch the weather or run code — it only predicts the next token (Part-1 lesson 1). What it *learned* is to emit a **call-shaped token sequence** when a task needs one: a little JSON blob like `{"tool":"get_weather","args":{"city":"Paris"}}`.
  2. A **harness** wrapped around the model watches the stream. The moment a complete call appears, it **stops generation**, **actually runs** the function (this is the harness's code, not the model's), and **pastes the result back into the context as new tokens**.
  3. Now the model resumes with the result sitting in its context — and predicts an answer that uses it. The "magic" is just: predict call → pause → external code runs → inject result tokens → keep predicting. Same loop, one interruption.
- **THE WIDGET — computational spec (step the stream through a real tool call):**
  - **Inputs:** a **city picker** (`Paris · Tokyo · Cairo · Oslo`, from `TOOLUSE.cities`) and a **Step ▸ / Run** control (reuse the flagship `.stg` staged-reveal + `btnStep/btnRun` pattern). Five phases: *(1) model reasons → (2) model emits the call tokens → (3) **PAUSE** — harness runs the tool → (4) result tokens injected → (5) model resumes the answer.*
  - **Computed/stepped live:** the stream advances token by token from `TOOLUSE.reason` + `callTokens` (with `<CITY>` substituted from the picker); at the pause the widget calls `runWeather(city)` and builds the injected tokens via `toolResultTokens(city)` and the final answer via `toolAnswer(city)`. **Recompute, not replay** — change the city and the call arg, the injected JSON, and the resumed sentence all change (`Tokyo` → `{"tempC":24,"sky":"clear"}` → "It's 24°C and clear in Tokyo right now."). The phase is derived from the step index, not hard-coded.
  - **What updates on screen:** a single growing "context" strip of token chips, color-coded by phase — amber (model-generated), then a violet **PAUSED · running get_weather(city)** banner, then cyan **injected** result chips (visually distinct: "this text came from the tool, not the model"), then amber resumed answer. A `.reads`-style status line: **phase**, **who's acting now (model ⟷ harness)**, **tokens so far**. Caption: "the model wrote the call; your machine ran it; the model never left the token loop."
  - **Formula on screen (conceptual):** `predict … → [complete call detected] → PAUSE → harness runs fn → inject result tokens → predict …`
- **under-the-hood (.hood):** the call `{"tool":"get_weather","args":{"city":"Tokyo"}}` is just tokens the model was trained to produce; a stop-condition on the closing brace ends the model's turn. The harness parses it, runs `runWeather('Tokyo') → {tempC:24, sky:'clear'}`, and appends `{"tempC":24,"sky":"clear"}` to the context as tokens. Generation resumes and predicts `It's 24°C and clear in Tokyo right now.` "Nothing in the model executed code — it emitted call-shaped tokens and later read result-shaped tokens. This is also the fix for Part-1 lesson 14: the tool injects a *true* fact the frozen weights never stored."
- **myth / reality:** *Myth* — "The model reaches out and runs the API itself." / *Reality* — "The model only predicts tokens. It emits a call, an external harness runs it and pastes the result back as tokens, and the model keeps predicting. It never executes anything."
- **self-check:** *"How does a text-only predictor 'call' a weather API?"* — (a) it opens a network connection during inference *(wrong)*; **(b) it emits a structured call as tokens; a harness runs the function and injects the result back into the context as tokens** *(correct)*; (c) the API is baked into its weights *(wrong)*. **Why:** tool use is predict-a-call → pause → external code runs → inject result → resume; the model stays inside the next-token loop the whole time.
- **prev / next:** prev: `mixture-of-experts.html` · next: `reasoning.html` ("Spend tokens to think")

---

## 6 · reasoning.html — Chain-of-thought: spend tokens to think, accuracy climbs then plateaus

- **title (h1):** <span class="green">Reasoning</span>: buying accuracy with tokens
- **eyebrow:** `Part 3 · Lesson 6/8`
- **objective:** Show that spending more "thinking" tokens before answering raises accuracy (test-time compute), with diminishing returns, and that reasoning helps hard problems far more than easy ones.
- **prereq:** Part-1 lesson 1 ("the math for one token is trivial" — reasoning exploits exactly that) and Part-2 lesson 1 (each thinking token costs one decode step).
- **prose beats:**
  1. Part-1's myth-buster said one token's math is trivial — the model isn't "thinking hard" on any single step. Reasoning models turn that into a strategy: emit **more tokens** as a visible scratchpad *before* the final answer, and each cheap step conditions the next.
  2. This is **test-time compute**: instead of a bigger model, spend a longer generation. Accuracy rises with thinking length — then **plateaus**: the first hundred tokens buy a lot, the next hundred buy little.
  3. It isn't free. Every thinking token is one **decode step** (Part-2 lesson 1) — latency for correctness. And it pays off most on **hard** problems; easy ones were already right, so there's little to gain.
- **THE WIDGET — computational spec (a live accuracy-vs-thinking-length curve):**
  - **Inputs:** a **thinking-tokens** slider `t` (0…600) and a **problem difficulty** toggle (`easy · typical · hard`, from `REASON.presets`).
  - **Computed live:** `reasonAcc(t, preset)` for the operating point, the whole curve `t=0…600` for the plot, `reasonMarginal(t, 60, preset)` for the marginal gain of the last 60 tokens, and a cost readout `t` decode-steps (optionally `t × decode_ms` reusing Part-2's `weightBytes/GPU.bandwidth`). **Recompute, not replay** — dragging `t` moves the dot along the computed curve; switching difficulty redraws the whole curve from the formula.
  - **What updates on screen:** (1) a line plot of accuracy vs thinking tokens (green) with a moving operating-point dot and the plateau visible; (2) `.reads`: **accuracy now**, **+pts from the last 60 tokens** (shrinks as you go right), **thinking cost (decode steps / ms)**; (3) a caption contrasting difficulty: "on an easy problem the curve starts at 70% and barely rises; on a hard one it climbs from 15% to 76% — reasoning earns its cost only when the problem is hard." Default typical, t=120: **64.8%, +7.9 pts from the last 60, 120 decode steps.**
  - **Formula on screen:** `accuracy(t) = floor + (cap − floor)(1 − e^(−t/τ))` · `cost = t decode steps` · marginal gain → 0.
- **under-the-hood (.hood):** typical `{floor 0.30, cap 0.85, τ 120}`: t=0 → 30%, t=60 → 51.6% (+21.6), t=120 → 64.8% (+13.2), t=240 → 77.6%, t=480 → 84.0%, t=600 → 84.6% (cap 85%). Marginal per 60 tokens: +21.6 (first) → +7.9 (at 180) → +0.4 (at 540) — clear diminishing returns. Hard `{0.15, 0.80, 200}`: 15% → 76% over 600 tokens. Easy `{0.70, 0.95, 60}`: 70% → 86% by 60 tokens. "This is a smooth stand-in for the real, noisier empirical curves — but the shape (fast then flat, hard ≫ easy) and the mechanism (each extra token is another cheap forward pass) are real. Some tasks even *dip* if the model overthinks; the plateau is where you stop."
- **myth / reality:** *Myth* — "A reasoning model is a smarter architecture that thinks before it speaks." / *Reality* — "It's the same next-token predictor spending **more tokens** as a scratchpad. Accuracy scales with generation length, up to a plateau — you're buying compute at test time, not a new brain."
- **self-check:** *"Why does letting a model 'think' longer raise accuracy?"* — (a) it retrains its weights on the problem *(wrong)*; **(b) each extra token is another cheap forward pass that conditions the next, so more thinking-tokens = more test-time compute** *(correct)*; (c) it lowers the temperature *(wrong)*. **Why:** reasoning spends generation length (decode steps) as compute; accuracy climbs with thinking tokens and then plateaus, most on hard problems.
- **prev / next:** prev: `tool-use.html` · next: `rag.html` ("Inject knowledge by retrieval")

---

## 7 · rag.html — Frozen weights, so inject knowledge as retrieved context

- **title (h1):** <span class="cyan">RAG</span>: retrieve the facts, then prompt with them
- **eyebrow:** `Part 3 · Lesson 7/8`
- **objective:** Show that because weights are frozen, new knowledge is injected as retrieved context — retrieval is embedding similarity (Part-1 cosine) — and place RAG against fine-tuning and prompting.
- **prereq:** Part-1 lesson 2 (cosine similarity between embedding vectors — retrieval *is* this) and lesson 14 (no truth signal → retrieval supplies one); the frozen weights (Part-1 "train vs run").
- **prose beats:**
  1. The weights are **frozen** after training (Part-1) — a running model can't learn a new fact. Three ways to get knowledge in: **prompting** (paste the text yourself), **fine-tuning** (retrain the weights — permanent, costly, then frozen again), and **RAG**.
  2. **RAG** = automated prompting. Keep your documents as **embedding vectors** in a library; at query time, embed the question and **retrieve** the closest passages by **cosine similarity — the exact operation of Part-1 lesson 2** — then **prepend** them to the prompt. The model reads them as ordinary context.
  3. So retrieval quality gates answer quality: if the top match is strong, the model answers from real text; if nothing's close, RAG injects nothing and you're back to the frozen weights guessing (Part-1 lesson 14). Prompting for one-off facts, RAG for a changing knowledge base, fine-tuning for changing behaviour.
- **THE WIDGET — computational spec (pick a query, cosine-retrieve, watch it prepend):**
  - **Inputs:** a **query picker** (the three `RAG.queries`, plus optionally a "type your own" that maps to one of the query vectors) and a **top-N** control (1/2/3, default 2).
  - **Computed live:** `ragRetrieve(query.vec, topN)` — cosine of the query against all 6 passages (reuse `cosine`), sorted, top-N kept. **Recompute, not replay** — every query re-runs the cosine loop and re-ranks; the retrieved passages and the assembled prompt change (cats→P0, onions→P2, planet→P3).
  - **What updates on screen:** (1) the 6 passages as cards, each showing its live cosine to the current query as a bar (cyan), the top-N highlighted; (2) an assembled **prompt** panel that visibly **prepends** the retrieved passage text above the user's question ("Context: <P2 text> … Question: <query>"); (3) `.reads`: **top match id + cosine**, **passages injected**, and a **retrieval-confidence** note (strong ≥ ~0.9 vs weak). Default "cook onions", top-2: **P2 (cos 1.00) + P5 (0.99) prepended.**
  - **Formula on screen:** `retrieve = top-N by cos(query, passage)` (Part-1 lesson 2) · `prompt = retrieved text + question`.
- **under-the-hood (.hood):** query "biggest planet" `[0,0.1,0,1]` vs passages: P3 Jupiter cos `1.02/(√1.01·√1.04) = 0.995` (top), P1 storm `0.198`, rest ~0. "cats sleep" → P0 (0.995); "cook onions" → P2 (1.00), P5 (0.989). "Retrieval is literally Part-1 lesson 2's cosine over a document library — an embedding model turns each passage into a vector once, offline; at query time you embed the question and rank. Fine-tuning would instead bake facts into the weights (and re-freeze them); prompting is you doing the retrieval by hand. RAG shines when the knowledge changes faster than you can retrain, and it's the honest fix for hallucination (lesson 14): it gives the softmax real text to lean on instead of plausibility."
- **myth / reality:** *Myth* — "RAG teaches the model new knowledge." / *Reality* — "The weights never change. RAG **retrieves** relevant text by embedding similarity and **pastes it into the prompt** — the model reads it as context, same as if you'd typed it."
- **self-check:** *"In RAG, how are the right documents found?"* — (a) the model's weights are updated with them *(wrong)*; **(b) by embedding similarity — cosine between the query vector and each passage vector (Part-1 lesson 2) — then the top matches are prepended to the prompt** *(correct)*; (c) by keyword exact-match only *(wrong)*. **Why:** retrieval ranks passages by cosine similarity in embedding space and injects the top ones as context; the frozen weights are untouched.
- **prev / next:** prev: `reasoning.html` · next: `multimodal.html` ("Images become tokens too")

---

## 8 · multimodal.html — Images become tokens the same transformer eats

- **title (h1):** <span class="amber">Multimodal</span>: an image is just more tokens
- **eyebrow:** `Part 3 · Lesson 8/8`
- **objective:** Show that an image is cut into patches, each patch projected into an embedding vector in the same space as text tokens, and fed into the same transformer stream — tokens aren't only text.
- **prereq:** Part-1 lesson 1 (a token → an embedding vector), lesson 13 (position added to each embedding), lessons 5–7 (attention over the vector stream — it doesn't care where the vectors came from).
- **prose beats:**
  1. The transformer only ever ate a **list of embedding vectors** (Part-1 lesson 1). Nothing about that list is text-specific. So to feed it an image, you just need to turn the image into embedding vectors.
  2. Cut the image into fixed **patches** (a grid of small squares). **Flatten** each patch's pixels and **project** them through one weight matrix (a single matvec, like Part-1's) into a vector the same size as a text embedding. Add a **position encoding** for where the patch sits (Part-1 lesson 13).
  3. Now append those patch-vectors to the token stream. Attention (Part-1 lessons 5–7) blends image-patches and text-tokens with the **same** math — it can't tell them apart. Audio becomes frames the same way. Tokens are whatever you can turn into a vector.
- **THE WIDGET — computational spec (click a patch, watch it become a vector):**
  - **Inputs:** a small clickable **4×4 image** (`IMG`, each cell a brightness the reader can bump up/down) and a **patch selector** (the 4 patches). Optional "add position" toggle (Part-1 lesson 13 tie-in).
  - **Computed live:** `patchesOf(IMG, 2)` cuts the (editable) image into 4 patches; for the selected patch, `patchEmbed(pixels) = matvec(WPATCH, pixels)` (reuse `matvec`), then `+ PATCH_POS[i]` if position is on. **Recompute, not replay** — edit a pixel or pick a different patch and its flattened pixels and 4-number embedding recompute (diagonal patches → `[0.5,0,0,0.8]`, dark patches → `[0.1,0,0,0]`).
  - **What updates on screen:** (1) the 4×4 image with a 2×2 patch grid overlay, the selected patch outlined; (2) the selected patch's 4 pixels shown flattening into a row, then the `matvec` producing a 4-dim embedding (labeled brightness/LR-edge/TB-edge/diagonal); (3) a **token stream** strip showing the 4 patch-embeddings (cyan) sitting alongside a couple of text-token embeddings (amber) — visibly the same shape — with a caption "attention (Part-1 lessons 5–7) eats this whole row identically." Default patch (0,0): pixels `[0.9,0.1,0.1,0.9]` → embedding `[0.5, 0.0, 0.0, 0.8]`.
  - **Formula on screen:** `patch_embedding = W_patch · flatten(pixels) + position` (Part-1 lesson 1 + lesson 13).
- **under-the-hood (.hood):** toy: `IMG` 4×4, 2×2 patches → 4 patches. Patch (0,0) `[0.9,0.1,0.1,0.9]` → `[0.25·2.0, 0, 0, 0.8] = [0.5, 0, 0, 0.8]` (bright + diagonal); off-diagonal patches `[0.1,0.1,0.1,0.1] → [0.1,0,0,0]` (dark). Real anchor: a ViT cuts a **224×224** image into **16×16** patches → `(224/16)² = 196` patches → 196 image-tokens, each projected to `d_model` (e.g. 4096) and position-encoded, then concatenated with text tokens. "The projection matrix `W_patch` is *learned* just like the embedding table (Part-1 lesson 1) — it's the image's tokenizer. After that step there is no 'image' and no 'text', only vectors, and the attention you built in Part 1 runs unchanged."
- **myth / reality:** *Myth* — "A multimodal model has a separate vision brain bolted onto the language model." / *Reality* — "The image is turned into embedding vectors (patch → project → position) and dropped into the **same** token stream. One transformer, one attention — it just sees more vectors."
- **self-check:** *"How does a language transformer 'see' an image?"* — (a) a separate CNN answers and passes text over *(wrong, not the token story)*; **(b) the image is cut into patches, each projected into an embedding vector in the same space as text tokens, and fed into the same transformer stream** *(correct)*; (c) it reads a text caption only *(wrong)*. **Why:** patches become embeddings (project + position, Part-1 lessons 1 & 13) and join the token stream; attention treats image-patches and text-tokens identically.
- **prev / next:** prev: `rag.html` · next: `index.html` (label "Back to the map — you've gone beyond the basics")

---

# D. HUB (index.html) CHANGES

Add a third lesson section below the Part-2 list, extend the intro to mention three parts, and add the 8 Part-3 cards. Reuse the existing `.lesson-card` markup verbatim; only content changes. **Insert the new `#part3` section AFTER the `#part2` `</section>` and BEFORE the classic-walkthrough `<section>`.**

### D0 · Intro + framing
- **Meta description** (`<head>`): append a Part-3 clause, e.g.: *"…Then Part 3 · Beyond the Basics opens the systems internals and capability patterns — GPU anatomy, GQA, FlashAttention, mixture-of-experts, tool use, reasoning, RAG, and multimodal — every widget still computing live."*
- **Header `<p class="lede">`:** after the existing Part-2 sentence, add: *"Then <span class="strong-ink">Part 3 · Beyond the Basics</span> goes past the core loop — the GPU it runs on, the attention and memory tricks (GQA, FlashAttention), and the capability patterns every assistant uses (mixture-of-experts, tool use, reasoning, RAG, multimodal)."*
- **Pill row:** add a third pill after the Part-2 pill: `<span class="meta-pill">🧩 <b>Part 3 · 8 lessons</b></span>`. (Keep `📚 Part 1 · 15 lessons` and `⚡ Part 2 · 8 lessons`.)
- **Anchor:** give the new section `id="part3"` (the Part-3 crumbs link `index.html#part3`).

### D1 · New "Part 3" section (8 cards)

Mirror the Part-2 markup exactly (`<section class="block wrap reveal" id="part3">` → heading + kicker → `<ol class="lesson-list">` → `.lesson-card` items). Numbers `01`–`08` restart for Part 3; color dot per the canon; blurb = "what you'll compute/see".

```html
<section class="block wrap reveal" id="part3" style="padding-top:10px">
  <h2 style="font-size:clamp(1.5rem,3.4vw,2.1rem);margin-bottom:6px">Part 3 · <span class="amber">Beyond the Basics</span></h2>
  <p class="prose" style="margin-bottom:8px">Past the core loop: the GPU it runs on, the attention and memory tricks that scale it, and the capability patterns every assistant is built from — each one a live widget, every number still computed in front of you.</p>
  <ol class="lesson-list">
    <!-- 8 cards below -->
  </ol>
</section>
```

Card list (dot color per canon; blurb = what you'll compute/see):

| # | href | dot | lc-ttl | lc-blurb (what you'll compute/see) |
|---|------|-----|--------|------------------------------------|
| 01 | gpu-anatomy.html | **cyan** | What a GPU actually is | Place weights, KV, and a Flash tile in the VRAM→SRAM hierarchy and see what fits. |
| 02 | grouped-query-attention.html | **cyan** | Share the Keys and Values (GQA) | Dial 32→8→1 KV heads and watch the cache shrink 4× → 32×. |
| 03 | flash-attention.html | **violet** | Keep the score matrix on-chip | Tile the n×n scores into SRAM; watch 32 MiB of VRAM traffic vanish. |
| 04 | mixture-of-experts.html | **rose** | Many experts, a few per token | Route a real token to its top-2 experts; 47B stored, 13.6B active. |
| 05 | tool-use.html | **amber** | It's still just predicting tokens | Step a token stream that emits a JSON call, runs it, injects the result. |
| 06 | reasoning.html | **green** | Buying accuracy with tokens | Slide thinking-tokens and watch accuracy climb, then plateau. |
| 07 | rag.html | **cyan** | Retrieve the facts, then prompt | Cosine-retrieve the top passages for a query and watch them prepend. |
| 08 | multimodal.html | **amber** | An image is just more tokens | Click a patch and watch it project into an embedding in the token stream. |

Each card is exactly the Part-2 shape, e.g.:
```html
<li>
  <a class="lesson-card" href="gpu-anatomy.html">
    <span class="lc-num">01</span>
    <span class="lc-dot cyan" aria-hidden="true"></span>
    <span class="lc-body">
      <span class="lc-ttl">What a GPU actually is</span>
      <span class="lc-blurb">Place weights, KV, and a Flash tile in the VRAM→SRAM hierarchy and see what fits.</span>
    </span>
    <span class="lc-arrow" aria-hidden="true">→</span>
  </a>
</li>
```
(Optional "the stage" flag on 01, since it grounds all of Part 2: `<span class="flag">the stage</span>` inside `.lc-ttl`, mirroring Part-1's `flagship` / Part-2's `the why`.)

### D2 · Color legend (no change required)
The existing hub legend already maps amber/cyan/violet/rose/green and Part 3 introduces no new colors. Leave as-is.

### D3 · Classic-walkthrough block (no change required)
Part 3 has no monolith counterpart, so the classic-walkthrough block is unchanged. (Optionally, one line under its `<p class="prose">`: *"Part 3 · Beyond the Basics has no classic tour — it's interactive-only."*)

---

# Builder checklist (no further decisions needed)

- New file `tools/course/course-data-p3.js` created verbatim from Section B; it reuses p1 (`E/TOKENS/matvec/dot/cosine/softmax`) and p2 (`MODEL/GPU/paramCount/weightBytes/kvCacheBytes/BYTES`); every widget computes from its helpers (recompute/step, not replay); Section-B verification numbers reproduce to the stated decimals.
- 8 new pages cloned from `_template.html`, in `tools/course/`, wired `course-data.js` → `course-data-p2.js` → `course-data-p3.js` → `course.js` → page script; eyebrow `Part 3 · Lesson N/8`; crumbs link `index.html#part3`.
- The four less-numeric pages (tool-use, reasoning, rag, multimodal) are genuinely interactive/steppable and mechanism-revealing — no page is prose-only.
- Every page ties back to a named earlier lesson (Part-1/Part-2), as specified per-page.
- Pager chain intact: `index → gpu-anatomy → grouped-query-attention → flash-attention → mixture-of-experts → tool-use → reasoning → rag → multimodal → index`.
- Hub gains a `#part3` section of 8 cards (Section D), intro/description/pills mention three parts.
```
