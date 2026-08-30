# Build Spec — Part 2
## "How LLMs Run Fast" — decomposing the inference monolith into 8 compute-live lessons

**For:** coding agents building directly from this spec.
**Design system (do not restyle):** `tools/course/course.css`, `tools/course/course.js`, clone `tools/course/_template.html` for every new page. All 8 files live in `tools/course/` beside the Part-1 pages so `prev`/`next` relative links resolve. Load order on every page: `<script src="course-data.js"></script>` → `<script src="course-data-p2.js"></script>` → `<script src="course.js" defer></script>` → the page's own `<script>`.
**The bar to hit (unchanged from Part 1):** every widget must **COMPUTE its result live from the shared constants on each interaction — recompute, never replay.** No headline number may be typed as a constant in the HTML; it must fall out of a formula the reader can trace, driven by the sliders. Part 2 widgets are mostly **formula widgets** (FLOPs, bytes, tokens/sec) rather than the Part-1 toy vectors. The reference bar for "a widget that computes" is still `attention-scoring.html` (staged, traceable arithmetic).

**Two monolith widgets are already real and must be PORTED, not rebuilt:** the **quantization mapper** (`llm-inference-efficiency.html` §03, lines 666–745 — real absmax rounding onto levels, per-group scales, outlier protection) → page 4, and the **speculative-decoding animation** (§06, lines 842–915 — real accept-run logic, k selector, draft-quality toggle, race bars) → page 7. Port their JS into the shared design system and drive their headline readouts from `course-data-p2.js` helpers; keep the visuals.

**Color canon (reuse consistently; from course.css):** amber = tokens/**structure** · cyan = vectors/embeddings/**memory** · violet = attention/**compute** · rose = **weights**/parameters · green = training/probabilities/**throughput**.

**Owner's rule:** ONE core idea + ONE widget per page. Do not merge pages. Do not add pages. Filenames, order, and eyebrow numbers are LOCKED (below). This is a **new Part 2 sub-track** — do NOT renumber Part 1.

**Locked sequence (eyebrow reads `Part 2 · Lesson N/8`):**
```
1. prefill-decode.html          prev: index.html
2. kv-cache.html
3. kv-memory.html
4. quantization.html            (PORT the monolith quant widget)
5. paged-attention.html
6. continuous-batching.html
7. speculative-decoding.html    (PORT the monolith spec-decode widget)
8. roofline.html                next: index.html
```
Page 1 `prev` → `index.html`; page 8 `next` → `index.html`. Internal pages chain the sequence above. Crumbs on every page: `Course / Part 2 · How LLMs Run Fast / NN · title`, with the middle crumb linking `index.html#part2`.

---

# A. CONCEPT CANON — Part 2 additions

Plain language, define each term the first time a page needs it, then reuse the phrasing. These continue the Part-1 canon; where a term ties back to Part 1, say so out loud — **the weights are the frozen `E / Wq / Wk / Wv / W1 / W2` from Part 1; decode must stream ALL of them out of memory to take one step.**

- **prefill vs decode** *(pg 1)* — running a model splits in two. **Prefill** reads your whole prompt in **one parallel pass**: every prompt token is processed at once, so the GPU's math units are the bottleneck (**compute-bound**). **Decode** then writes the answer **one token at a time**: to produce each new token the GPU must stream *every* weight out of memory, do a tiny amount of math, and emit one token — then do it all again for the next. Decode is sequential and dominates wall-clock. Say "one parallel pass" (prefill) and "one token at a time" (decode).

- **memory-bandwidth-bound vs compute-bound** *(pg 1, deepened pg 8)* — a step is **compute-bound** when the GPU's math units are the limit (lots of arithmetic per byte fetched — prefill), and **memory-bandwidth-bound** when the limit is how fast weights can be *moved* from memory to the math units (little arithmetic per byte — decode at batch 1). The physical fact: for one decode token the math is trivial; the cost is *reading all the weights once*.

- **KV cache — what is cached, and why not Q** *(pg 2)* — in attention (Part-1 lesson 6) every new token's **Query** is dotted against every earlier token's **Key**, then blends every earlier token's **Value**. Earlier tokens' **Keys and Values never change** as you generate, so compute each token's K and V **once** and store them (the *KV cache*), keyed **per token, per layer, per attention head**. The new token computes only its own K and V and reads the rest. **Q is not cached** because each step you only ever need the *current* token's Query — a past token's Query is never reused. Without the cache, step *t* recomputes the K/V of all *t−1* prior tokens: the **triangular blow-up** (1+2+…+n = n(n+1)/2).

- **KV-cache memory formula** *(pg 3)* — the cache grows **linearly with sequence length** and eats VRAM:
  `kv_bytes = 2 (K and V) × layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_elem`.
  With our 7B preset (`n_kv_heads × head_dim = d_model = 4096`), that is `2 × 4096 = 8192` numbers per token per layer, `512 KiB` per token across 32 layers. Long context or many concurrent requests can make the cache **outweigh the model itself**. (Ties back to Part-1 Q15: the vector flowing through is recomputed; K/V are the *stored* part.)

- **quantization** *(pg 4)* — store each weight in **fewer bits**. `fp16` = 2 bytes, `int8` = 1 byte, `int4` = 0.5 byte per weight. Quantization maps a range of real values onto a small set of evenly spaced **levels**: pick a **scale**, round each weight to the nearest level. `int8` = 256 levels, `int4` = 16 levels. Fewer bits = less to store **and** less to move — and since decode is bandwidth-bound, fewer bits is fewer milliseconds. **Per-group scales:** one scale for a whole tensor forces every weight into the same 16 levels; a separate scale per small block (a "group") keeps precision where values are small. **Outliers:** a few giant weights stretch the range so everyone else crams into a handful of levels — protecting them (keeping the few big ones in higher precision, the GPTQ/AWQ idea) rescues the rest. The weights being quantized are exactly the frozen Part-1 matrices.

- **paged attention** *(pg 5)* — the KV cache is variable-length (you don't know a reply's length up front). The naive fix reserves each request's **maximum** cache slot in one contiguous block, so short replies leave most of it **empty = padding waste** and the pool **fragments**. **Paged attention** (the vLLM idea) chops the cache into fixed-size **pages** placed wherever there's room, like an OS's virtual memory: a sequence uses only as many whole pages as it needs, so wasted memory drops from "up to max-length per request" to "under one page per request." Same answer, pure memory management.

- **continuous batching** *(pg 6)* — one decode step reads the entire model no matter how many requests ride along, so push **many** requests through that single weight-read at once and the cost is amortized. **Throughput** = total tokens/second across all users; **latency** = how fast one reply arrives. **Static** batching launches a fixed batch and waits for the *longest* member to finish — short requests idle their slots. **Continuous** batching swaps a finished sequence out and a waiting one in **every step** (interleaving new prefills with in-flight decodes), keeping every slot busy. It multiplies throughput; it does **not** lower any single request's latency.

- **speculative decoding** *(pg 7)* — a small, fast **draft** model proposes the next *k* tokens cheaply; the big **target** model then **verifies all k in a single parallel pass** (same parallelism as prefill — verifying k tokens costs about one normal decode step). It keeps the longest prefix it agrees with and **rejects from the first disagreement**, always emitting one correct token at the break. **Acceptance rate `a`** = the chance the target agrees with a drafted token; higher `a` and longer accepted runs mean more tokens per expensive step. The output is **exactly what the target would have produced alone** — speed, not a quality change. Expected tokens per verify pass `= (1 − a^(k+1)) / (1 − a)`.

- **arithmetic intensity / roofline** *(pg 8)* — **arithmetic intensity** = FLOPs performed per byte moved from memory. A GPU has a **ridge point** = peak FLOP/s ÷ memory bandwidth (≈156 FLOP/byte on an A100); below it you're **bandwidth-bound**, above it **compute-bound**. A batch-1 decode step is a matrix-**vector** product: each weight is read once and used for ~2 FLOPs, so intensity ≈ `2 / bytes_per_param` ≈ **1 FLOP/byte at fp16** — far below 156, so decode is **bandwidth-bound**. This is the *why* under the whole part: **KV cache** avoids re-moving past work, **quantization** shrinks the bytes moved, **batching** raises intensity toward the ridge (`AI ≈ 2·batch / bytes_per_param`) — all three attack the same bottleneck: *moving weights, not math.* (Direct answer to Part-1 Q18: these change how the computation is *scheduled and stored*, not any number the model outputs.)

---

# B. SHARED DATA / HELPERS — new file `tools/course/course-data-p2.js`

Create this new file (do **not** append to `course-data.js`; Part-2 pages load both). JS-ready and hand-verified. It defines one realistic 7B-class model preset, a reference GPU, and pure helper functions every widget calls so no headline number is hard-coded. Reuses the Part-1 idea that **each number costs bytes** (kv-memory, quantization) — no Part-1 vector ops are needed here, but `softmax` from `course-data.js` remains available if a page wants a distribution.

```js
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
```

### Hand-verification (DONE — numbers below must reproduce; if a build differs, the build is wrong)

**Model preset.** `paramCount()` = embed `32000·4096 = 131,072,000` + 32·(attn `4·4096² = 67,108,864` + ffn `3·4096·11008 = 135,266,304`) = 32·202,375,168 + 131,072,000 = **6,607,077,376 ≈ 6.6B** ("7B-class"). Weight memory: fp16 **13.2 GB**, int8 **6.6 GB**, int4 **3.30 GB** → **int4 is exactly 4× smaller than fp16.** ✓

**GPU / ridge.** `GPU_RIDGE = 312e12 / 2.0e12 =` **156 FLOP/byte.** ✓

**Page 1 · prefill vs decode.** decode ms/token `= weightBytes(fp16)/BW = 1.3214e10 / 2.0e12 = ` **6.61 ms**; prefill ms/token `= flopsPerToken/peak = 1.3214e10 / 312e12 =` **0.0424 ms** → decode is **156×** slower *per token*. Whole request, prompt P=512 / gen G=512: prefill = `2·N·P/peak =` **21.7 ms**; decode = `G · 6.61 ms =` **3.38 s** → decode dominates wall-clock ~156× at equal token counts. ✓

**Page 2 · KV recompute.** generating n=2048 tokens: no cache `recomputeWork(2048) = 2048·2049/2 =` **2,098,176** token-K/V computes vs **2,048** with cache → **1024.5×** the work (ratio `(n+1)/2`). ✓

**Page 3 · KV memory.** per token `= 2·32·32·128·2 =` **524,288 bytes = 512 KiB**. At seq=4096, batch=1, fp16: `kvCacheBytes = 2,147,483,648 =` **2.0 GiB (2.15 GB)**. At seq=131,072 (128k): **68.7 GB** — larger than the 13.2 GB fp16 weights, so at long context the cache outweighs the model. ✓ (int8 KV halves each of these.)

**Page 4 · quantization.** `quantizeSym(QUANT_W, 4)`: absmax `0.42`, qmax `7`, **scale `0.06`**; codes `q = [2,-7,6,-2,1,5,-6,4]`; dequant `[0.12,-0.42,0.36,-0.12,0.06,0.30,-0.36,0.24]`; **mean-abs-error `0.015`** (≈3.6% of the 0.42 range); levels `16`. Model memory: fp16 13.2 GB → int4 3.30 GB (**4×**). ✓

**Page 5 · paged.** lengths `[512,37,1900,128,4]`, used `= 2581`. Reserve-max: `5·2048 = 10,240` → waste `7659` = **74.8%**. Paged (pageSize 16): `pagesFor` → `512+48+1904+128+16 = 2608` → waste `27` = **1.04%**. Same pool holds ~`10240/2608 ≈ 3.9×` more requests. ✓

**Page 6 · continuous batching.** `BATCH_LENS=[20,8,12,6]`, sum `46`, max `20`, N `4`. Static utilization `= 46/(4·20) =` **57.5%** (batch runs 20 steps, only 46 of 80 slot-steps do work). Continuous keeps 4 slots full → 46 tokens in `46/4 = 11.5` steps vs 20 → **~1.74× throughput** at **~96–100% utilization**; single-request latency unchanged (still 1 token/step/request). ✓

**Page 7 · speculative decoding.** `specExpectedTokens`: `(a=0.8,k=4)=(1−0.8⁵)/0.2 =` **3.36 tok/pass**; `(0.5,4)=` **1.94**; `(0.9,4)=` **4.10**; `(0.8,2)=` **2.44**. Net speedup with draft cost c=0.1: `(0.8,4)` = 3.36/1.4 = **2.40×**; `(0.9,4)` = **2.93×**; `(0.5,4)` = **1.38×**. ✓

**Page 8 · roofline.** `arithmeticIntensity(1,2) =` **1 FLOP/byte** (batch-1 fp16) ≪ ridge 156 → **bandwidth-bound**. int4 `arithmeticIntensity(1,0.5) =` **4** — still bandwidth-bound. Crossover at fp16: `AI = batch = 156` → need **batch ≈ 156** to become compute-bound (int4 crossover ≈ batch 39). ✓

---

# C. PER-PAGE SPEC (8 pages)

> Eyebrow reads `Part 2 · Lesson N/8`. Every widget carries a `.stage-note` with a `TRY →` line naming the exact interaction. Every page: prose (2–4 beats) → widget (recompute, not replay) → `.hood` with exact numbers → myth/reality + `.takeaway` → self-check → pager. Meta pills: `⏱ ~X min` · a middle pill naming the compute · `math: some`. Use the shared readout classes already in the monolith/course.css where handy (`.stage`, `.stage-label`, `.stage-note`, `.bneck`/`.bn-compute`/`.bn-band`/`.bn-cap` bottleneck badges, `.metric-row` + `.bar-track`/`.bar-fill`, `.read`/`.reads` big-number readouts, `.tank`, `.hood table`). Reuse the three bottleneck badges as a running visual language: **violet = compute**, **cyan = bandwidth**, **rose = capacity.**

---

## 1 · prefill-decode.html — Reading the prompt is fast; writing the reply is slow

- **title (h1):** Reading your prompt is fast. <span class="cyan">Writing</span> the reply is the slow part.
- **eyebrow:** `Part 2 · Lesson 1/8`
- **objective:** Split inference into **prefill** (whole prompt in one parallel pass, compute-bound) and **decode** (one token at a time, bandwidth-bound), and show that decode dominates wall-clock.
- **prereq:** Part 1 (you know a forward pass turns a context into next-token logits).
- **prose beats:**
  1. Running a model has two phases. **Prefill** reads your entire prompt in **one parallel pass** — all prompt tokens at once — so the GPU's math units are the limit (**compute-bound**).
  2. **Decode** then writes the answer **one token at a time**. For *every* token the GPU streams **all** the model's weights (the frozen Part-1 `E / Wq / Wk / Wv / W1 / W2`) out of memory just to take one step. The math is tiny; the memory traffic is huge (**memory-bandwidth-bound**).
  3. Same total token count, wildly different cost: 512 prompt tokens prefill in ~22 ms; generating 512 tokens takes ~3.4 seconds. Decode is the wall-clock.
- **THE WIDGET — computational spec (two costs, computed live):**
  - **Inputs:** two sliders — **prompt length P** (`0–4096`) and **generated tokens G** (`0–1024`). A precision toggle (fp16 / int4) optional but recommended (it changes decode time and previews page 4).
  - **Computed live** from `MODEL` + `GPU`: `prefill_ms = 2·paramCount()·P / GPU.flops · 1000`; `decode_ms_per_tok = weightBytes(MODEL, BYTES[prec]) / GPU.bandwidth · 1000`; `decode_ms = G · decode_ms_per_tok`. **Recompute, not replay** — every slider move re-evaluates these; no time is stored.
  - **What updates on screen:** (1) a two-segment phase timeline (reuse the monolith `.phase-cells` / `.pcell.pre` violet + `.pcell.dec` cyan) sized to P and G; (2) two `.read` big numbers — **prefill total** (violet, compute badge) and **decode total** (cyan, bandwidth badge) — plus a **"decode is Nx slower"** ratio that recomputes; (3) two `.metric-row` bars: "GPU math units" pinned high during prefill, "memory bandwidth" pinned high during decode. At the default P=512/G=512 the readout shows **21.7 ms vs 3.38 s, ~156×**.
  - **Formula on screen:** `prefill ≈ 2·N·P / FLOP/s` and `decode ≈ G · (weight_bytes / bandwidth)`.
- **under-the-hood (.hood):** N = 6.6B params. Per token: prefill `= 2·6.6e9 / 312e12 = 0.042 ms` (all P tokens share this, done in parallel); decode `= 13.2 GB / 2.0 TB/s = 6.6 ms` (one token, but every weight moved once). Ratio `6.6 / 0.042 = 156`. "Prefill does lots of math per byte fetched → compute-bound. Decode fetches all 13.2 GB to do 13 GFLOP → **bandwidth-bound**. The rest of Part 2 is tricks that attack that decode number."
- **myth / reality:** *Myth* — "Generating a token is slow because the model thinks hard." / *Reality* — "The math for one token is trivial. The cost is physically moving every weight from memory to the math units — once per token."
- **self-check:** *"Why does generating 500 tokens take far longer than reading a 500-token prompt?"* — (a) generation runs more layers than prefill *(wrong)*; **(b) prefill processes all prompt tokens in one parallel pass, but decode does one token at a time and re-streams every weight each step** *(correct, `data-correct="1"`)*; (c) the prompt is cached but the output isn't *(wrong)*. **Why:** prefill is compute-bound and parallel; decode is sequential and bandwidth-bound, so it dominates wall-clock.
- **prev / next:** prev: `index.html` ("Back to the course map") · next: `kv-cache.html` ("Don't redo the past: the KV cache")

---

## 2 · kv-cache.html — Decode reuses stored Keys and Values instead of recomputing them

- **title (h1):** The <span class="cyan">KV cache</span>: don't recompute the past
- **eyebrow:** `Part 2 · Lesson 2/8`
- **objective:** Show that each earlier token's **Key and Value** are computed once and cached, so a decode step does one token's work instead of the whole sequence's — and quantify the triangular blow-up without a cache.
- **prereq:** Part-1 lesson 6 (attention: Q·K over every token, then blend V), lesson 1 here (decode = one token at a time).
- **prose beats:**
  1. In attention the new token's **Query** is scored against every earlier token's **Key**, then blends every earlier token's **Value**. Those earlier **K and V never change** as you generate.
  2. So compute each token's K and V **once** and store them — per token, per layer, per head — in the **KV cache**. Each new step computes only its own K/V and reads the rest. **Q is not cached:** you only ever use the *current* token's Query; a past Query is never reused.
  3. Without the cache, step *t* recomputes the K/V of all *t−1* prior tokens. Over a whole generation that's `1+2+…+n = n(n+1)/2` — the **triangular blow-up**.
- **THE WIDGET — computational spec (cache on/off, count the redundant work):**
  - **Inputs:** a toggle **cache ⟷ no cache** and a slider **tokens generated n** (`1–2048`, log-ish steps 8/64/256/1024/2048 is fine).
  - **Computed live:** `withCache = n`; `noCache = recomputeWork(n) = n·(n+1)/2`; `ratio = (n+1)/2`. **Recompute, not replay** — the slider recomputes the triangular number each move.
  - **What updates on screen:** (1) reuse the monolith `.cellgrid` step animation (`.gcell.kv` = cached, `.gcell.newkv` = the one new token) — in **cache** mode one new cell lights per step while the rest stay solid; in **no-cache** mode the whole prefix re-lights every step; (2) two `.read` numbers — **K/V computations with cache** vs **without** — and a live **ratio** ("×1024.5 at n=2048"); (3) a `.metric-row` "work this step" bar: flat-low with cache, climbing linearly without. Wire the readouts to `recomputeWork`, not to the animation.
  - **Formula on screen:** `no cache: 1+2+…+n = n(n+1)/2` · `with cache: n` · `ratio = (n+1)/2`.
- **under-the-hood (.hood):** at n=2048: without cache `2048·2049/2 = 2,098,176` token-K/V computations; with cache `2,048`; ratio `1024.5`. "The cache trades compute for memory: you stop recomputing K/V, but now you must *store* them — which is the next page's problem. What's cached is the **computed K and V vectors** (per token, layer, head), never the Query and never the raw text."
- **myth / reality:** *Myth* — "The KV cache stores the conversation text." / *Reality* — "It stores computed **Key/Value vectors** for every past token, per layer and head — so decode does one token's attention work per step instead of the whole sequence's."
- **self-check:** *"What does the KV cache let decode skip?"* — (a) running the feed-forward layers *(wrong)*; **(b) recomputing the Keys and Values of every earlier token, every step** *(correct)*; (c) reading the model weights *(wrong)*. **Why:** past K/V don't change, so they're computed once and reused; only the weights (page 8) still stream every step.
- **prev / next:** prev: `prefill-decode.html` · next: `kv-memory.html` ("What the cache costs in VRAM")

---

## 3 · kv-memory.html — The cache grows linearly and eats VRAM

- **title (h1):** The cache grows with every token — and eats <span class="cyan">VRAM</span>
- **eyebrow:** `Part 2 · Lesson 3/8`
- **objective:** Compute the KV-cache memory formula live and show it grows linearly with sequence length and concurrency until it rivals or exceeds the weights.
- **prereq:** lesson 2 (what's in the cache).
- **prose beats:**
  1. The cache buys speed but costs memory, and the bill is a clean formula: `2 (K and V) × layers × n_kv_heads × head_dim × seq_len × batch × bytes`.
  2. For our 7B preset `n_kv_heads × head_dim = d_model = 4096`, so it's **512 KiB per token** across 32 layers — and it grows **linearly** with sequence length and with the number of concurrent requests.
  3. At long context the cache **outweighs the model**: 128k tokens ≈ 69 GB of cache vs 13 GB of fp16 weights. This is why long chats and big batches get expensive — and what paging (next) and KV quantization tame.
- **THE WIDGET — computational spec (size the cache, live):**
  - **Inputs:** three sliders — **sequence length** (`512 … 131072`, stepped 512/4k/8k/32k/128k), **concurrent requests / batch** (`1–32`), and a **KV precision** toggle (fp16 / int8). Optional slider for **layers** to show the linear-in-depth term.
  - **Computed live:** `kv = kvCacheBytes(MODEL, seq, batch, BYTES[kvPrec])`; `weights = weightBytes(MODEL, 2)`. **Recompute, not replay** — every input re-evaluates the formula.
  - **What updates on screen:** (1) a `.tank` split bar — rose segment = weights (13.2 GB, fixed) + cyan segment = KV cache (grows) — with the 80 GB `.gpu-line` marker; when weights+KV cross 80 GB, flag "won't fit" (rose capacity badge); (2) `.read` numbers: **KV cache GB**, **per-token KiB**, **% of the 80 GB GPU**; (3) the formula spelled with the current numbers substituted. At default seq=4096/batch=1/fp16 it reads **2.0 GiB**; drag to 128k → **68.7 GB**, tank overflows.
  - **Formula on screen:** `kv_bytes = 2 × 32 layers × 4096 (heads·dim) × seq × batch × bytes` → substitute live.
- **under-the-hood (.hood):** per token = `2·32·32·128·2 = 524,288 bytes = 512 KiB`. seq=4096 → `2.15 GB`; seq=128k → `68.7 GB` (> 13.2 GB weights). int8 KV halves both. "GQA (grouped-query attention) is the standard lever: cut `n_kv_heads` from 32 to, say, 8 and the cache shrinks 4×, because K/V are shared across query heads. Tie-back to Part-1 Q15: the residual-stream vector is *recomputed* each step; the K/V are the *stored* part that piles up here."
- **myth / reality:** *Myth* — "Memory is all about the model's size." / *Reality* — "At long context or high concurrency the **KV cache** — not the weights — is what runs you out of VRAM; it grows linearly with tokens × requests."
- **self-check:** *"Double the context length. The KV cache…"* — (a) stays the same size *(wrong)*; **(b) roughly doubles — it grows linearly with sequence length** *(correct)*; (c) quadruples like the attention matrix *(wrong)*. **Why:** the cache stores a fixed number of bytes per token per layer, so total bytes scale linearly with tokens (and with concurrent requests).
- **prev / next:** prev: `kv-cache.html` · next: `quantization.html` ("Store weights in fewer bits")

---

## 4 · quantization.html — Store weights in fewer bits (PORT the monolith widget)

- **title (h1):** <span class="rose">Quantization</span>: store each weight in fewer bits
- **eyebrow:** `Part 2 · Lesson 4/8`
- **objective:** Show weights snapping onto a small set of levels (fp16 → int8 → int4), trading precision for memory and bandwidth, with per-group scales and outlier protection.
- **prereq:** lesson 1 (decode is bandwidth-bound, so fewer bytes = fewer ms), lesson 3 (memory pressure).
- **prose beats:**
  1. Weights don't need full precision — the model shrugs off small rounding. Quantization maps a range of real values onto evenly spaced **levels**: pick a **scale**, round each weight to the nearest level. `int8` = 256 levels, `int4` = 16.
  2. Fewer bits shrinks memory **and** bandwidth: a 7B model is 13.2 GB in fp16 but **3.3 GB in int4** (4× smaller) — and since decode is bandwidth-bound, 4× fewer bytes is up to 4× fewer milliseconds.
  3. The hard part is **outliers**: a few giant weights stretch the range so everyone else crams into a few levels. **Per-group scales** (a separate scale per small block) and **protecting outliers** (keeping the few big ones in high precision — GPTQ/AWQ) rescue the precision.
- **THE WIDGET — computational spec (PORT the monolith quant mapper, drive readouts from helpers):**
  - **Reuse** `llm-inference-efficiency.html` §03 (`drawQuant`, lines 666–745): the SVG value-axis with weights as dots snapping onto level gridlines, the **bits** selector (fp16/int8/int4/int3/int2), the **per-tensor ⟷ per-group** toggle, and the **outliers included ⟷ protect outliers** toggle. Keep its deterministic 64-weight array + two planted outliers. Restyle only to the course design tokens (already the same CSS variables) — do not rebuild the geometry.
  - **Make it compute-live against the shared model:** the memory readout must call `weightBytes(MODEL, BYTES[…])` (13.2 / 6.6 / 3.30 GB for fp16/int8/int4) rather than the monolith's hard-coded `{16:140,8:70,…}` table; the **avg rounding error** bar stays computed from the real snap loop (`totErr/errN`); the **levels** readout = `2^bits`. Every toggle recomputes the snapped dots and the error — **recompute, not replay.**
  - **What updates on screen:** dots snap onto fewer gridlines as bits drop; the memory `.tank` shrinks (with the 80 GB line); the error bar grows at low bits and **visibly shrinks when per-group / protect-outliers is on**; a quality chip (indistinguishable → tiny drop → noticeable → breaks down). Provide the small `QUANT_W` worked example beside it (or in the hood) so a reader can hand-check one snap.
  - **Formula on screen:** `scale = absmax / (2^(bits−1) − 1)` · `q = round(w / scale)` · `ŵ = q · scale`.
- **under-the-hood (.hood):** `quantizeSym([0.14,-0.42,0.35,-0.14,0.07,0.28,-0.35,0.21], 4)`: absmax 0.42, scale **0.06**, codes `[2,-7,6,-2,1,5,-6,4]`, dequant `[0.12,-0.42,0.36,-0.12,0.06,0.30,-0.36,0.24]`, mean-abs-error **0.015** (~3.6%). Model memory fp16 **13.2 GB** → int4 **3.30 GB** (4×). "Per-tensor forces every weight into the same 16 levels; **per-group** gives each small block its own scale, so small weights aren't crushed by one big outlier. Modern low-bit formats: `int4 · nf4 · fp8 · mxfp4 · nvfp4`."
- **myth / reality:** *Myth* — "A 4-bit model is a crippled, 4×-worse model." / *Reality* — "The **memory** is 4× smaller; the quality drop is small and non-linear. The weights were never that precise to begin with."
- **self-check:** *"Going from fp16 to int4 weights…"* — (a) makes the model 4× less accurate *(wrong)*; **(b) cuts weight memory and bandwidth ~4× for a small, non-linear quality cost** *(correct)*; (c) changes the model's architecture *(wrong)*. **Why:** quantization rounds weights onto fewer levels; it shrinks bytes moved (helping bandwidth-bound decode) far more than it hurts quality.
- **prev / next:** prev: `kv-memory.html` · next: `paged-attention.html` ("Store the cache in pages")

---

## 5 · paged-attention.html — Fixed-size pages so variable-length sequences don't waste memory

- **title (h1):** <span class="cyan">Paged</span> attention: stop reserving memory you won't use
- **eyebrow:** `Part 2 · Lesson 5/8`
- **objective:** Show that reserving each request's *maximum* KV slot wastes most of it on padding, and fixed-size **pages** cut that waste to under one page per request.
- **prereq:** lesson 3 (the cache grows per token and per request).
- **prose beats:**
  1. You don't know a reply's length up front, so the naive cache **reserves the maximum** contiguous slot per request. A 4-token reply in a 2048 slot leaves 2044 empty — **padding waste** — and the pool **fragments**.
  2. **Paged attention** (the vLLM idea) chops the cache into fixed **pages** (e.g. 16 tokens) placed wherever there's room, like an OS's virtual memory. A sequence uses only as many whole pages as it needs.
  3. Waste drops from "up to max-length per request" to "under one page per request" — so the same VRAM serves several times more concurrent requests. It never changes the answer; it's pure memory management.
- **THE WIDGET — computational spec (reserve vs paged, count wasted slots):**
  - **Inputs:** a toggle **reserve-max ⟷ paged**, a **page size** slider (`4–64`, default 16), and an editable set of request lengths (default `PAGE_REQS = [512,37,1900,128,4]`; a "+ add request" button or a few sliders is fine).
  - **Computed live:** `used = Σ len`; `reserve = reqs.length · PAGE_MAXLEN`; `paged = Σ pagesFor(len, pageSize)`; `waste% = (alloc − used) / alloc`; `capacityFit = floor(pool / alloc_per_req)` for a fixed pool. **Recompute, not replay** — page size and lengths re-run `pagesFor`.
  - **What updates on screen:** reuse the monolith `.cellgrid` (`.gcell` colored per request `r1…r5`, `.gcell.reserved` hatched for padding): in reserve mode each request fills one row and hatches the rest; in paged mode requests pack tightly with at most `pageSize−1` hatched tail cells each. Two `.read` numbers — **memory wasted %** and **requests served in the same pool** — recompute from the formula (default: **74.8% → 1.0%**, ~**4×** more requests).
  - **Formula on screen:** `reserve = N × max_len` · `paged = Σ ceil(lenᵢ / page) × page` · `waste = 1 − used / allocated`.
- **under-the-hood (.hood):** lengths `[512,37,1900,128,4]`, used `2581`. Reserve `5·2048 = 10,240` → waste **74.8%**. Paged (16) → `512+48+1904+128+16 = 2608` → waste **1.0%**. "Paging trades a tiny per-page rounding waste for the freedom to place pages anywhere, killing fragmentation. Smaller pages waste less but add bookkeeping; 16 is a common middle. Same K/V, same output — only the *layout* changed."
- **myth / reality:** *Myth* — "Each request needs one big contiguous block of cache." / *Reality* — "Split the cache into fixed pages placed anywhere; a request uses only the pages it fills, so wasted memory drops from ~75% to ~1% and far more requests fit."
- **self-check:** *"Why does reserving each request's maximum cache waste memory?"* — (a) the weights are copied per request *(wrong)*; **(b) most replies are far shorter than the max, so the reserved-but-unused slots sit empty as padding** *(correct)*; (c) the cache is stored twice for safety *(wrong)*. **Why:** paged attention allocates fixed pages on demand, so a request occupies only what it fills — under one page of slack — instead of the whole max-length block.
- **prev / next:** prev: `quantization.html` · next: `continuous-batching.html` ("Share one weight-read across many requests")

---

## 6 · continuous-batching.html — Many requests share one forward pass; finished ones swap out mid-flight

- **title (h1):** <span class="green">Continuous batching</span>: read the weights once, serve many
- **eyebrow:** `Part 2 · Lesson 6/8`
- **objective:** Show that one decode step reads the whole model regardless of batch size, so packing many requests into that read multiplies throughput — and that continuous (vs static) batching keeps every slot busy. Distinguish throughput from latency.
- **prereq:** lesson 1 (decode reads all weights each step, bandwidth-bound).
- **prose beats:**
  1. One decode step streams the entire model **no matter how many requests ride along**. So push many requests' tokens through that single weight-read and the expensive read is **amortized** across everyone — more total tokens/second (**throughput**) for the same bytes moved.
  2. **Static** batching launches a fixed batch and waits for the **longest** member to finish; short requests idle their slots until the whole batch resets — wasted capacity.
  3. **Continuous** batching swaps a finished sequence out and a waiting one in **every step** (interleaving new prefills with in-flight decodes), so slots stay full. It raises throughput; it does **not** make any single reply arrive sooner.
- **THE WIDGET — computational spec (static vs continuous, compute utilization live):**
  - **Inputs:** a toggle **static ⟷ continuous** and an editable set of request decode lengths (default `BATCH_LENS = [20,8,12,6]`; a "length variance" slider is an acceptable simplification). Batch size B = number of slots.
  - **Computed live:** static: `util = Σ len / (B · max(len))`; `steps = max(len)`; `throughput ∝ Σ len / max(len)`. continuous: slots stay full → `util ≈ 1`; `steps = ceil(Σ len / B)`; `throughput ∝ B`. `speedup = static_steps / continuous_steps`. **Recompute, not replay** — editing lengths re-runs the sums (default: util **57.5% → ~100%**, throughput **~1.74×**).
  - **What updates on screen:** reuse the monolith batch timeline (`.batch-grid` rows, `.bg-cell.on`/`.idle`): static shows short rows going hatched-idle while the longest keeps running; continuous shows idle cells immediately refilled by a new-colored request. Two `.read` numbers — **GPU utilization** and **throughput ×** — computed from the formula, plus a **"latency per request: unchanged"** note (green throughput badge).
  - **Formula on screen:** `static util = Σ len / (B × max len)` · `continuous ≈ full slots every step` · `throughput ∝ tokens / steps`.
- **under-the-hood (.hood):** lengths `[20,8,12,6]`, B=4, sum 46, max 20. Static: 46 useful slot-steps of `4·20 = 80` → **57.5%**; produces 46 tokens in 20 steps. Continuous: 4 full slots → 46 tokens in `⌈46/4⌉ = 12` steps → **~1.74×** throughput, **~96–100%** utilization. "Throughput ≠ latency: batching multiplies *total* tokens/second across users but a single request still gets one token per step. It's why one served model answers thousands of people at once for the same weight-reads."
- **myth / reality:** *Myth* — "Batching makes each user's reply faster." / *Reality* — "Batching multiplies **throughput** (tokens/second across all users) by sharing one weight-read; a single request's **latency** is unchanged."
- **self-check:** *"Continuous batching improves mostly…"* — (a) the latency of one request *(wrong)*; **(b) total throughput — it keeps every slot busy by swapping finished requests out mid-flight** *(correct)*; (c) the model's accuracy *(wrong)*. **Why:** the weight-read is fixed per step, so filling every slot amortizes it across more tokens; each individual reply still advances one token per step.
- **prev / next:** prev: `paged-attention.html` · next: `speculative-decoding.html` ("Guess ahead, verify in one pass")

---

## 7 · speculative-decoding.html — A small draft proposes k tokens; the big model verifies in one pass (PORT the monolith widget)

- **title (h1):** <span class="green">Speculative</span> decoding: guess ahead, verify in one pass
- **eyebrow:** `Part 2 · Lesson 7/8`
- **objective:** Show a small draft model proposing k tokens that the big target verifies in a single parallel pass, keeping the accepted run; compute expected speedup from an acceptance-rate slider.
- **prereq:** lesson 1 (one decode step reads the whole model — wasteful for one token), Part-1 lesson 11 (a distribution over next tokens).
- **prose beats:**
  1. Decode's expensive step reads the whole model to make **one** token — wasteful. So a small fast **draft** model proposes the next *k* tokens cheaply.
  2. The big **target** model checks all *k* **in a single parallel pass** — same parallelism as prefill, so verifying k tokens costs ~one normal decode step. It keeps the longest agreed prefix and **rejects from the first disagreement**, always emitting one correct token at the break.
  3. When the draft is right you get several tokens for the price of one expensive step — and the output is **exactly** what the target would have written alone. Higher **acceptance rate** = more free tokens.
- **THE WIDGET — computational spec (PORT the monolith animation, drive speedup from the helper):**
  - **Reuse** `llm-inference-efficiency.html` §06 (`runSpec`, lines 842–915): the draft/target towers, the **draft length k** selector (2/4/6), the **aligned ⟷ weak draft** toggle, the chip line (`.tok.draft/.accepted/.rejected/.filled`), the committed sentence, and the plain-vs-spec **race** bars. Keep the accept-run logic (accept consecutive matches, reject at first miss, always commit one target token).
  - **Add a live acceptance-rate model:** an **acceptance-rate `a`** slider (`0.3–0.95`) and a **draft-cost `c`** control (or fixed `c=0.1`). Compute `E = specExpectedTokens(a, k)` and `speedup = specSpeedup(a, k, c)` and show them as `.read` numbers that recompute on every slider/k change — **recompute, not replay.** (The aligned/weak toggle can set `a` to a preset high/low value so the animation and the number agree.)
  - **What updates on screen:** the animation commits an accepted run then one target token; the **tokens/step** and **expected speedup** readouts update from `specExpectedTokens`/`specSpeedup`; the race bars show spec finishing in fewer expensive steps. At default `a=0.8, k=4`: **3.36 tokens/verify pass, ~2.4× net.**
  - **Formula on screen:** `E[tokens/pass] = (1 − a^(k+1)) / (1 − a)` · `net speedup = E / (1 + k·c)`.
- **under-the-hood (.hood):** `(a=0.8,k=4)` → `(1−0.8⁵)/0.2 = 3.36` tokens/pass; net `/1.4 = 2.40×`. `(a=0.9,k=4) → 4.10 → 2.93×`; `(a=0.5,k=4) → 1.94 → 1.38×`; `(a=0.8,k=2) → 2.44 → 2.03×`. "Longer k helps only while acceptance stays high — a wrong early guess wastes the tail. Verifying k tokens in one pass is the same parallelism as prefill (lesson 1); that's why the target pass is ~one decode step, not k of them. **The output is identical to plain decoding** — every token is target-verified; rejects are discarded."
- **myth / reality:** *Myth* — "The small draft model's mistakes leak into the answer." / *Reality* — "Every token is verified by the target; rejects are thrown away. The text is exactly what the target alone would have written — this is speed, not a quality change."
- **self-check:** *"How can speculative decoding be faster yet identical in output?"* — (a) it averages the two models' predictions *(wrong)*; **(b) the draft proposes several tokens, the target verifies them in one pass and keeps only the ones it agrees with** *(correct)*; (c) it lowers the temperature *(wrong)*. **Why:** verification is a single parallel target pass; accepted tokens are exactly what the target would have produced, so speed rises with the acceptance rate while output is unchanged.
- **prev / next:** prev: `continuous-batching.html` · next: `roofline.html` ("Why all of this works: the roofline")

---

## 8 · roofline.html — Compute- vs bandwidth-bound: decode is bottlenecked by moving weights, not math

- **title (h1):** The <span class="violet">roofline</span>: decode moves weights, it doesn't do math
- **eyebrow:** `Part 2 · Lesson 8/8`
- **objective:** Compute **arithmetic intensity** live and place a decode step below the GPU's **ridge point** (bandwidth-bound) — the single *why* under KV cache, quantization, and batching.
- **prereq:** all of Part 2 (each trick attacked one bottleneck; this names it).
- **prose beats:**
  1. **Arithmetic intensity** = FLOPs done per byte moved from memory. A GPU has a **ridge point** = peak FLOP/s ÷ bandwidth (≈156 FLOP/byte on an A100). Below it you're **bandwidth-bound**; above it, **compute-bound**.
  2. A batch-1 decode step is a matrix-**vector** product: each weight is read once and used for ~2 FLOPs, so intensity ≈ **1 FLOP/byte at fp16** — 156× below the ridge. Decode is **bandwidth-bound**: the GPU waits on memory, not math.
  3. That's the whole part in one picture. **Quantization** shrinks the bytes moved; **KV cache** avoids re-moving past work; **batching** raises intensity toward the ridge (`AI ≈ 2·batch / bytes_per_param`). All three attack *moving weights*, not math.
- **THE WIDGET — computational spec (place the dot under the roofline, live):**
  - **Inputs:** a **batch size** slider (`1–256`) and a **precision** toggle (fp16 / int8 / int4). Optional GPU preset selector (A100 default).
  - **Computed live:** `AI = arithmeticIntensity(batch, BYTES[prec]) = 2·batch / bytes`; `ridge = GPU_RIDGE = 156`; `bound = AI < ridge ? 'bandwidth' : 'compute'`; achievable throughput = `min(GPU.flops, AI · GPU.bandwidth)`. **Recompute, not replay** — every change repositions the operating point.
  - **What updates on screen:** a log-log **roofline** SVG — a rising bandwidth line that flattens at the compute ceiling, with the ridge at 156 FLOP/byte — and a moving **operating-point dot** at the current `AI`. A cyan bandwidth badge below the ridge, violet compute badge above. `.read` numbers: **arithmetic intensity**, **bound: bandwidth/compute**, **batch needed to reach the ridge** (`156` at fp16, `39` at int4). At default batch=1/fp16 the dot sits far left: **1 FLOP/byte, bandwidth-bound.**
  - **Formula on screen:** `AI = 2 × batch / bytes_per_param` · `ridge = peak FLOP/s ÷ bandwidth = 156` · `bandwidth-bound while AI < ridge`.
- **under-the-hood (.hood):** A100: `312e12 / 2.0e12 = 156 FLOP/byte`. batch-1 fp16: `AI = 2·1/2 = 1` ≪ 156 → **bandwidth-bound**; int4: `AI = 4`, still bandwidth-bound. Compute-bound needs `AI ≥ 156` → **batch ≈ 156** at fp16 (≈39 at int4). "Map it back: **quantization** cuts `bytes_per_param` (raising AI and cutting the bytes moved); **continuous batching** raises `batch` (raising AI toward the ridge); **KV cache** removes redundant K/V movement so the bytes you *do* move are only the weights. Every Part-2 trick is a move on this one plot. And none of them change a single number the model outputs (Part-1 Q18) — only how the computation is scheduled and stored."
- **myth / reality:** *Myth* — "Faster generation needs a faster GPU (more FLOPs)." / *Reality* — "At batch 1, decode is **bandwidth-bound** — it's waiting on memory, not math. Faster *memory*, fewer *bytes* (quantization), or more *batch* help far more than more FLOPs."
- **self-check:** *"Batch-1 decode is bandwidth-bound because…"* — (a) the model has too many layers *(wrong)*; **(b) it does only ~1 FLOP per byte of weights moved, far below the GPU's ridge point, so memory bandwidth is the limit** *(correct)*; (c) the GPU lacks compute units *(wrong)*. **Why:** each weight is read once for ~2 FLOPs, so arithmetic intensity ≈ 1 ≪ 156; the fix is fewer bytes (quantization) or higher intensity (batching), not more FLOPs.
- **prev / next:** prev: `speculative-decoding.html` · next: `index.html` (label "Back to the map — you've seen how it runs fast")

---

# D. HUB (index.html) CHANGES

Add a second lesson section below the Part-1 list, extend the intro to mention both parts, and reframe the classic-walkthrough inference link. Reuse the existing `.lesson-card` markup verbatim; only content changes.

### D0 · Intro + framing
- **Meta description** (`<head>`): extend to mention both parts, e.g. append: *"…Then Part 2 shows how that same computation is made fast to run — KV cache, quantization, batching, speculative decoding — every number still computed live."*
- **Header `<p class="lede">`:** after the existing Part-1 sentence, add one sentence: *"Then <span class="strong-ink">Part 2 · How LLMs Run Fast</span> takes the same model and shows how it's served in milliseconds — the KV cache, quantization, paging, batching, speculative decoding, and the roofline that explains why they all help."*
- **Pill row:** update the lessons count pill `📚 <b>15 lessons</b>` → `📚 <b>15 + 8 lessons</b>` (or add a second pill `📚 <b>Part 2 · 8 lessons</b>`).
- **Anchor:** give the existing Part-1 lessons `<section>` an `id="part1"` if not present (the Part-1 crumbs already link `index.html#part1`), and the new Part-2 section `id="part2"` (the Part-2 crumbs link `index.html#part2`).

### D1 · New "Part 2" section (8 cards) — insert AFTER the Part-1 `</section>`, BEFORE the classic-walkthrough section

Mirror the Part-1 markup exactly (`<section class="block wrap reveal">` → `<ol class="lesson-list">` → `.lesson-card` items). Add a section heading and a one-line kicker above the `<ol>`:

```html
<section class="block wrap reveal" id="part2" style="padding-top:10px">
  <h2 style="font-size:clamp(1.5rem,3.4vw,2.1rem);margin-bottom:6px">Part 2 · <span class="green">How LLMs Run Fast</span></h2>
  <p class="prose" style="margin-bottom:8px">Same model, now served in milliseconds. Eight pages, one lever each — and every FLOP, byte, and token-per-second is computed live from one 7B-class model preset.</p>
  <ol class="lesson-list">
    <!-- 8 cards below -->
  </ol>
</section>
```

Card list (numbers `01`–`08` restart for Part 2; color dot per the canon; blurb = "what you'll compute"):

| # | href | dot | lc-ttl | lc-blurb (what you'll compute) |
|---|------|-----|--------|--------------------------------|
| 01 | prefill-decode.html | **amber** | Reading the prompt is fast; writing is slow | Compute prefill vs decode time and watch decode dominate (~156×). |
| 02 | kv-cache.html | **cyan** | The KV cache: don't recompute the past | Count the triangular blow-up of recomputing every past token's K/V. |
| 03 | kv-memory.html | **cyan** | The cache grows and eats VRAM | Size the KV cache live: layers × heads × dim × seq × 2 × bytes. |
| 04 | quantization.html | **rose** | Store each weight in fewer bits | Snap weights onto int8/int4 levels; watch memory drop 4×. |
| 05 | paged-attention.html | **cyan** | Paged attention: stop reserving unused memory | Compute padding waste for reserve-max vs paged (~75% → ~1%). |
| 06 | continuous-batching.html | **green** | Read the weights once, serve many | Compute GPU utilization and throughput for static vs continuous. |
| 07 | speculative-decoding.html | **green** | Guess ahead, verify in one pass | Compute expected tokens/step and speedup from an acceptance rate. |
| 08 | roofline.html | **violet** | The roofline: moving weights, not math | Compute arithmetic intensity and land below the ridge point. |

Each card is exactly the Part-1 shape, e.g.:
```html
<li>
  <a class="lesson-card" href="prefill-decode.html">
    <span class="lc-num">01</span>
    <span class="lc-dot amber" aria-hidden="true"></span>
    <span class="lc-body">
      <span class="lc-ttl">Reading the prompt is fast; writing is slow</span>
      <span class="lc-blurb">Compute prefill vs decode time and watch decode dominate (~156×).</span>
    </span>
    <span class="lc-arrow" aria-hidden="true">→</span>
  </a>
</li>
```
(Optional flagship-style flag on 08: `<span class="flag">the why</span>` inside `.lc-ttl`.)

### D2 · Classic-walkthrough block — reframe the inference link
Keep the block; reword the two `.ov-desc` lines so the interactive Part 2 is clearly the primary path now:
- `How LLMs Work` → desc: *"the original narrated version of Part 1"*
- `How LLMs Run Fast` → desc: *"the original narrated version — Part 2 is now the interactive track above"*

Optionally add one line under the `<p class="prose">`: *"Part 2 above is the interactive rebuild of the second walkthrough — one idea per page, every number live."*

### D3 · Color legend (optional)
The existing hub legend maps amber/cyan/violet/rose/green. Add "green = throughput" alongside "probabilities", or leave as-is (green already covers both). No new colors are introduced.

---

# E. Part-1 copy micro-fixes (from iter-3 "new confusion") — checklist for the coding pass

One line each; file + intent. These are copy-only edits the orchestrator may apply directly.

1. **`unembedding.html`** — add a **bridging sentence** at the top of the setup prose explaining the L9→L10 position switch: *"prediction happens at the **last** token, `was`, not at `it` — here's its final vector,"* so the magic `[0.9,0.3,1.4,0.1]` no longer appears from nowhere.
2. **`unembedding.html`** — add a **"cat 61% vs tired 30.3%" reconciliation** line: state that "cat 61%" is a stand-in for any explainer's confident number, and the toy's own computed answer is `tired` at 30.3% over 7 words — the *mechanism* (one dot per word) is what's real.
3. **`sampling.html`** — add the **"richer candidate set" half-sentence** where it opens claiming the same distribution: *"…swapping to a richer candidate set (8 illustrative next-words) to make the dials visible,"* so the switch from the 7-token vocab / tired-30.3% to the 8 candidates / tired-51% is disclosed.
4. **`positional-encoding.html`** — add the **separate-toy-vocab heads-up** (mirroring analogy's fix): flag that `dog / bites / man` with dims `[animacy, action, position, position]` is a **separate demo vocabulary**, not the sentence tokens, and dims 2–3 here are repurposed as position slots.
5. **`hallucination.html`** — soften **"famous" → "frequent in training"**: reword the prose "when a false answer is more *famous* or *frequent*" so the mechanism reads as *frequency/association in training data*, not real-world fame.

---

## Builder checklist (no further decisions needed)
- New file `tools/course/course-data-p2.js` created verbatim from Section B; every widget computes from its helpers (recompute, not replay); Section-B verification numbers reproduce to the stated decimals.
- 8 new pages cloned from `_template.html`, in `tools/course/`, wired `course-data.js` → `course-data-p2.js` → `course.js` → page script; eyebrow `Part 2 · Lesson N/8`; crumbs link `index.html#part2`.
- Pages 4 and 7 **port** the monolith's real quant and spec-decode widgets (don't rebuild), with readouts driven by `weightBytes` / `quantizeSym` (pg 4) and `specExpectedTokens` / `specSpeedup` (pg 7).
- Pager chain intact: `index → prefill-decode → kv-cache → kv-memory → quantization → paged-attention → continuous-batching → speculative-decoding → roofline → index`.
- Hub gains a `#part2` section of 8 cards (Section D), intro/description mention both parts, classic-walkthrough inference link reframed as "the original narrated version."
- Section E copy micro-fixes applied to the 4 named Part-1 pages.
