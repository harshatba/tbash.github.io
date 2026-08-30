# Learner Review — Part 2 "How LLMs Run Fast" (iteration 4, FINAL pass)

*Same reviewer: junior SWE, Python/JS, zero ML background. In iter-3 I signed off the 15-page Part 1 and asked for the monolithic Part 2 to be decomposed into the same one-idea-per-page, compute-live format — I sketched a 7-page batch + an 8th roofline capstone. That 8-page batch is now built. I read all 8 pages prose-first, poked every widget, then read each page's `<script>` + `course-data-p2.js` to check computed-vs-replayed. I hand-verified **all 8** compute-live widgets against the shared helpers (not just the 3 asked). I also re-opened the four Part-1 pages I flagged for copy fixes.*

---

## Does Part 2 deliver what you asked for?

I asked for exactly these 8. Every one shipped, as its own page, with a live widget wired to a shared helper in `course-data-p2.js` — **none replays a hard-coded headline number.** Verdict per page, with the widget + what the JS actually does, and my hand-check:

### 1. Prefill vs decode → **RESOLVED** (`prefill-decode.html`)
Widget: two sliders (prompt P, generated G) + fp16/int4 toggle → a wall-clock timeline, two big readouts, and a live substituted formula. `recompute()` computes `prefill_ms = 2·N·P/GPU.flops·1000` and `decode_ms = G·(weightBytes/GPU.bandwidth·1000)` every interaction — nothing stored. **Hand-verified:** N=`paramCount()`=6.607e9; prefill-per-token = 1.32e10/312e12 = 0.0424 ms; decode-per-token = 13.2 GB/2.0 TB/s = 6.61 ms; ratio **6.61/0.0424 = 156×** ✓. At P=G=512: prefill 21.7 ms, decode 3.38 s ✓. The AI-based bottleneck note (compute-bound when `2P/bytes ≥ 156`) is a genuinely careful touch.

### 2. The KV cache → **RESOLVED** (`kv-cache.html`)
Widget: cache/no-cache toggle animates a 24-cell grid; a separate `n` slider drives the headline via `recomputeWork(n) = n(n+1)/2`. The stage-note explicitly says the two big numbers come from `recomputeWork(n)`, "not from the animation." **Hand-verified:** n=2048 → no-cache `2048·2049/2 = 2,098,176`, with-cache 2,048, ratio `(2048+1)/2 = 1024.5×` ✓. Whole hood table (8/64/256/1024/2048) checks.

### 3. KV-cache memory → **RESOLVED** (`kv-memory.html`)
Widget: seq/batch/layers sliders + fp16/int8 toggle → a weights-vs-cache "tank" against an 80 GB line, plus a live formula. Driven by `kvCacheBytes(m, seq, batch, bytes)`. **Hand-verified:** per token = `2 × 32 layers × 4096 (heads·dim) × 2 B = 524,288 B = 512 KiB/token` ✓. At seq 4096, batch 1: `512 KiB × 4096 = 2.147e9 B = 2.0 GiB / 2.15 GB` ✓. At 128k: 68.7 GB (bigger than the 13.2 GB model) ✓; batch 32 at 4k gives the same 68.7 GB ✓.

### 4. Quantization → **RESOLVED** (`quantization.html`)
Widget: an SVG mapper (64 weights snapping onto level gridlines, 2 planted outliers) with bit-width, per-tensor/per-group, and outlier-protect toggles; memory tank; and a hood worked-example from `quantizeSym(QUANT_W, 4)`. Memory from `weightBytes(MODEL, bits/8)`. **Hand-verified:** int4 = 6.607e9 × 0.5 B = **3.30 GB, exactly 4× under fp16 13.2 GB** ✓. `quantizeSym([0.14,-0.42,…], 4)`: absmax 0.42, scale 0.42/7 = 0.06, codes `[2,-7,6,-2,1,5,-6,4]`, MAE **0.015** (~3.6%) ✓ — I re-derived all 8 dequantized values.

### 5. Paged attention → **RESOLVED** (`paged-attention.html`)
Widget: editable request-length rows → a per-request memory map (used / hatched-padding / free) with reserve-max vs paged toggle and page-size slider. Uses `pagesFor(len, page) = ceil(len/page)·page`. **Hand-verified** the default `[512,37,1900,128,4]`: used 2581; reserve alloc 5×2048 = 10,240, waste `1−2581/10240 = 74.8%` ✓; paged (page 16) allocs `512/48/1904/128/16 = 2608`, waste `1−2581/2608 = 1.0%` ✓; capacity in a 40,960-slot pool `⌊40960/2048⌋=20 → ⌊40960/521.6⌋=78`, **3.9×** ✓.

### 6. Continuous batching → **RESOLVED** (`continuous-batching.html`)
Widget: 4 editable decode lengths + static/continuous toggle → a slot timeline (idle-hatch vs refilled) and util/throughput/steps readouts. Everything falls out of Σlen, max, B. **Hand-verified** `[20,8,12,6]`: Σ=46, max=20, B=4; static util `46/80 = 57.5%` ✓; continuous ⌈46/4⌉=12 steps; speedup `20/11.5 = 1.74×` = `1/0.575` ✓.

### 7. Speculative decoding → **RESOLVED** (`speculative-decoding.html`)
Widget: draft/target towers + an accept/reject token animation (separate) and a **live** model from `specExpectedTokens(a,k)` and `specSpeedup(a,k,c)`, driven by an acceptance slider — the slider "drives the MODEL only — no replay." **Hand-verified** default a=0.8, k=4: `E = (1−0.8⁵)/0.2 = 0.67232/0.2 = 3.36` ✓; speedup `3.36/(1+4·0.1) = 3.36/1.4 = 2.40×` ✓; the whole hood table (0.9/0.5/k=2) checks.

### 8. The roofline (capstone) → **RESOLVED** (`roofline.html`)
Widget: batch slider + fp16/int8/int4 toggle → a log-log roofline SVG with a moving operating-point dot, using `arithmeticIntensity(batch,bytes) = 2·batch/bytes` and `GPU_RIDGE = 312e12/2.0e12 = 156`. **Hand-verified:** batch-1 fp16 AI = `2·1/2 = 1` (156× below ridge) ✓; batch to reach ridge = `156·bytes/2` = 156 (fp16) / 78 (int8) / **39 (int4)** ✓.

**Every widget computes live.** The one place a number is typed as a constant is `c = 0.1` (draft-cost fraction) on the spec-decode page — and it's labeled "draft size ÷ target size," a stated assumption, not a replayed result. No headline number is hard-coded anywhere I checked.

---

## Does Part 2 hang together, and connect back to Part 1?

**Yes — this is a real chain, and the roofline capstone lands.** Walking 1→8 as a learner:

- **L1** establishes the spine: decode owns the wall-clock, and it's **bandwidth-bound** ("every trick in the rest of Part 2 is a move against that one decode number"). Names the *frozen* Part-1 weights `E/Wq/Wk/Wv/W1/W2` explicitly.
- **L2** KV cache saves *compute* (the triangular blow-up) and closes with "but now you have to *store* them… which is exactly the next page's problem."
- **L3** KV memory picks that up ("Last page the cache saved us… but that storage has to live somewhere"), and its hood has an explicit **Q15 tie-back**: the residual vector is recomputed-and-thrown-away, the K/V are the stored part. Ends → paging + KV quant.
- **L4** quantization opens on "the *frozen* weights from Part 1" and connects to L1 ("since decode is bandwidth-bound, 4× fewer bytes is up to 4× fewer ms/token").
- **L5** paging references L3 ("the KV cache grew per token and per request"). Ends → batching.
- **L6** batching references L1 ("Decode is bandwidth-bound: every step streams all the weights…") and L5 ("Paging… is what makes room in VRAM for all those concurrent KV caches").
- **L7** spec-decode reuses L1's parallelism ("the same parallelism as prefill, so verifying k costs about one decode step").
- **L8** roofline is the payoff: **it names quantization, KV cache, and batching as three levers on one plot** — "quantization cuts bytes_per_param (dot slides right); batching raises batch (walks up the diagonal); KV cache removes redundant K/V movement" — and closes with **"None of them touch a single number the model outputs (Part-1 Q18)."**

So it explicitly answers the three questions Part 1 *ended* on: **what's frozen** (L1/L3/L4 all name it), **how does it run** (the whole part), **why is it slow** (roofline: bandwidth, not math). My old Q18 ("do KV/quant/spec leave every number identical?") is answered in three places (L7 "output identical," L8 "none touch a number the model outputs").

**No number appears from nowhere.** Everything descends from the one `MODEL` 7B preset and the A100 `GPU` constants. Rounding is consistent (weights quoted as "13 GB" in prose, 13.2 GB in hoods; 128k cache "~69 GB" vs 68.7 GB exact — same number).

Two places where a page *computes but the point could land a hair harder* — both minor:
- **L6's headline speedup (1.74×) is computed from the idealized `Σ/B = 11.5` steps, while the "steps to clear" readout shows the whole-step `12`.** With 12 the speedup would be 1.67×. The hood shows the 11.5 openly, so it's a disclosed idealization (ignores the half-step tail + prefill), not a contradiction — but a learner who divides 20/12 in their head gets a slightly different number than the 1.74 on screen.
- **L7's free acceptance slider can desync from the fixed animation.** The slider drives *only* the model (E, speedup); the accept/reject animation uses a fixed good/weak match pattern. The page discloses this ("the aligned toggle sets a high a so the animation and the numbers agree"), and the aligned/weak buttons snap the slider to a matching preset — but if you drag the raw slider to a=0.30 the model reads E=1.36 while the animation still commits its "good" run. Defensible (slider is explicitly model-only), just a possible double-take.

---

## Did the Part-1 copy fixes land?

All four RESOLVED:

1. **Unembedding L9→L10 bridge → RESOLVED.** `unembedding.html` line 96 is exactly the bridging sentence I asked for: *"the last few pages followed `it`… → its running vector `[2.13, 0.54, 0.36, 0.61]`. But the **next** word is predicted from the **last** token, `was` — every position carries its own running vector; this is just `was`'s… That's the one we turn into a word."* The un-signposted token switch is now signposted.
2. **Sampling "richer/illustrative candidate set" note → RESOLVED.** `sampling.html` line 159 now appends *"(Illustrative candidates — the mechanism is the real thing.)"* to the hood, acknowledging the 8 demo tokens aren't the 7-token sentence vocab. (Phrased as "illustrative" rather than "richer set," but it closes the honesty gap I flagged.)
3. **Positional-encoding separate-vocab heads-up → RESOLVED.** `positional-encoding.html` line 138 now carries the same courtesy analogy got: *"this page uses its own tiny demo words (dog · bites · man)… a separate demo from the cat/sat/it sentence table, whose four dims meant something else."*
4. **Hallucination "frequent in training" → RESOLVED.** `hallucination.html` line 128 now leads with the real mechanism: *"when a false answer appeared more **frequently in the training text**… it can carry the higher logit"* — "famous" is now the parenthetical proxy, not the claim.

---

## New confusion / errors on the Part 2 pages

Being fair about the deliberate toy simplifications (single 7B preset, batch-1 first-order timing, geometric acceptance model — all standard and mostly disclosed), the only *real* things I'd flag:

- **`prefill-decode.html`: the prefill number dips below the physical weight-read floor at small P.** `prefill_ms = 2·N·P/peak` is a pure compute estimate. At P=32 it reports ~1.35 ms — but reading all 13.2 GB of weights once costs 6.61 ms minimum, so prefill can't actually be faster than one weight-read. The `tlNote` *does* flip to "prefill here leans on bandwidth" at small P, but the headline number doesn't apply the bandwidth floor, so a digit-tracer sees "prefilling 32 tokens (1.35 ms) is faster than decoding 1 token (6.61 ms)," which is below what the memory can physically deliver. Real, but toy-acceptable and partially disclosed — a one-line `max(compute, weight-read)` would make it airtight.
- **`kv-memory.html`: dropping the "layers" slider shrinks the cache but leaves weights at the full 13.2 GB.** Correct in intent (you're isolating the cache term, and weights are labeled "fixed"), but a learner who drags layers 32→1 sees the cache collapse while the model bar doesn't budge, which can read as "fewer layers, same model." The labels ("cache depth" / "weights fp16, fixed") carry it, but it's the one control that risks a wrong mental model.
- **`quantization.html`: two different weight sets in one lesson.** The SVG mapper uses 64 pseudo-random weights (with 2 planted outliers); the hood worked-example uses the 8-value `QUANT_W`. Both are honest and neither claims to be the other, but the MAE bar in the SVG and the MAE `0.015` in the hood are unrelated numbers on the same page — I briefly expected them to match. Cosmetic.

Nothing here is a factual error or a broken widget. Cross-links (pager prev/next across all 8, hub `#part2` list, eyebrows "Lesson X/8") are all correct and chain cleanly index → 1 → … → 8 → index.

---

## Remaining questions (short) — candidates for an optional Part 3, not blockers

1. **GQA** is mentioned twice (data-file comment + kv-memory hood "cuts n_kv_heads 32→8, cache shrinks 4×") but never gets a widget. It's the single biggest cache lever — does it deserve its own page?
2. **FlashAttention** never appears. The attention score matrix is quadratic (self-check on kv-memory even says so) — how is that computed without materializing an n×n matrix?
3. **Disaggregated prefill/decode** — since L1 shows the two phases have opposite bottlenecks (compute vs bandwidth), do real serving stacks run them on *different* hardware?
4. **Multi-GPU / tensor & pipeline parallelism** — the whole part assumes the model fits one A100. What changes when weights are split across 8 GPUs, and does that add a *network* bottleneck to the roofline?
5. **Prefix/KV cache sharing** across requests (shared system prompts) — paging places pages anywhere; can two requests point at the *same* pages?
6. **Does quantizing the KV cache to int8 (offered on L3) hurt quality** the way weight-quant does, or is the cache more forgiving?
7. **Speculative decoding's draft model** — where does it come from? Is it a separately trained small model, or a trimmed version of the target?
8. **Continuous batching + prefill** — a new request's prefill is compute-heavy; does slotting it mid-decode-batch stall the decoders (the "prefill stall" problem)?
9. **The roofline uses one A100.** How much of Part 2's advice flips on a bandwidth-rich chip (H100/MI300) where the ridge moves?
10. **MoE models** — "2×params FLOPs/token" assumes every weight is used. What happens to the roofline when only a few experts fire per token?

---

## FINAL VERDICT

**(1) Is Part 2 satisfactory?** **Yes — I would not pass it back.** All 8 pages I asked for shipped in the one-idea-per-page, compute-live format; I hand-verified all 8 widgets to the decimal against `course-data-p2.js` (156×, 1024.5×, 512 KiB/2.0 GiB, int4 3.30 GB/4×, 74.8%→1.0%, 57.5%/1.74×, E=3.36/2.40×, AI=1/ridge-39); nothing replays a hard-coded headline; the chain 1→8 is tightly cross-linked; and the roofline capstone genuinely unifies KV cache + quantization + batching while answering the exact "what's frozen / how does it run / why is it slow" questions Part 1 ended on. The only findings are minor, mostly-disclosed toy idealizations (prefill's small-P floor, L6's 11.5-vs-12 step, L7 slider/animation desync) — copy-polish, not redesign.

**(2) Is the OVERALL course complete enough to STOP the loop?** **Yes — stop the review→build loop.** Part 1 (15 lessons) was signed off in iter-3 and its four flagged copy fixes have all landed; Part 2 (8 lessons) is satisfactory; both classic overview pages remain linked from the hub as the narrated tours. The course now runs end-to-end from "a token ID" to "why it's served in milliseconds," every number traceable. **No blocking issue remains.** The Part-3 questions above (GQA, FlashAttention, multi-GPU, MoE, disaggregation) are genuine and a learner is primed for them — but they're optional expansion, not a gap in what's shipped. Ship it.
