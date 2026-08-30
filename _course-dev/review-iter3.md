# Learner Review — "How LLMs Work" full 15-lesson track (iteration 3)

*Same reviewer: junior SWE, Python/JS, zero ML background. I read all 8 new pages (8–15) prose-first, poked every widget, then read each page's `<script>` + `course-data.js` to check computed-vs-replayed. I hand-verified the FFN, unembedding, drift, hallucination and sampling arithmetic against the shared ops so I could be fair about the "it genuinely computes" claim. I also re-opened the old pages to confirm the iter-2 fixes landed.*

---

## Did the iter-2 seams get fixed?

I raised seven specific things in iter-2. Verdict on each, with the page and what I actually saw in the widget/JS:

### (a) The blended vector's fate → **RESOLVED** (`residual-stream.html`, lesson 8)
This was my single biggest seam ("the track stops one step too early"). It's now the very first new page, and it's honest. The widget computes `A = attend(TOKENS.indexOf('it'), Wq, Wk).out` — the *real* lesson-6 pipeline, not the stored `RESID_VEC` (line 250; there's even a dev-assert at line 343 that recompute matches the documented target). Add mode shows `new[d] = e[d] + A[d]` written out per dimension (`0.8 + 0.62 = 1.42`); the Replace toggle strikes the old column through and the read-out says "the token forgot itself." The myth/reality nails the point: *refine, don't replace; the vector is a running total*. The hood even earns the residual with a one-liner on why deep nets stay trainable (clean gradient path down the sum). Exactly the "missing next sentence after page 6" I asked for.

### (b) The neuron no longer islanded → **RESOLVED** (`feed-forward.html`, lesson 9)
In iter-2 the lesson-4 neuron "sat islanded before attention with no payoff." Now the FFN page *is* that neuron, stacked (4→8→4) and eating `it`'s real post-attention residual `[1.42, 0.34, 0.45, 0.40]`. The lede literally says "lesson 4's neuron, finally eating a real post-attention vector," and links back to `neuron.html`. `render()` calls `ffnForward(input)` live; nudge any input slider and all 8 hidden + 4 output tiles recompute. **I verified by hand:** h0 = `tanh(1.42·1 − 0.5)` = `tanh(0.92)` = 0.726 ≈ 0.73 (fires hardest, correctly highlighted); out0 = `0.6·0.73 + 0.4·0.69` = 0.71; added back → `[2.13, 0.54, 0.36, 0.61]`. Every number in the hood table checks. The "each token thinks alone / attention never lets them talk" contrast with attention is crisp.

### (c) "Where does cat 61% come from" → **RESOLVED** (`unembedding.html`, lesson 10)
My iter-2 question 7 verbatim. The page dots the final vector against every row of `E` (reuse `E` "backwards"), softmaxes, ranks. `compute()` = `TOKENS.map(t => dot(vec, E[t]))` then `softmax` (lines 230–236) — recomputed on every slider move, re-sorted. **Verified by hand:** for `[0.9,0.3,1.4,0.1]`, `tired` logit = `0.9·0.3+0.3·0.1+1.4·1.0` = 1.70; `cat` = 1.10; `it` = 1.06; denominator Σe^(z−1.70) = 3.306; `tired` = 1/3.306 = 30.3%. All seven rows match. The myth ("a special lookup / separate classifier head") vs reality ("same dot product, often tied weights") is the right correction. Genuinely closes the loop.

### (d) Temperature widget now lives inside the track → **RESOLVED** (`sampling.html`, lesson 11)
In iter-2, pages 2 and 6 name-dropped "the classic temperature widget" that lived only on the old monolith. It now has a home *inside* the track, and both callbacks were rewired: attention-scoring line 130 and 160 now link to `sampling.html` as "lesson 11," so the callback lands on something. The page delivers my explicit iter-1 ask — temperature **plus** top-k **plus** top-p on one distribution — and the JS pipeline (`z/T → softmax → top-k cut → top-p cut → renormalize`, lines 219–243) is the honest order. **Verified:** top-p=0.9 walks cumulative `0.510→0.740→0.843→0.906`, crosses 0.9 at `resting`, keeps 4 tokens vs top-k=3's exactly-3 — the "mass vs count" distinction is the whole point and it's demonstrated, not asserted.

### (e) attention-logits vs vocab-logits distinction → **RESOLVED** (`attention-scoring.html` line 129 + `unembedding.html` line 97/146)
I asked for "one line distinguishing them." I got a whole paragraph in two places. Flagship line 129: *"These are **attention-logits** — one score per token in the sentence. Later, next-token prediction produces **vocab-logits** — one score per word in the whole vocabulary. Same softmax, different lists"* with a forward link to lesson 10. Unembedding line 97 opens with the same distinction (~7 vs ~50,000) and repeats it in the hood. This is over-delivered, in a good way — a reader can no longer conflate the two lists.

### (f) The two-toy-vocab confusion on analogy → **RESOLVED** (`analogy.html` line 106)
Now flagged explicitly: *"this page uses a **separate little demo vocabulary** (man / woman / king …) with its own axes [gender, royalty, age] — not the cat / sat / it … sentence tokens … **king is not a row in the sentence table**."* That's precisely the heads-up I said was missing. (See new confusion below — page 13 quietly introduces a *third* toy vocab without the same courtesy.)

### (g) neuron "past the edges" wording → **RESOLVED** (`neuron.html` line 165)
The inaccurate "watch the sum climb past the edges" is gone. New wording: *"Drag x₁ all the way up and watch the sum grow (to about +3) — yet tanh still flattens the output near +1. Flip to no activation and … the output shoots off the chart, unbounded."* That is now factually correct — the *output without activation* is what leaves the chart (confirmed in JS: `offlab` = "off the chart, unbounded" at line 351), not the sum. Exactly the fix.

**Net: all seven iter-2 seams are RESOLVED, several over-delivered.** The one lingering iter-2 *blemish* — `Wv` shared across all heads — is unchanged in `course-data.js` (still a single `Wv`, still honestly commented at line 27, still never surfaced visually because the heads page only draws routing arcs). Not a regression; still a latent simplification a data-file reader would notice.

---

## Does the forward pass now hang together, ID → word?

Mostly yes — this is a real end-to-end story now, and I can trace most of it. Walking it as a learner: token `it` → embedding `[0.8,0.1,0.2,0.3]` (L1) → similarity/analogy/neuron primitives (L2–4) → Q/K/V and attention blend (L5–7) → **residual add-back `[1.42,0.34,0.45,0.40]`** (L8) → **FFN, added back → `[2.13,0.54,0.36,0.61]`** (L9) → unembedding dots a final vector against `E` → next-word logits (L10) → sampling reshapes them (L11) → the block, stacked (L12) → position (L13) → hallucination (L14) → tokenization blind spot (L15). Each page's pager points to the next and the prose picks up the previous page's result. The spine is there.

But three seams break the *number* chain — one meaningful, two minor-and-disclosed:

**1. The tracked vector is silently abandoned between L9 and L10 (the biggest remaining seam).** For five pages the batch follows `it`, and L9 ends by proudly building `it`'s running vector `[2.13, 0.54, 0.36, 0.61]`. Then L10 (unembedding) never touches it. Instead a **brand-new vector `[0.9, 0.3, 1.4, 0.1]` appears from nowhere** — `UNEMBED_VEC` in the data file, uncomputed, unexplained — labeled "the running vector at the last position (`was`, after attention + FFN)." This is *mechanically correct* (a decoder predicts from the **last** position, and `it` is a mid-sentence token), and L10 is honest that it's now `was`'s vector. But pedagogically it's a jolt: we spent the whole batch watching `it`, built its `[2.13]` vector, and then quietly switch tokens and unembed a magic number that doesn't derive from anything on any prior page. A learner tracing digits asks "why didn't we unembed the `[2.13]` we just made?" The page needs **one bridging sentence** — "prediction happens at the *last* token, `was`, not at `it`; here's its final vector" — to close the gap. Right now the position-switch is under-signposted and `[0.9,0.3,1.4,0.1]` is asserted, not traceable. (Note: the review prompt's own expected chain, `it → residual → FFN → unembedding`, likewise assumes `[2.13]` flows into L10 — it doesn't; the pages switch positions instead. That mismatch *is* the seam.)

**2. L10's logits don't carry into L11.** Unembedding produces vocab-logits over the 7 sentence tokens — `tired` wins at logit 1.70 / 30.3%. Sampling opens "Last page turned the final vector into vocab-logits and softmaxed them" — then uses a **completely different set of eight logits** (`tired 3.2, sleepy 2.4, hungry 1.6 …`) with new candidate words that aren't in the 7-token vocab, and `tired` is now 51%. It's flagged "illustrative candidates," but the "*same* distribution, now reshaped" framing is loose: the distribution the learner just built (tired 30.3% over 7 words) is not the one on screen (tired 51% over 8 different words). A half-sentence — "swapping to a richer candidate set to make the dials visible" — would keep the thread honest.

**3. L12's drift numbers contradict L8–L9 for the same token.** L8–L9 move `it`'s animacy `0.8 → 1.42 → 2.13` in **one** block. L12's drift toy moves `it`'s animacy `0.8 → 1.04` after one layer and only reaches `1.37` after **six** — and its attractor animacy (1.4) is *below* where one real block (L9) already landed (2.13). Same token, same `v0`, contradictory per-layer trajectory. The page discloses this loudly (the amber "This is a toy" box, and the hood repeats it): it's a compact fixed-fraction stand-in for the whole attention+FFN+residual stack, chosen so the drift is legible. So it's *honest*, and the *shape* of the story (fast early, fine-tune late; a running total) is real — but a number-tracer will notice L12 can't be reconciled with L8–L9's explicit arithmetic. Acceptable as disclosed, worth being aware of.

Nothing else contradicts. The residual→FFN handoff (L8's `[1.42…]` is literally L9's input, stated on both pages) is the one place two pages *do* share a number, and it's airtight.

---

## New confusion on the new pages

- **A third toy vocabulary, this time un-flagged (`positional-encoding.html`, L13).** After you fixed the analogy page to flag its separate `king/queen` vocab, L13 quietly introduces `POS_WORDS = {dog, bites, man}` with yet another dim convention (dims 0–1 = animacy/action = "meaning," dims 2–3 = "position") — a *third* toy space, and it gets none of the "this is a separate demo, not the sentence tokens" courtesy that analogy now has. It even re-purposes dims 2–3, which were `state`/`function` in the canonical `E`, as "spare position slots," which a careful reader will find jarring against lesson 1's `DIM_LABELS`. One heads-up line (mirroring analogy's fix) would close it.

- **L13 is out of pipeline order.** Positional encoding is an *input* step (position is added before attention, i.e. before lesson 5), but it's taught at lesson 13, after the entire forward pass and even after sampling. The page handles this gracefully as a redeemed promise ("Back on lesson 1 … position would be folded in later. This is later.") — but a learner may reasonably wonder how the L5–7 attention they already saw worked without the position that L13 says attention *needs*. It reads as an appendix, not a step. Defensible, but it's the one page that isn't where the pipeline would put it.

- **The "cat 61%" hook vs the toy's actual answer (`unembedding.html`).** Title, description and takeaway all sell "next token: cat, 61%," but the widget predicts `tired` at 30.3%. They're using "cat 61%" as a stand-in for "any explainer's confident number," and the honesty note explains the toy differs — but the mismatch between the promised 61% and the delivered 30.3% is a small friction on the one page whose whole job is demystifying that exact number.

- **L12 layer-norm is asserted but never shown.** The schematic and prose include layer-norm ("recenter to mean 0, rescale to a standard size"), the hood says it's "skipped in the toy arithmetic." That's fine and honest, but layer-norm is the one block component with *zero* compute-live moment anywhere in the 15 pages — it's the sole "recite, don't derive" leftover. Minor; a candidate for the "one more widget" wishlist, not a defect.

- **`sampling.html` bar widths are normalized to the leader, not absolute.** `fill.style.width = (final/maxFinal)*100%` (line 274) makes the top token always span the full track, so at `T=2` the flattened field still *looks* like the leader dominates until you read the percentages. The numbers are right; the bar visual slightly undersells "flatter." Cosmetic.

- **Hallucination is airtight — one nit.** `hallucination.html` is my favorite of the batch: `computeDist()` softmaxes the raw logits only and the truth flag "add[s] NO boost to the math" (verified, line 240). Verified numbers: Sydney 50.1%, Canberra (true) 15.1% at rank 3. The only nit is the prose "when a false answer is more *famous* or *frequent*" — "frequent in training" is the real mechanism; "famous" is a proxy that a literal reader might over-generalize. Tiny.

---

## New questions this batch raised

Basic → advanced; several sit on the L9→L10 and L11 seams above.

1. On the unembedding page, where did the final vector `[0.9, 0.3, 1.4, 0.1]` come from? We just built `it`'s `[2.13, 0.54, 0.36, 0.61]` on the FFN page — why isn't *that* the thing we turn into a word?
2. Why do we predict from `was` (the last token) and not from `it`? Does every token get a next-word prediction, or only the last one?
3. If every position produces logits, are the other six predictions just thrown away during generation? (I think during training they're all used — is that the difference?)
4. The unembedding "reuses `E` transposed / tied weights." Does reusing the input table to *also* score outputs ever cause problems, or is it just a free parameter saving?
5. Residual adds attention's output back on — but doesn't the vector just keep growing every layer (0.8 → 1.42 → 2.13 …)? Is that what layer-norm is quietly fixing?
6. The FFN is 4→8→4 here. In a real model is the hidden layer bigger than the vector (I've heard "4× wider")? Why widen then shrink back?
7. Sampling's eight candidate words (`sleepy`, `hungry`, `resting`…) aren't in the 7-token vocab from every other page. Are these a different vocabulary, or did the "vocab" quietly grow?
8. Temperature divides logits by `T` *before* softmax; top-k/top-p cut *after*. If I stack all three, does the order matter — would top-p on a high-temperature distribution keep more tokens?
9. The drift page says animacy climbs toward 1.4 over six layers, but the FFN page already pushed it to 2.13 in one block. Which trajectory is real — or is neither, and the real one is just un-drawable?
10. Position lives in "spare dimensions the meaning barely uses." In a real model with hundreds of dims, are some dims *permanently* reserved for position, or does the model learn to share them?
11. If position is added at the input (L13), how did the attention pages (L5–7) work at all — were they secretly using position, or is order genuinely ignored there?
12. Hallucination: `Sydney` beats `Canberra` because it's more frequent in training. Does that mean feeding the model *more* correct text about Canberra would flip the logits — i.e. is hallucination fixable by data alone?
13. The tokenization page says letters are "gone" after tokenization. Then how do models ever spell words correctly, or handle a brand-new made-up word they've never seen?
14. If a real BPE tokenizer chunks by frequency, does the *same* word tokenize differently depending on surrounding text, or is the split fixed per word?
15. Across the whole forward pass, which numbers are *frozen after training* (the `E` table, `Wq/Wk/Wv`, `W1/W2`, layer-norm scales?) and which are *recomputed per token per layer* (the residual stream vector)? I think I finally know but I'd like it stated once.
16. Each transformer block "has its own learned weights." For a 90-layer model that's 90 separate attention+FFN weight sets — is that where most of the "billions of parameters" actually live?
17. The unembedding and the embedding can be the *same* table. But the FFN and attention weights differ every layer. So is the embedding table the only thing shared end-to-end?
18. Speculative decoding, KV cache, quantization — none of these appeared. Are they *how the model runs* rather than *what it computes*, i.e. do they leave every number on these 15 pages identical?
19. If truth is "never a term in the computation" (L14), is that exactly what RAG / tool use / chain-of-thought are trying to bolt on from outside — and would those be the natural next lessons?
20. A real block also has a causal mask so a token can't attend to its future. The bidirectional example from L6 is still there — at what layer/step does the mask actually kick in during generation?

---

## What (if anything) to build next — prioritized

Part 1 ("How LLMs Work," ID → word) is now **complete and self-contained.** It's worth one more batch, and the clear target is **(a): decompose the monolithic Part 2 `llm-inference-efficiency.html` ("How LLMs Run Fast") into the same one-idea-per-page, compute-live style.** Rationale: it already exists as a wide narrated tour (still linked from the hub as "the classic walkthrough"), the questions above (15–18) show learners hit the "what's frozen / how does this *run*" wall right at the end of Part 1, and every page below reuses primitives already shipped (dot product, softmax, matvec, the neuron, bytes-of-a-vector). It converts "what the model computes" into "how the computation is made fast" — the natural Part 2.

A buildable **7-page batch**, each with ONE idea + ONE compute-live widget:

1. **Prefill vs decode.** Idea: the prompt is processed in one parallel pass (prefill); generation then runs one token at a time (decode). Widget: type a prompt length + gen length, compute the two FLOP costs live and show why decode dominates wall-clock.
2. **The KV cache.** Idea: decode reuses each earlier token's stored Key/Value instead of recomputing them every step. Widget: toggle cache on/off and compute the redundant K/V recomputations (step *t* redoes *t−1* tokens without it) — the triangular-number blow-up, live.
3. **KV cache memory.** Idea: the cache grows linearly with sequence length and is what actually eats VRAM at long context. Widget: sliders for layers × heads × head-dim × seq-len × 2 × bytes → cache size in MB/GB, computed live (answers Q5/Q15).
4. **Quantization.** Idea: store weights in fewer bits (fp16 → int8 → int4), trading precision for memory and bandwidth. Widget: quantize a real weight vector to N bits, show per-element rounding error and total bytes saved, computed live.
5. **Paged attention (KV paging).** Idea: store the cache in fixed-size pages so variable-length sequences don't waste memory on padding. Widget: pack a few different-length sequences into pages, compute used vs wasted slots for contiguous vs paged, live.
6. **Continuous batching.** Idea: many requests share one forward pass, and finished sequences are swapped out mid-flight so the GPU never idles. Widget: a batch timeline of mixed-length requests; compute tokens/sec for static vs continuous batching, live.
7. **Speculative decoding.** Idea: a small draft model proposes *k* tokens, the big model verifies them in a single pass; accepted tokens come nearly free. Widget: draft proposes, target accepts/rejects, compute expected speedup from an acceptance-rate slider, live.

Optional 8th (bridges to the iter-1 GPU wishlist and makes the batch *why*-complete):

8. **Compute- vs bandwidth-bound (the roofline).** Idea: decode is bottlenecked by *moving weights*, not by math — which is *why* KV cache, quantization and batching all help. Widget: set batch size, compute arithmetic intensity and whether you're compute- or bandwidth-bound, live. This is the single best answer to Q18 and ties the batch together.

The remaining iter-1 wishlist topics (tool use / function-calling, chain-of-thought / reasoning, RAG vs fine-tune vs prompt, multimodal) are a strong **Part 3** — and Q12/Q19 show learners are primed for them right after hallucination — but they're a different mode (systems *around* the model, less "watch the number") and shouldn't jump the Part 2 queue.

---

## Verdict

**The full 15-lesson "How LLMs Work" track is SATISFACTORY — I would stop passing it back for redesign.** All seven iter-2 seams are fixed (several over-fixed), every widget I checked computes live and the arithmetic is correct to the decimal (FFN, unembedding, drift, sampling, hallucination all verified by hand against `course-data.js`), and the forward pass now genuinely closes ID → word. The one meaningful rough edge — the un-signposted token/vector switch at L9→L10 (`it`'s `[2.13]` abandoned, `was`'s `[0.9,0.3,1.4,0.1]` appears uncomputed) — is a *one-sentence copy fix*, not a redesign; it's mechanically correct, just under-explained. Ship it.

**On the overall course (this track + the two classic overview pages): not done — one more batch is clearly worth doing.** Part 1 stands on its own, but the hub still points at a monolithic Part 2 (`llm-inference-efficiency.html`) that never got the one-idea-per-page, compute-live treatment, and the learner questions this track *ends* on (what's frozen, how does this actually run, why is it slow) are exactly Part 2's material. Decomposing "How LLMs Run Fast" into the 7–8 pages above is the obvious, high-payoff next iteration. After that, the review→build loop can reasonably stop (Part 3 topics are optional polish, not a gap).
