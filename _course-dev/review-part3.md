# Learner Review — Part 3 "Beyond the Basics" (8 new lessons)

*Same reviewer: junior SWE, Python/JS, zero ML background. Across iter-1/3/4 I signed off Part 1 (15 lessons) and Part 2 (8 lessons) and left a wishlist of "systems-around-the-model" topics I kept asking for. Part 3 is those topics. I read all 8 pages prose-first, poked every widget, then read each page's `<script>` + `course-data-p3.js` to check computed-vs-replayed. I hand-verified every headline number the prompt named, plus a few it didn't.*

---

## Did Part 3 close the wishlist?

Every topic I repeatedly asked for is now a page with a live widget. Verdict per topic, with the page, what the widget actually does, and my hand-check:

| Wishlist topic (where I asked) | Verdict | Page + what the widget computes |
|---|---|---|
| **How a GPU actually works** (iter-1 top wish; iter-4 hardware Qs) | **COVERED** | `gpu-anatomy.html`. A "does it fit?" log-ruler: pick weights / KV / score-matrix / Flash-tile, toggle fp16↔int4, drag seq; every item re-evaluates its byte formula and re-decides which tier (registers/SRAM/VRAM) lights, via `fitsInSram`. Not replayed — `update()` recomputes bytes and the tier on every input. |
| **GQA** (iter-4 Q1) | **COVERED** | `grouped-query-attention.html`. Toggle MHA→GQA-8→GQA-4→MQA, 32 query dots re-bracket into n_kv groups, cache recomputed from `gqaKvCacheBytes`; MHA baseline bar + 80 GB line. |
| **FlashAttention** (iter-1; iter-4 Q2) | **COVERED** | `flash-attention.html`. Drag n and tile B, fp16↔int8; naive `scoreMatrixBytes(n)` vs `flashTileBytes(B,d)`, a live "won't fit SRAM" badge when B×B blows the 192 KiB budget, plus a genuine **online-softmax** explainer (running max m, running sum ℓ, rescale by e^(m−m′)). |
| **MoE routing — "who routes and how"** (iter-1 Q17; iter-4 Q10) | **COVERED** | `mixture-of-experts.html`. Pick a token → `moeRoute(E[token],k)` runs the real `matvec(gate,·)`+`softmax`+top-k; a *different* expert pair fires per token; `moeParams` recomputes active-vs-total. The router is a live dot-product-then-softmax, exactly the "who routes" I asked for. |
| **Tool use / function calling** (iter-1; iter-3 Q19) | **COVERED** | `tool-use.html`. A 5-phase stepper (reason→call→PAUSE→inject→answer); the tool **actually runs** (`runWeather(city)`), and the injected JSON + resumed sentence rebuild from the picked city via `toolResultTokens`/`toolAnswer`. |
| **Reasoning / chain-of-thought** (iter-1) | **COVERED** | `reasoning.html`. Drag thinking-tokens; `reasonAcc(t,preset)` draws a live saturating curve with a moving operating point, a marginal-gain card, ghost curves for the other two difficulties, and a decode-step latency cost. |
| **RAG vs fine-tune vs prompt** (iter-1; iter-3 Q19) | **COVERED** | `rag.html`. Live `cos(query,passage)` over a 6-passage library (`ragRetrieve`), re-ranks and rebuilds the assembled prompt; a 3-way prompting/RAG/fine-tune decision card sits up top. |
| **Multimodal** (iter-1) | **COVERED** | `multimodal.html`. Editable 4×4 image → 2×2 patches; `patchEmbed(pixels)=matvec(WPATCH,·)` recomputes on every pixel edit, with a traceable per-dim matvec, a position toggle, and the patch dropped into a shared stream beside `cat`/`sat` text vectors. |

**8 / 8 covered — no MISSING, no PARTIAL.** This is the cleanest wishlist-closure of the three build passes.

### Hand-verified numbers (against `course-data-p3.js`)

- **MoE `cat`→e0,e4.** `E[cat]=[1.0,0.2,0.1,0]`; `matvec(gate,·)=[3.0,0.6,0.3,0,1.8,1.65,0.65,−1.3]` → top-2 e0(3.0), e4(1.8) → `softmax([3.0,1.8])=[0.769,0.231]` → **e0 76.9%, e4 23.1%** ✓ (hood matches).
- **MoE 47.4B / 13.6B.** embed 131.07M; attn/layer 4·4096²=67.11M; router 8·4096=0.033M; expert 3·4096·14336=176.16M. total = 131.07M + 32·(67.11+0.033+8·176.16) = **47.38B → 47.4B**; active (k=2) = 131.07M + 32·(67.11+0.033+2·176.16) = **13.55B → 13.6B**; activeFrac 0.286 → **29%** ✓.
- **GQA 32→8 = 4×.** `gqaShrink = n_heads/n_kv = 32/8 = 4×` ✓; per-token 2·32·8·128·2 = 131,072 = **128 KiB**; @seq 8192,batch 1 = **1.0 GiB / 1.07 GB** ✓; MHA baseline 512 KiB·8192 = 4.0 GiB/4.29 GB ✓.
- **Flash 32 MiB→56 KiB.** `scoreMatrixBytes(4096)=4096²·2=33,554,432=32 MiB`; `flashTileBytes(64)=(3·64·128+64²)·2=(24576+4096)·2=57,344=56 KiB`; reduction (4096/64)²=**4096×** ✓. n=8192 → 128 MiB ✓.
- **RAG cosines.** "cook onions" [0,0.1,1,0] vs P2 [0,0.1,1,0] = 1.01/(1.005·1.005)=**1.00** ✓, P5 **0.989** ✓; "biggest planet" [0,0.1,0,1] vs P3 [0,0.2,0,1]=**0.995** ✓, next P1 **0.198** ✓; "cats sleep" vs P0 **0.995**, P4 **0.944** ✓.
- **Multimodal patch [0.5,0,0,0.8].** patch (0,0) pixels [0.9,0.1,0.1,0.9]; WPATCH·pixels → brightness 0.25·2.0=**0.5**, LR-edge 0, TB-edge 0, diagonal 0.45−0.05−0.05+0.45=**0.8** → **[0.5,0,0,0.8]** ✓.
- **GPU-anatomy** KV@4k = 512 KiB·4096 = **2.15 GB**, weights int4 = 6.607e9·0.5 = **3.30 GB**, 19/2.0 = **9.5×**, decode fetch 13.2e9/2.0e12 = **6.6 ms** ✓.
- **Reasoning** `reasonAcc(120,typical)=0.30+0.55·(1−e⁻¹)=0.648=64.8%`, marginal last-60 = 64.8−51.6 = **+13.1** ✓; decode 13.2 GB/2 TB/s = **6.61 ms/token** ✓.

Every number I checked is correct to the decimal, and every widget recomputes — none replays a hard-coded headline.

---

## Do the four "soft" pages actually teach interactively (not prose)?

This was my worry going in — tool-use / reasoning / rag / multimodal are "systems around the model," a different mode from "watch the number." Verdict: **three are genuine mechanism-revealing computation; one (tool-use) is a genuine stepper rather than a live calculator, which is the right choice for its concept.**

- **`rag.html` — genuinely computes.** `render()` runs `cosine(query.vec, p.vec)` over all 6 passages on every click (line ~270), re-sorts, rebuilds the bars, the #rank badges, and the *assembled prompt* panel from the retrieved passages. This is live cosine retrieval, not a diagram — it's Part-1 lesson-2's operation looped over a library. Best of the four.
- **`multimodal.html` — genuinely computes.** `patchEmbed(pt.pixels)=matvec(WPATCH,pixels)` runs on every pixel edit / patch pick / position toggle (line ~380). The matvec is shown per output dim as traceable arithmetic (`0.25·0.9 + …`), the embedding slots animate in, and the patch vector is dropped into a stream next to real `E['cat']`/`E['sat']` vectors of the same shape. A real live patch→vector projection.
- **`reasoning.html` — genuinely computes a live curve.** `drawPlot()` redraws the whole SVG from `reasonAcc(t,preset)` (not a stored polyline); the operating dot, marginal-gain card, ghost curves and difficulty caption all recompute. It's honest that the *curve shape* is a toy (loud "This is a toy" box) — but the interaction is a live-evaluated function, not a replayed animation.
- **`tool-use.html` — a genuine stepper, not prose, and correctly *not* a calculator.** There's no real model here (there can't be), so the model's reason/call/answer tokens are scripted strings. But the load-bearing part — the tool — actually runs (`runWeather`), and the injected JSON + resumed sentence rebuild from the picked city (`buildSeq`, line ~252): switch Paris→Oslo and both the cyan injected `{"tempC":6,"sky":"rain"}` and the answer sentence change. The mechanism it reveals (predict-a-call → PAUSE → external code runs → inject tokens → resume, with a phase/actor/token-count status strip) is exactly the hand-off, revealed by stepping. This is the one page that is a stepper rather than a live formula — appropriate, since the insight is *control flow*, not a number.

None of the four slipped into prose-with-a-diagram.

---

## Does Part 3 connect back to Parts 1–2?

Mostly the ties land as real dependencies, not name-drops. Page by page:

- **gpu-anatomy → Part-2 L1 (bandwidth-bound), L3 (KV cache), L8 (roofline).** Lands hard — the hood literally re-derives the 6.6 ms decode fetch from Part-2 L1 and calls itself "the roofline rendered in hardware." Anchors the whole Part-2 mental model in physical memory. ✓
- **GQA → Part-1 L6 (MHA/attention), Part-2 L3 (KV cache).** Lands — explicit "tie-back: `kv-memory.html` named this lever… this page is that lever with the dial in your hand," and the shrink is the exact `n_heads/n_kv` the kv-memory hood mentioned. ✓
- **FlashAttention → Part-1 L6 (`attention-scoring`), gpu-anatomy L1, roofline.** Lands — the online-softmax explainer explicitly says the answer is "identical to Part 1 lesson 6; only the memory schedule changed," reusing the SRAM budget from L1. ✓
- **MoE → Part-1 L9 (FFN), L6 (softmax router), roofline.** Lands — "back in lesson 9 every token ran through one FFN; MoE replaces that single FFN with N," and the router is explicitly "the lesson-6 move again (dot product then softmax)." ✓
- **Reasoning → Part-1 myth-buster, Part-2 L1 (decode step).** Lands — opens on Part-1's "the math for one token is trivial" and prices every thinking token as one Part-2 decode step (6.61 ms). ✓
- **Tool use → Part-1 L14 (hallucination), "lesson 1."** Hallucination tie lands well ("a tool injects ground truth the frozen weights never stored"). **Minor slip:** it calls the model "the same next-token predictor from Part 1 lesson 1" and links `embeddings-lookup.html` — but in the decomposed 15-lesson track, lesson 1 is *embeddings* ("where does the vector come from"), not next-token prediction. The spirit is fine; the lesson-number attribution is loose.
- **RAG → Part-1 L2 (similarity), L14 (hallucination), "train vs run" frozen weights.** Frozen-weights and hallucination ties land. **Minor slip:** RAG repeatedly calls retrieval "cosine similarity — the exact operation of Part 1 lesson 2," but lesson 2 (`embedding-space.html`) ranks by **dot-product** similarity, not cosine (the hub blurb says "dot-product similarity"). Cosine is normalized dot-product — a close cousin, and the `cosine` helper does live here — but "the exact operation of lesson 2" overstates the identity.
- **Multimodal → Part-1 L1 (embeddings), L13 (positional), L5–7 (attention).** Lands — patch projection is explicitly "the image's tokenizer… learned, just like the embedding table from lesson 1," position uses the L13 idea, and the shared stream is eaten by L5–7 attention "unchanged." ✓

So six of eight ties are real dependencies; two (tool-use "lesson 1," RAG "lesson 2 = cosine") are one-word attribution slips, not conceptual errors.

---

## New confusion / errors on the Part-3 pages

Being fair about disclosed toys (single 7B/A100 preset, hand-authored gate, geometric reasoning curve), the only things I'd flag — all minor:

- **RAG conflates cosine with lesson-2's dot-product (real, small).** As above: "the exact operation of Part 1 lesson 2" is inexact — lesson 2 used raw dot-product; this page uses cosine. A learner who does lesson 2 by hand and lesson 7 by hand gets two different formulas under one "it's the same operation" claim. One word ("the normalized version of lesson 2's similarity") fixes it.
- **MoE's gate matrix is hand-planted but reads as representative.** The routing computation (`matvec`+`softmax`+top-k) is genuinely live, but the gate values themselves (`[3,0,0,0]`, `[1.5,1.5,0,0]`, …) are authored so experts map cleanly to animacy/action/state. The page says "the router is a *learned* matrix" in general, but never says *these specific clean weights are a designed demo* — so a careful reader might think real routers have interpretable one-hot-ish gates. Same courtesy the analogy/positional pages got for their toy vocabs would close it.
- **MoE 47.4B/13.6B vs real Mixtral 46.7B/12.9B.** Disclosed as "Mixtral-8×7B-*shaped*," so it's honest, but the numbers are a hair off the model it names; a reader who looks up Mixtral sees 46.7B/12.9B. Cosmetic.
- **Tool-use model text is canned.** The reason/call/answer tokens are fixed strings (no model exists to generate them); only the tool result + answer-from-result are computed. Correct and unavoidable, and the page frames the model's output as illustrative — but it never quite says "the model's words here are scripted; the tool call and its result are the live part." A one-liner would prevent a "wait, is the model real?" double-take.
- **Flash "K/V passes = 64" is subtle and, to its credit, *not* overclaimed.** The page reports the (n/B)² *peak-score-memory* reduction, and separately that K/V are re-streamed once per Q-block (`⌈n/B⌉` passes) — so it does **not** falsely claim total HBM traffic drops 4096×. This is actually more careful than most explainers; I only flag it because a fast reader might read "4096× less memory" as "4096× faster," which the page's "peak score memory" wording guards against but doesn't hammer.

Nothing here is a broken widget or a wrong computed number. Cross-links (pager prev/next across all 8, hub `#part3` list, eyebrows "Lesson X/8") chain cleanly index → 1 → … → 8 → index.

---

## New questions this batch raised (basic → advanced)

1. GQA shares K/V across query heads — does the model *learn* which query heads should share, or is the grouping just fixed by index (heads 0–3 share group 0, etc.)?
2. On the tool-use page the harness stops on the closing `}` — what happens if the model emits *malformed* JSON, or never closes the brace? Does the whole thing hang?
3. Tool use injects a *true* fact — but the model could still ignore it and hallucinate anyway. Does injecting context *guarantee* the answer uses it, or just make it likely?
4. Reasoning buys accuracy with more tokens — but who decides *when to stop thinking*? Is the stop itself a learned token the model emits, or an external budget cap?
5. The reasoning curve plateaus — but the page notes some tasks *dip* if the model overthinks. What makes accuracy go *down* with more thinking?
6. RAG retrieval is cosine over embeddings — but who made the passage embeddings, and is that embedding model the *same* transformer, or a separate smaller one trained just for similarity?
7. RAG says "top cos < 0.9 → thin match → closer to guessing." In a real system, does it *know* the match is thin and say "I don't know," or does it inject the weak passage and confidently answer anyway?
8. Multimodal projects a patch through `W_patch` — is that matrix trained *jointly* with the language model, or is the vision part pre-trained separately and then bolted on?
9. If image patches and text tokens share one stream, can the model attend from a text token *to* an image patch and back? Is that how "describe this picture" works — text queries attending over patch keys?
10. FlashAttention keeps the exact same answer — so is there *ever* a reason not to use it? Why was naive attention ever the default?
11. MoE stores 47B but computes 13.6B — but all 47B still have to *fit in VRAM*. So MoE saves compute/FLOPs but **not** memory? Does that change the roofline story (bandwidth vs compute) versus a dense model?
12. In MoE, if two tokens in the same sentence route to different experts, are those experts run in *parallel* on the GPU, or one after another — and does uneven routing (everyone picks e0) stall things?
13. GQA, FlashAttention, MoE — are these baked into the *weights* (so you pick them at training time) or can you switch them on at serving time like KV cache / quantization?
14. Reasoning tokens are a "visible scratchpad" — if I can *read* the model's chain-of-thought, can the model also lie in its scratchpad and still reach the right answer? Is the visible thinking always faithful to the real computation?
15. Multimodal says "audio becomes frames the same way." Does that mean video is just *even more* tokens (patches × frames), and is that why long videos blow the context window instantly?

---

## Verdict

**(1) Is Part 3 itself SATISFACTORY?** **Yes — I would not pass it back.** All 8 wishlist topics shipped as one-idea-per-page live widgets in the established format; I hand-verified every headline number the prompt named (MoE e0/e4 + 47.4B/13.6B/29%, GQA 4×, Flash 32 MiB→56 KiB/4096×, three RAG cosines, the multimodal [0.5,0,0,0.8] patch) plus gpu-anatomy and reasoning, all correct to the decimal; the four "soft" pages teach by genuine computation (rag/multimodal/reasoning are live-evaluated, tool-use is a proper stepper) rather than prose; and the ties back to Parts 1–2 are real dependencies. The only findings are two one-word attribution slips (tool-use "lesson 1," RAG "lesson 2 = cosine") and a couple of undisclosed-toy nits (planted MoE gate, canned tool-use model text) — all copy-polish, not redesign.

**(2) Is the now-31-lesson, three-part course COMPLETE / shippable?** **Yes — it's done. Ship it.** The course now runs end-to-end from "a token ID" (Part 1) → "why it's served in milliseconds" (Part 2) → "the GPU it runs on and every capability pattern an assistant uses" (Part 3), every number traceable, every widget computing live. Part 3 closed the entire wishlist I'd been carrying since iter-1. There is **no remaining blocking gap.** The open questions this batch raised (multi-GPU/tensor parallelism, faithfulness of chain-of-thought, joint-vs-bolted vision training) are genuine next-frontier curiosities a well-taught learner *should* now be asking — they're evidence the course succeeded, not holes in it. The review→build loop can stop here.
