# Learner Review — "Meaning & Attention" mini-track (iteration 2)

*Same reviewer: junior SWE, Python/JS, zero ML background. I read all 8 new pages, poked every widget, and read every page's `<script>` + `course-data.js` to check computed-vs-replayed. I also hand-verified several of the "show me the actual numbers" tables against the shared ops so I could be fair about the "it genuinely computes" claim.*

---

## Did iteration 1's fixes land?

My five original headline complaints, judged against this new track:

### 1. Attention taught by analogy / hard-coded arc weights → **RESOLVED**
This was the #1 thing I bounced off, and it's genuinely fixed. In iter-1 the arc widget had `aWeights` with `"it"→cat = 0.72` typed in as a constant. That's gone. On **attention-scoring.html**, `compute()` does the real thing every click:
```
const q = matvec(Wq, E[query]);
const scores = TOKENS.map((t,j)=> dot(q, K[j]));   // q·k logits
const weights = softmax(scores);                    // e^z / Σe^z
const out = [0,1,2,3].map(d => TOKENS.reduce((s,_,j)=> s + weights[j]*V[j][d], 0));
```
I verified the pipeline by hand from `course-data.js`: `q_it = [1.9,0.2,0.2,0.3]`, `q·k_cat = 1.9·1.0+0.2·0.2+0.2·0.1 = 1.96`, `q·k_it = 1.607`, and the `e^score` column sums to ~19.83 so `cat = 7.10/19.83 = 35.8%`. The hood table matches to the decimal. The Stage-3 arc `stroke-width = 1.5 + o.w*26` is driven straight off `softmax` output — thickness *is* the computed weight, not a diagram of a conclusion. This is exactly the widget I asked for in iter-1 ("I want the calculation, not the arcs"), and it delivers.

### 2. Decorative mid-Part-1 widgets (animate pre-baked numbers) → **RESOLVED**
Every widget on the track computes from the shared `E` / `W` data, and I traced each one:
- **embeddings-lookup.html**: `setReadout` reads `E[TOKENS[i]]` by index; the hood even makes the point explicitly ("`E[TOKENS[4]]` fetched by index every time … addressed, not memorized"). It's a lookup demo that is *itself* a lookup.
- **embedding-space.html**: `render()` loops `cosine(a, E[t])` and `dot(a, E[t])` over live `E` and re-sorts. I checked `cos(it,cat)=0.93` against the hood's `0.84/(0.883·1.025)` — correct. Bars re-rank when you change anchor; nothing stored.
- **analogy.html**: `r = va.map((x,i)=> x - vb[i] + vc[i])` then a real nearest-neighbour search over `VOCAB` excluding A/B/C. This is precisely the "live analogy calculator" I begged for in iter-1 ("let me *do* the subtraction"). Change any dropdown and the winner changes; off-target combos honestly return a lower-cos word instead of faking `queen`.
- **neuron.html**: `Math.tanh(x.reduce((a,xi,i)=>a+xi*w[i],0)+b)` recomputed on every slider input, with per-product tiles (`0.90 × 1.3 = 1.17`) surfaced — the "show me one multiplication" ask from iter-1, done.

### 3. Crammed chapters (3+ ideas per section) → **RESOLVED (for this track)**
The decomposition is the headline win. Old §06 (Q/K/V + weighting + softmax + multi-head, four ideas in one scroll) is now **three** pages: qkv → scoring → heads, each one idea + one widget with a crisp "we do NOT score them yet" boundary on the qkv page. The neuron got its own page. Old §04's forward-pass/training/param-scale pileup is gone (those ideas simply aren't in this track yet — see scope note below). Pages are ~4–7 min each with a single takeaway. This is the one-idea-per-page structure I sketched in iter-1.

### 4. softmax / logits / dot-product never defined → **RESOLVED**
All three are now named and defined the first time they matter:
- **dot product**: defined on embedding-space.html with `a·b = Σ aᵢbᵢ` and a worked example.
- **softmax**: defined on attention-scoring.html as `e^z / Σe^z`, *and* shown — Stage 2 prints a running "Σ = 100.0%" so I can see the fractions summing to 1.
- **logits**: finally named — "Those raw scores are the logits of the attention softmax" — and tied to next-token logits ("the exact same softmax that turns next-token logits into probabilities").
- Bonus: **activation function** ("squish") is named and *motivated* on neuron.html ("without it, ten layers collapse into one straight line") — my iter-1 "why squish at all?" is answered.

### 5. Overall "recite not derive" → **RESOLVED for everything the track covers**
I can now trace a number end-to-end: ID → row → vector (pg1) → similarity (pg2) → direction arithmetic (pg3) → the multiply-add-squish primitive (pg4) → Q/K/V matvec (pg5) → q·k→softmax→blend (pg6) → three heads re-routing (pg7). Every page has a "Show me the actual numbers" hood with honest worked arithmetic. I came away able to *derive*, not just recite. The only residual "recite" feeling is about the parts that aren't built yet (what the blended vector becomes) — that's a scope gap, not a widget that lies. See below.

**Net:** all five original complaints land as RESOLVED on this track. The remaining work is *additive* (more pages), not *corrective* (fixing these).

---

## New confusion on the new pages

Things that are now newly confusing, mostly at the seams the decomposition created:

- **The track stops one step too early — the blended vector is left dangling.** attention-scoring.html Stage 4 ends with "it's no longer the raw embedding; it's a blend of what it attended to" and… that's it. Nowhere does any page say what *happens* to that vector: does it replace the embedding, get added back (residual), feed the next layer, go through neurons? This is the single biggest seam. The neuron page (pg4) taught me the FFN primitive but nothing ever *uses* it on a real token vector, so pg4 sits islanded before attention with no payoff. A reader finishes the flagship page holding a 4-number vector and no idea how it becomes a prediction.

- **Two different toy embedding spaces, never reconciled.** Pages 1–2 and 5–7 use the 7-token sentence with dims `[animacy, action, state, function]`. Page 3 (analogy) silently switches to a *different* vocab (`man/woman/king…`) with *different* dims `[gender, royalty, age]`. Both are called "the lookup table / real word vectors." A careful learner will ask "wait, is `cat` in the same table as `king`? why did the axes change?" It's justified (you need a clean gender axis for the analogy to land exactly), but nothing on pg3 flags that this is a separate demo vocabulary.

- **The neuron page's inputs are abstract, disconnected from tokens.** neuron.html uses `NEURON.x = [0.90,0.30,0.60]` — three anonymous numbers, not a token's embedding. So "billions of these are the model" is asserted but I can't place *where* a neuron sits. Coming right after three pages about token vectors, I expected the neuron to eat an embedding.

- **"past the edges" overstates what the neuron widget shows.** The stage-note says "Drag x₁ all the way up and watch the sum climb past the edges." With the sliders capped at `[-1,1]` and `w=[1.3,-0.9,0.7]`, the max achievable sum is ~3.0, and the X-domain is ±3.2 — so the sum never actually reaches the edge, let alone passes it. (The *output without activation* does shoot off the Y-axis, which is the real point — but the "sum past the edges" wording is inaccurate and I went looking for an overflow that doesn't happen.)

- **"logits" is used for two different things without flagging it.** Page 6 calls the attention *scores* "logits," and also says they're the same as *next-token* logits. Generically that's fine (any pre-softmax score), but a learner may conflate "attention logits" (n numbers, one per token) with "vocab logits" (50k numbers, one per word) as the same object. One line distinguishing them would help — especially since the track never actually shows vocab logits.

- **Cross-links point outside the track.** Pages 2 and 6 both lean on "the classic temperature widget" as if I've seen it ("the exact same softmax the temperature widget uses"). But that widget lives on the old monolithic `llm-visual-course.html`, not in this 7-page track. A learner who started here has never met it, so the callback lands on nothing.

- **Value matrix is shared across heads (latent inaccuracy).** `course-data.js` defines `Wq2/Wk2/Wq3/Wk3` for heads 2 and 3 but reuses the single `Wv` for all heads ("value = … shared by all heads"). Real multi-head attention has a separate `Wv` per head. The heads page dodges this by only drawing routing arcs (not the value blend), so it never surfaces visually — but it's technically wrong and a reader of the data file would notice the missing `Wv2`/`Wv3`.

- **Bidirectional attention on a generation-flavored example.** On pg6, query `it` attends to `was` and `tired`, which come *after* it in "The cat sat because it was tired." The page honestly flags the causal mask as "a later topic," which is good — but using a forward-looking attention on a sentence that reads like next-token generation may plant the idea that a generating model can see its future, exactly the misconception the mask exists to prevent.

---

## New questions this track raised

Ordered basic → advanced; several sit right on the seams above.

1. On the neuron page, what *are* `x₁,x₂,x₃`? Are they a token's embedding numbers, or something unrelated? Where does this neuron physically sit in the model?
2. Page 1 says "no human labels these axes" and then labels them animacy/action/state/function. So are those real learned axes or made up for readability? (I think made up — but the page half-claims both.)
3. The analogy page uses different words *and* different dimensions than the sentence pages. Is `king` in the same lookup table as `cat`, or is that a second, separate table?
4. The neuron squishes with `tanh`; attention's `q = Wq·e` is multiply-add with **no** squish. Why does one multiply-add get an activation and the other doesn't?
5. **The blended output vector at the end of page 6 — what happens to it next?** Does it replace the token's embedding, or get added back onto it, before the next step?
6. If `it`'s vector becomes "cat-flavored" after attention, does the model now literally treat `it` as `cat`? How does that help it predict the next word?
7. The whole track never shows a next-token prediction. Where does "cat 61%" actually come from — is it another dot product, of the final vector against the embedding table?
8. Multiple heads each produce an output vector for the same token. How are those combined into one — concatenated, averaged, added?
9. Where do `Wq/Wk/Wv` come from? Are they learned like the embedding table, and does each layer have its own set?
10. Real vectors have hundreds of dims and a model has many heads — is the number of dims related to the number of heads, or independent?
11. The lookup table is "frozen" after training, but attention rebuilds each token's vector at every layer. So what exactly is frozen — the table and the `W` matrices, but *not* the per-token vectors flowing through?
12. Page 6 divides scores by `√d` "to keep numbers from getting too large." What breaks if they get large — does softmax saturate to a one-hot?
13. Softmax forces the weights to sum to 1, so attention only *redistributes* existing Values. Where does genuinely *new* information enter the vector, if the output is always a mixture of inputs?
14. Positional encoding is mentioned as "meaning + position, literal addition" but never shown. If you add a position vector onto the embedding, doesn't that clobber the meaning dimensions? How are the two kept apart?
15. In a real generating model, would `it` be *forbidden* from attending to `tired` (a future token) by the causal mask? Does the mask change which token wins?
16. How many attention layers does a real model stack, and does each layer relearn its own heads, or is it the same heads repeated?
17. Attention "logits" are one-per-token; next-word "logits" are one-per-vocab-word (~50k). Are these the same kind of number, or just the same softmax applied to two different lists?
18. The Value uses one shared `Wv` across all heads in the data file. Do real models share the Value matrix across heads, or is that a demo simplification?
19. If temperature reshapes a softmax, does *attention* have a temperature dial too — or is `√d` the closest equivalent?
20. Each layer refines the vector — is `it`'s vector at layer 40 meaningfully different from layer 1, and can you watch it drift?

---

## What to build in iteration 2 (prioritized)

The track nails "meaning & attention" but *ends on a cliff* — the blended vector goes nowhere. The highest-value batch is the one that **closes the forward pass**, because it makes the track self-contained (ID in → word out) and it reuses primitives already built (dot product, matvec, the neuron, softmax). Pages 1–6 below do that; 7–8 are the two most-requested standalone topics that build on this foundation.

**Tier A — finish the forward pass (build these first, in this order):**

1. **Where the blended vector goes: the residual stream.** ONE idea: attention's output is *added back* onto the token's own vector (refine, don't replace), so the vector is a running total across layers. Widget: show `e + Δattn = new vector` element-wise with an "add vs replace" toggle that demonstrates replacing throws away the token's identity. *This is the missing next-sentence after page 6 and should ship first.*

2. **The feed-forward step: attention mixes tokens, the FFN thinks per-token.** ONE idea: after attention, each token's vector runs *independently* through a little neuron network. Widget: run the page-4 neuron across all 4 dims of one token's post-attention vector, live — literally the neuron page finally applied to a real token, closing pg4's dangling primitive.

3. **From vector to prediction: the unembedding.** ONE idea: the final vector is dot-producted against every row of the embedding table to score all vocab tokens → those are the next-token logits. Widget: dot the output vector against all 7 rows of `E`, rank them, watch the winner emerge — reuses page-2's dot product and finally answers "how does any of this predict a word." *Highest single-page value in the batch.*

4. **Logits → probabilities: softmax, temperature, top-p, top-k.** ONE idea: the vocab logits become a probability distribution you can reshape. Widget: the softmax bars over the vocab with temperature **plus** top-p/top-k sliders (my explicit iter-1 ask) — reuses the softmax primitive, and gives the "classic temperature widget" the pages already reference an actual home inside the track.

5. **The transformer block, stacked.** ONE idea: attention + FFN + residual (+ a one-line layernorm mention) is one block, repeated N times. Widget: a layer slider that re-runs the toy pipeline and shows `it`'s vector drifting from `[0.8,0.1,0.2,0.3]` toward cat-flavored over layers — answers question 20 directly.

6. **Positional encoding: meaning + position, added.** ONE idea: literal element-wise addition of a position vector (redeems page 1's deferred promise). Widget: "dog bites man" vs "man bites dog" — toggle position on/off and watch the two orderings' vectors diverge, computed live.

**Tier B — the two highest-value standalone topics (build after Tier A):**

7. **Why models hallucinate, mechanically.** ONE idea: prediction picks the highest-probability continuation with no truth-check, so a plausible-but-false token can outrank the true one. Widget: a toy next-token distribution where the confident-wrong token beats the correct one, with a visible "no 'is this true?' signal anywhere" callout. Builds straight on page 3 above; it's the #1 question a real user asks.

8. **Tokenization edge cases: why it can't count the r's in "strawberry."** ONE idea: the model sees opaque token IDs, not letters, so per-letter questions are invisible to it. Widget: type a word, watch it split into token chunks with IDs, and a letter-counter the token view literally cannot recover. Concrete, memorable, follows directly from the token concept.

That's a clean **8-page batch**. Tier A (6 pages) is the priority — it converts the track from "meaning & attention" into a complete "ID → word" story and reuses every primitive already shipped, so it's cheap to build and high-payoff. If forced to ship fewer, pages 1, 3, and 4 (residual → unembedding → sampling) are the irreducible core that closes the loop.

---

## Verdict

**Yes — the "Meaning & Attention" track itself is now satisfactory; I'd stop passing it back for redesign.** Every widget genuinely computes (I verified the flagship's `q·k → softmax → blend` and the multi-head numbers by hand against `course-data.js`), attention is *derived* instead of asserted, dot-product/softmax/logits/activation are all defined the first time they're used, and the crammed chapters are now clean one-idea pages. The old hard-coded `aWeights` table — my single biggest iter-1 complaint — is gone and replaced with live arithmetic. What's left is *scope* (the blended vector's fate, the full forward pass) and two small blemishes (the neuron "past the edges" overstatement, and the shared-`Wv`-across-heads simplification), none of which is a widget that misleads — they're the next batch, not a redo of this one.
