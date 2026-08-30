# Build Spec — Iteration 2
## "Part 1 · Meaning & Attention" — closing the forward pass (8 new lessons; track becomes 15)

**For:** coding agents building directly from this spec.
**Design system (do not restyle):** `tools/course/course.css`, `tools/course/course.js`, clone `tools/course/_template.html` for every new page. All new files live in `tools/course/` beside the existing pages so `prev`/`next` relative links resolve.
**The bar to hit (unchanged from iter-1):** every widget must **COMPUTE its result live from the shared data on each interaction — recompute, never replay.** No answer may be typed as a constant. Reuse the shared globals in `course-data.js` (`TOKENS, E, DIM_LABELS, Wq/Wk/Wv, Wq2/Wk2, Wq3/Wk3, matvec, dot, mag, cosine, softmax, attend`) plus the new globals appended in Section B. The reference bar remains `attention-scoring.html` (the flagship: a "compute + hood + myth + check" bar with staged, traceable arithmetic).

**Color canon (reuse consistently):** amber = tokens/structure · cyan = vectors/embeddings/memory · violet = attention/generation · rose = weights/parameters · green = training/correct/probabilities.

**Owner's rule:** ONE core idea + ONE widget per page. Do not merge pages. Do not add pages. Filenames, order, and eyebrow numbers are LOCKED (below).

**Locked sequence (eyebrow reads `Part 1 · Lesson N/15`):**
```
8.  residual-stream.html        prev: attention-heads.html
9.  feed-forward.html
10. unembedding.html
11. sampling.html
12. transformer-block.html
13. positional-encoding.html
14. hallucination.html
15. tokenization-edges.html      next: index.html
```

---

# A. CONCEPT CANON — additions (extend iter-1's canon; define each the first time it's needed, plain language, reuse verbatim in spirit thereafter)

These continue the iter-1 canon. No term may be used before it is introduced on the page listed.

- **residual stream / "add back, don't replace"** *(pg 8)* — attention (and later the FFN) does not overwrite a token's vector; its output is **added onto** the vector the token already had: `new = old + Δ`. So the token's vector is a **running total** that every layer nudges, not a value that gets thrown out and rewritten. The original meaning is always still in there; each step only *refines* it. Say the words "add back, don't replace" and "running total."

- **feed-forward network (FFN) / "attention mixes across tokens, the FFN transforms each token alone"** *(pg 9)* — after attention has blended information *between* tokens, each token's vector is passed, **independently**, through a small neural network (the same multiply-add-squish **neuron** from lesson 4, stacked into a layer): `hidden = tanh(W1·v + b1)`, then `out = W2·hidden + b2`. Contrast to state on-page: **attention is the step where tokens look at each other; the FFN is the step where each token thinks about itself, on its own, no other token involved.** It is where lesson 4's neuron finally eats a real token vector.

- **unembedding / output projection ("the embedding table, read backwards")** *(pg 10)* — to turn the final vector into a next-word guess, **dot it against every row of the embedding table** `E`: one dot product per vocabulary token gives one score per token. Reuse lesson 2's dot product verbatim — same operation, now scoring the output vector against every word instead of scoring two words against each other. (Real models often reuse the very same `E` transposed — "tied weights"; state that as the common case.)

- **attention-logits vs vocab-logits — same softmax, different lists** *(introduce the distinction on pg 10; this is the review's flagged conflation)* — "logits" just means *raw scores before softmax*. There are two different lists of them in a transformer and they are **not the same object**:
  - **attention-logits** — one score **per token in the context** (~7 here, a few thousand in a real model): `q·k` for each token. Softmax over them → attention weights (how much to listen to each token). *This is what lessons 6–7 computed.*
  - **vocab-logits** — one score **per word in the whole vocabulary** (7 here, ~50,000 in a real model): the final vector dotted against every row of `E`. Softmax over them → next-token probabilities. *This is what lesson 10 computes.*
  Same softmax formula, applied to two completely different lists. Never call one "the logits" without saying which.

- **temperature / top-k / top-p — three dials on the same distribution** *(pg 11)* — all three reshape the vocab-probability distribution without changing the logits:
  - **temperature `T`** — divide every logit by `T` before softmax. `T<1` sharpens (the leader pulls further ahead, more deterministic); `T>1` flattens (long shots get a real chance, more random); `T=1` leaves it untouched.
  - **top-k** — keep only the `k` highest-probability tokens, zero the rest, renormalize. A hard cap on *how many* candidates survive.
  - **top-p (nucleus)** — keep the smallest set of top tokens whose probabilities add up to at least `p`, zero the rest, renormalize. A cap on *how much probability mass* survives — the number of candidates floats with how confident the model is.
  top-k / top-p decide *which tokens are eligible*; temperature decides *how peaky the choice among them is.*

- **transformer block** *(pg 12)* — one repeatable unit = **attention → add back (residual) → FFN → add back (residual)**, with a **layer-norm** step keeping the numbers in range between sub-steps. A model is this same block **stacked N times** (real models: dozens to ~100). Each block has its own learned weights; the vector that flows through gets refined block by block.

- **layer-norm (one honest line)** *(pg 12)* — a rescaling step that **resets the vector's numbers to a sane, standard range** (roughly: recenter to mean 0, rescale to a standard size) so that after many add-backs the values don't blow up or vanish between steps. That's the whole role for our purposes: keep the running total numerically tidy.

- **positional encoding — meaning + position, literally added, in its own directions** *(pg 13; redeems lesson 1's deferred promise)* — a token's position in the sentence is its own small vector `POS[pos]`, **arithmetically summed** onto the embedding: `vector = meaning + position`, element by element. **Why it doesn't clobber meaning:** the position vectors live in *different slots/directions* than the ones meaning uses, so adding them leaves the meaning dimensions untouched — a word keeps its meaning coordinates exactly and only picks up a position tag in the spare directions. The model learns to read meaning from the meaning directions and position from the position directions.

- **hallucination — plausibility ≠ truth (no truth signal)** *(pg 14)* — prediction always picks a **high-probability** continuation; nothing in the softmax computes "is this true?". A token that is common, fluent, and strongly associated can outrank the token that is actually correct. There is **no fact-check anywhere in the arithmetic** — only learned plausibility — so a confident, plausible, false token can win.

- **token opacity — IDs carry no spelling** *(pg 15)* — the model reads **opaque token IDs**, not letters. Once text is tokenized, the spelling is gone: a token is just a row number. So questions about letters *inside* a token (how many r's in "strawberry") are literally invisible to the model — it never sees the characters, only the chunk's ID.

**Named gaps this batch must close (verify each appears):** the blended vector's fate = residual (pg 8); the neuron finally applied to a real token (pg 9); "where does cat 61% come from" = unembedding dot products (pg 10); the temperature widget gets a home *inside* the track (pg 11); vector drift over layers (pg 12); position "literally added" shown, and "doesn't it clobber meaning?" answered (pg 13); why models hallucinate (pg 14); why it can't count letters (pg 15).

---

# B. NEW SHARED DATA — append to `course-data.js` (JS-ready, hand-verified)

Append the block below to `course-data.js` after the existing `NEURON` line. It reuses `E`, `attend`, `matvec`, `dot`, `softmax`, `NEURON` and adds only what the new pages need. Matrix-vector convention is unchanged everywhere: **`out[i] = Σⱼ W[i][j]·in[j]`** (row `i` dotted with the input). Do not transpose.

```js
/* ============================================================
   ITER-2 additions — closing the forward pass (pages 8–15).
   All numbers hand-verified; see "Hand-verification" in the spec.
   ============================================================ */

/* ---- Page 8 (residual): it's post-attention output, added back onto e_it ----
   attend(TOKENS.indexOf('it'), Wq, Wk).out ≈ [0.62, 0.24, 0.25, 0.10]  (computed, not stored)
   residual: new = e_it + attn_out                                              */
const RESID_TOKEN = 'it';                         // the token we follow through the pipeline
// e_it = E['it'] = [0.8, 0.1, 0.2, 0.3]  (from the shared table)
// RESID_VEC below is what the widget must REPRODUCE by computing e_it + attend(...).out:
const RESID_VEC = [1.42, 0.34, 0.45, 0.40];       // = [0.8+0.62, 0.1+0.24, 0.2+0.25, 0.3+0.10]

/* ---- Page 9 (FFN): a 4 -> 8 -> 4 net, tanh hidden, linear out ----
   Input is it's residual-updated vector RESID_VEC = [1.42, 0.34, 0.45, 0.40].
   Each hidden unit and each output unit is exactly the lesson-4 neuron.        */
const FFN = {
  in: [1.42, 0.34, 0.45, 0.40],                   // = RESID_VEC (post-attention, post-residual)
  // 8 hidden rows (4 weights each) + 8 biases. Sparse & readable on purpose.
  W1: [
    [ 1.0,  0.0,  0.0,  0.0],   // h0  "animate?"        reads animacy
    [ 0.0,  1.0,  0.0,  0.0],   // h1  "action?"         reads action
    [ 0.0,  0.0,  1.0,  0.0],   // h2  "state?"          reads state
    [ 0.0,  0.0,  0.0,  1.0],   // h3  "function/glue?"  reads function
    [ 1.0, -0.5,  0.0,  0.0],   // h4  "entity, not verb"
    [ 0.5,  0.5,  0.0,  0.0],   // h5  "subject-doing"
    [ 0.5,  0.0,  0.5,  0.0],   // h6  "animate + state"
    [ 0.0,  0.0,  1.0,  0.0]    // h7  "strong-state only"
  ],
  b1: [-0.5, -0.3, -0.3, -0.3, -0.4, -0.5, -0.5, -0.8],
  // 4 output rows (8 weights each) + 4 biases. Linear (no activation on output).
  W2: [
    [0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],     // out0 animacy  <- h0,h4
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],     // out1 action   <- h1,h5
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],     // out2 state    <- h2,h7
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.4, 0.0]      // out3 function <- h3,h6
  ],
  b2: [0.0, 0.0, 0.0, 0.0],
  act: Math.tanh
  // hidden  h  ≈ [0.726, 0.040, 0.149, 0.100, 0.691, 0.363, 0.409, -0.336]
  // output Δffn ≈ [0.71, 0.20, -0.09, 0.21]   (this is ADDED back too, see page 12)
};
// helper the FFN page uses (reuses matvec):
function ffnForward(v){
  const pre = matvec(FFN.W1, v).map((s,i)=> s + FFN.b1[i]);
  const hid = pre.map(FFN.act);
  const out = matvec(FFN.W2, hid).map((s,i)=> s + FFN.b2[i]);
  return { pre, hid, out };
}

/* ---- Page 10 (unembedding): a final vector for the predicting position, dotted
   against every row of E -> 7 vocab-logits -> softmax -> ranked next token.
   Context: "The cat sat because it was ___". Predicting position = last token
   ('was'); after attention (which pulled in 'tired' + 'cat') + FFN its running
   vector is state-heavy. Predicted next token = 'tired' (defensible, illustrative). */
const UNEMBED_VEC = [0.9, 0.3, 1.4, 0.1];         // final vector at the predicting position
// vocab-logit(t) = dot(UNEMBED_VEC, E[t]) for every t in TOKENS  (reuses lesson-2 dot)
// verified logits: The 0.10, cat 1.10, sat 0.62, because 0.40, it 1.06, was 0.82, tired 1.70
// verified softmax: tired 30.3%, cat 16.6%, it 16.0%, was 12.6%, sat 10.3%, because 8.2%, The 6.1%

/* ---- Page 11 (sampling): a standalone candidate set for "...it was ___".
   ~8 next-token candidates with logits; clear winner + real runners-up so
   temperature / top-k / top-p visibly change the eligible set.               */
const NEXT_LOGITS = [
  { tok: 'tired',   z: 3.2 },
  { tok: 'sleepy',  z: 2.4 },
  { tok: 'hungry',  z: 1.6 },
  { tok: 'resting', z: 1.1 },
  { tok: 'scared',  z: 0.7 },
  { tok: 'purring', z: 0.3 },
  { tok: 'gone',    z: -0.2 },
  { tok: 'blue',    z: -1.0 }
];
// verified (T=1): tired 51.0, sleepy 22.9, hungry 10.3, resting 6.3, scared 4.2,
//                 purring 2.8, gone 1.7, blue 0.8   (percent)
// verified top-k=3  -> keeps {tired, sleepy, hungry}
// verified top-p=0.9 -> keeps {tired, sleepy, hungry, resting}   (cum 0.510,0.740,0.843,0.906)
// verified temp: T=0.5 -> tired ~79% ; T=1 -> 51% ; T=2 -> ~30%

/* ---- Page 12 (transformer block): watch it's vector DRIFT over layers.
   Toy layer op = nudge the running vector a fixed fraction toward a
   cat-flavored attractor (illustrative stand-in for attention+FFN+residual).  */
const DRIFT = {
  v0: [0.8, 0.1, 0.2, 0.3],                        // e_it, layer 0
  attractor: [1.4, 0.3, 0.25, 0.1],                // contextual "cat-flavored" target
  rate: 0.4,                                       // v_L = v_{L-1} + rate*(attractor - v_{L-1})
  layers: 6
  // verified trajectory (2dp):
  // L0 [0.80,0.10,0.20,0.30]  L1 [1.04,0.18,0.22,0.22]  L2 [1.18,0.23,0.23,0.17]
  // L3 [1.27,0.26,0.24,0.14]  L4 [1.32,0.27,0.24,0.13]  L5 [1.35,0.28,0.25,0.12]
  // L6 [1.37,0.29,0.25,0.11]
};
function driftTo(L){                               // compute the running vector at layer L
  let v = DRIFT.v0.slice();
  for(let i=0;i<L;i++) v = v.map((x,d)=> x + DRIFT.rate*(DRIFT.attractor[d]-x));
  return v;
}

/* ---- Page 13 (positional encoding): its OWN tiny demo set.
   Meaning lives in dims 0,1 (animacy, action); position lives in dims 2,3
   (spare directions) so adding position never touches the meaning dims.       */
const POS_WORDS = {
  'dog':   [1.0, 0.2, 0.0, 0.0],
  'bites': [0.1, 1.0, 0.0, 0.0],
  'man':   [0.9, 0.1, 0.0, 0.0]
};
const POS = [                                      // position vectors, in dims 2,3 only
  [0.0, 0.0,  0.3, 0.1],   // position 0
  [0.0, 0.0,  0.0, 0.4],   // position 1
  [0.0, 0.0, -0.3, 0.7]    // position 2
];
const POS_ORDERINGS = {
  'dog bites man': ['dog','bites','man'],
  'man bites dog': ['man','bites','dog']
};
// slot(word,pos) = POS_WORDS[word] + POS[pos]   (element-wise)
// verified: meaning dims (0,1) are IDENTICAL to the bare word at every position;
// only dims (2,3) change -> position is added without clobbering meaning.

/* ---- Page 14 (hallucination): a toy distribution where a confident-but-false
   token outranks the true one. "The capital of Australia is ___".
   Canberra is the true capital; Sydney is the famous, higher-frequency wrong guess. */
const HALLUCINATE = {
  prompt: 'The capital of Australia is',
  logits: [
    { tok: 'Sydney',    z: 3.0, truth: false },
    { tok: 'Melbourne', z: 2.2, truth: false },
    { tok: 'Canberra',  z: 1.8, truth: true  },   // the correct answer, rank 3
    { tok: 'Perth',     z: 1.0, truth: false },
    { tok: 'Brisbane',  z: 0.8, truth: false }
  ]
  // verified softmax: Sydney 50.1, Melbourne 22.5, Canberra 15.1, Perth 6.8, Brisbane 5.5 (%)
  // -> the confident, plausible, FALSE token (Sydney) beats the true one (Canberra).
};

/* ---- Page 15 (tokenization edges): illustrative, NOT computed.
   Hard-coded chunkings + plausible-but-fake IDs. Real BPE differs — say so.    */
const TOKENIZE = {
  'strawberry':   { chunks: ['str','aw','berry'],        ids: [1618, 707, 15717] },
  'cat':          { chunks: ['cat'],                     ids: [9246] },
  'unbelievable': { chunks: ['un','bel','iev','able'],   ids: [359, 6667, 12796, 429] },
  'GPT':          { chunks: ['G','PT'],                  ids: [38, 6316] }
};
const COUNT_LETTER = 'r';   // "how many r's in strawberry?" -> a human sees 3; the token IDs carry none
```

### Hand-verification (DONE — numbers below must reproduce; if a build differs, the build is wrong)

**Page 8 · residual** — `attend('it')` weights (head 1, from iter-1, re-derived): The 5.5% · cat 35.8% · sat 9.2% · because 5.8% · it 25.2% · was 7.4% · tired 11.1%. Values `v=Wv·e` (dim3 ×0.5). Blend `out = Σ wⱼ·vⱼ`:
- out[0] = .358·1 + .252·.8 + .092·.2 + .111·.3 + … = **0.618**
- out[1] = .358·.2 + .092·1 + .252·.1 + .074·.5 + … = **0.243**
- out[2] = .111·1 + .358·.1 + .252·.2 + .074·.4 + … = **0.248**
- out[3] = .252·.15 + .058·.45 + .055·.5 + .074·.1 = **0.099**
→ `attn_out ≈ [0.62, 0.24, 0.25, 0.10]`. Residual `e_it + attn_out = [0.8+0.62, 0.1+0.24, 0.2+0.25, 0.3+0.10] =` **`[1.42, 0.34, 0.45, 0.40]`** ✓ (Replace-mode would instead give `[0.62,0.24,0.25,0.10]` — it *loses* it's original animacy 0.8 and function 0.3.)

**Page 9 · FFN** on `in = [1.42, 0.34, 0.45, 0.40]`:
- pre-activations `Σx·w + b`: h0 `1.42−0.5=0.92`; h1 `0.34−0.3=0.04`; h2 `0.45−0.3=0.15`; h3 `0.40−0.3=0.10`; h4 `1.42−0.17−0.4=0.85`; h5 `0.71+0.17−0.5=0.38`; h6 `0.71+0.225−0.5=0.435`; h7 `0.45−0.8=−0.35`.
- tanh → `h ≈ [0.726, 0.040, 0.149, 0.100, 0.691, 0.363, 0.409, −0.336]`.
- outputs (linear): out0 `0.6·0.726+0.4·0.691=` **0.712**; out1 `0.5·0.040+0.5·0.363=` **0.201**; out2 `0.5·0.149+0.5·(−0.336)=` **−0.094**; out3 `0.5·0.100+0.4·0.409=` **0.214**.
→ `Δffn ≈ [0.71, 0.20, −0.09, 0.21]` ✓ (FFN pushed animacy up further, trimmed the spurious state it picked up from `tired`.)

**Page 10 · unembedding** — `UNEMBED_VEC = [0.9,0.3,1.4,0.1]` dotted against each `E` row:
- tired `.9·.3+.3·.1+1.4·1+.1·0 =` **1.70** · cat `.9·1+.3·.2+1.4·.1 =` **1.10** · it `.9·.8+.3·.1+1.4·.2+.1·.3 =` **1.06** · was `.9·.1+.3·.5+1.4·.4+.1·.2 =` **0.82** · sat `.9·.2+.3·1+1.4·.1 =` **0.62** · because `.3·.1+1.4·.2+.1·.9 =` **0.40** · The `.1·1 =` **0.10**.
- softmax (sum of `e^(z−1.70)` = 3.305): tired **30.3%** · cat **16.6%** · it **16.0%** · was **12.6%** · sat **10.3%** · because **8.2%** · The **6.1%** (Σ = 100%). Predicted next token = **`tired`** ✓ — sensible for "…it was ___", runners-up `cat`/`it` (the subject). Illustrative, not literal.

**Page 11 · sampling** — `NEXT_LOGITS`, T=1 (sum `e^(z−3.2)` = 1.959): tired **51.0** · sleepy **22.9** · hungry **10.3** · resting **6.3** · scared **4.2** · purring **2.8** · gone **1.7** · blue **0.8** (%). top-k=3 → **{tired, sleepy, hungry}** (renormalized 60.6/27.2/12.2). top-p=0.9 → cumulative 0.510 → 0.740 → 0.843 → 0.906, so **{tired, sleepy, hungry, resting}**. Temperature: T=0.5 → tired **~79%**; T=2 → tired **~30%** ✓.

**Page 12 · drift** — `v_L = v_{L−1} + 0.4·(attractor − v_{L−1})`, attractor `[1.4,0.3,0.25,0.1]`:
L0 `[0.80,0.10,0.20,0.30]` → L1 `[1.04,0.18,0.22,0.22]` → L2 `[1.18,0.23,0.23,0.17]` → L3 `[1.27,0.26,0.24,0.14]` → L4 `[1.32,0.27,0.24,0.13]` → L5 `[1.35,0.28,0.25,0.12]` → L6 `[1.37,0.29,0.25,0.11]`. Animacy climbs 0.80→1.37 (toward `cat`'s 1.0), function drops 0.30→0.11; fast early, fine-tuning late ✓.

**Page 13 · positional** — `slot = word + POS[pos]`:
- "dog bites man": slot0 `dog+POS0=[1.0,0.2,0.3,0.1]`, slot1 `bites+POS1=[0.1,1.0,0.0,0.4]`, slot2 `man+POS2=[0.9,0.1,−0.3,0.7]`.
- "man bites dog": slot0 `man+POS0=[0.9,0.1,0.3,0.1]`, slot1 `bites+POS1=[0.1,1.0,0.0,0.4]`, slot2 `dog+POS2=[1.0,0.2,−0.3,0.7]`.
- **Meaning preserved:** `dog`'s dims (0,1) are exactly `1.0, 0.2` at every position; only dims (2,3) carry the position tag → position added, meaning untouched ✓.
- **Orderings diverge:** slot0 is `dog`-flavored vs `man`-flavored across the two orderings; slot2 flips too. **Without** position, the bare word vectors are position-independent (the two orderings are the same multiset of vectors) — a position-blind model cannot tell subject from object ✓.

**Page 14 · hallucination** — `HALLUCINATE.logits`, softmax (sum `e^(z−3.0)` = 1.997): Sydney **50.1%** · Melbourne **22.5%** · Canberra **15.1%** · Perth **6.8%** · Brisbane **5.5%**. The false-but-famous `Sydney` outranks the true `Canberra` (rank 3). Facts: Canberra IS Australia's capital; Sydney is its largest city (the common wrong guess) ✓.

**Page 15 · tokenization** — illustrative (not computed): "strawberry" → `["str","aw","berry"]` → IDs `[1618,707,15717]`. Human letter-count of `r` in s-t-**r**-a-w-b-e-**r**-**r**-y = **3**; the three token IDs contain no letters, so the model cannot recover it ✓.

---

# C. PER-PAGE SPEC (8 pages)

> Eyebrow reads `Part 1 · Lesson N/15`. Every widget carries a `.stage-note` with a `TRY →` line naming the exact interaction. Every page: prose (2–4 beats) → widget (recompute, not replay) → `.hood` with exact numbers → myth/reality + takeaway → self-check → pager. Load `<script src="course-data.js"></script>` then `course.js` then the page script, exactly as the flagship does.

---

## 8 · residual-stream.html — Where the blended vector goes: add back, don't replace

- **title (h1):** Where does the <span class="violet">blended vector</span> go?
- **eyebrow:** `Part 1 · Lesson 8/15`
- **objective:** Show that attention's output is **added back** onto the token's own vector (refine, don't replace), so the vector is a running total across layers.
- **prereq:** lesson 6 (attention produces a blended output vector for a token).
- **prose beats:**
  1. Last flagship page ended holding a 4-number blend for `it` — and left it dangling. Here's its fate: the model does **not** throw away `it`'s original vector and swap in the blend.
  2. It **adds the blend back on**: `new = old + Δattn`, element by element. The token keeps everything it already was and gains a refinement. Name it: the **residual stream** — a running total the whole network keeps nudging.
  3. Why add instead of replace? Replacing would erase the token's own identity every layer; adding lets each layer *contribute* without destroying what came before.
- **THE WIDGET — computational spec (add vs replace, live):**
  - **Inputs:** the residual token is `it` (fixed for this page). A toggle **Add back ⟷ Replace**. (Optional: reuse the lesson-6 token picker so the reader can follow a different token; `it` is the default and the one the hood documents.)
  - **Computed live on load / toggle:** `e = E['it']`; `A = attend(TOKENS.indexOf('it'), Wq, Wk).out` (the real lesson-6 pipeline — **recompute, do not read `RESID_VEC` as a constant**; `RESID_VEC` is only the verification target). In **Add** mode show `new[d] = e[d] + A[d]` for each of the 4 dims; in **Replace** mode show `new[d] = A[d]` and grey out `e`.
  - **What updates on screen:** three vector rows — `old (e)`, `Δ from attention (A)`, `new` — with the per-dim sum written out (`0.8 + 0.62 = 1.42`, …). A one-line read that flips with the toggle: Add → "`it` still remembers it was animacy-0.8 `it`, now pushed toward `cat` (animacy 1.42)"; Replace → "`it`'s original 0.8 animacy and 0.3 function are **gone** — the token forgot itself."
  - **Formula on screen:** `residual:  new = old + Δattn   (element-wise)`.
- **under-the-hood (.hood):** `e_it = [0.8, 0.1, 0.2, 0.3]`; `Δattn = attend(it).out = [0.62, 0.24, 0.25, 0.10]` (the blend from lesson 6); `new = [1.42, 0.34, 0.45, 0.40]`. "Every layer does this add; the vector is a **running total**, and the original embedding is never overwritten — that's why very deep models stay trainable. This same `[1.42, 0.34, 0.45, 0.40]` is the input to the FFN on the next page."
- **myth / reality:** *Myth* — "Attention overwrites the token's vector with the blend." / *Reality* — "It **adds** the blend onto the vector the token already had. The token keeps its identity and gains context; the running total is the residual stream."
- **self-check:** *"After attention, `it`'s vector becomes…"* — (a) replaced entirely by the blend of what it attended to *(wrong)*; **(b) its old vector plus the blend, added element-wise** *(correct, `data-correct="1"`)*; (c) the average of all seven tokens *(wrong)*. **Why:** the residual stream adds the attention output back on — refine, don't replace — so the original meaning is always still in there.
- **prev / next:** prev: `attention-heads.html` ("Many heads, many kinds of attention") · next: `feed-forward.html` ("The FFN: each token thinks alone")

---

## 9 · feed-forward.html — The FFN: attention mixes tokens, the FFN thinks per-token

- **title (h1):** The <span class="rose">feed-forward</span> step: each token thinks alone
- **eyebrow:** `Part 1 · Lesson 9/15`
- **objective:** Show that after attention, each token's vector runs **independently** through a small neural network — lesson 4's neuron, finally eating a real token vector.
- **prereq:** lesson 4 (one neuron = multiply, add, squish), lesson 8 (the residual vector).
- **prose beats:**
  1. Attention was the step where tokens **look at each other**. The FFN is the opposite: each token's vector goes through a little network **on its own**, no other token involved.
  2. That little network is just lesson 4's neuron, stacked: a hidden layer of neurons (`tanh(W1·v + b1)`) then an output layer (`W2·h + b2`). Same multiply-add-squish, now eating a real token vector.
  3. This is where the model *transforms* what attention gathered — sharpening features, dropping noise — one token at a time. Its output is **added back** too (residual, lesson 8).
- **THE WIDGET — computational spec (the FFN, computed):**
  - **Inputs:** the input vector is `FFN.in = [1.42, 0.34, 0.45, 0.40]` (the residual vector from lesson 8), shown fixed. Optional: an "edit input" affordance (4 tiny sliders) so the reader sees the whole net recompute — but the documented default is `FFN.in`.
  - **Computed live via `ffnForward(v)`:** 8 hidden neurons, each rendered as a lesson-4-style neuron tile: `hⱼ = tanh(Σᵢ FFN.in[i]·W1[j][i] + b1[j])`, showing the summed pre-activation and its tanh. Then 4 output neurons: `outₖ = Σⱼ hⱼ·W2[k][j] + b2[k]` (linear, no squish — flag that the output layer has no activation). **Recompute, do not replay** — reuse `matvec` and `Math.tanh`; change the input and every hidden and output number must move.
  - **What updates on screen:** a column of 8 hidden-unit tiles (each with its worked `Σ·w+b → tanh`), then the 4-number output vector `Δffn`, then one line: "the FFN pushed **animacy** up to make `it` more clearly `cat`-flavored and trimmed the stray **state** it picked up from `tired`." Highlight hidden unit **h0** ("animate?") as the one that fires hardest (0.73).
  - **Formula on screen:** `hidden = tanh(W1·v + b1)`, then `FFN out = W2·hidden + b2`, then `residual: v + FFN out`.
- **under-the-hood (.hood):** pre-activations `[0.92, 0.04, 0.15, 0.10, 0.85, 0.38, 0.435, −0.35]` → tanh `[0.73, 0.04, 0.15, 0.10, 0.69, 0.36, 0.41, −0.34]` → outputs `out0 = 0.6·0.73 + 0.4·0.69 = 0.71`, `out1 = 0.20`, `out2 = 0.5·0.15 + 0.5·(−0.34) = −0.09`, `out3 = 0.21`. `Δffn ≈ [0.71, 0.20, −0.09, 0.21]`, and (residual) the token's running vector becomes `[1.42+0.71, 0.34+0.20, 0.45−0.09, 0.40+0.21] = [2.13, 0.54, 0.36, 0.61]`. "Every token in the sentence runs through this **same** FFN, independently — attention shared information between them; the FFN never lets them talk."
- **myth / reality:** *Myth* — "After attention there's more attention; it's attention all the way down." / *Reality* — "Half of every block is the FFN — each token pushed through the same little neuron network **alone**. Attention mixes across tokens; the FFN transforms each token by itself."
- **self-check:** *"How is the FFN different from attention?"* — (a) it's the same operation with more heads *(wrong)*; **(b) attention mixes information across tokens; the FFN transforms each token's vector on its own, no other token involved** *(correct)*; (c) it looks the answer up in a table *(wrong)*. **Why:** attention is the cross-token step (Q·K over all tokens); the FFN runs each token's vector through the lesson-4 neuron independently.
- **prev / next:** prev: `residual-stream.html` · next: `unembedding.html` ("Vector → next-token logits")

---

## 10 · unembedding.html — Vector → next-token logits: dot against every row of E

- **title (h1):** From vector to a <span class="cyan">next-word</span> guess
- **eyebrow:** `Part 1 · Lesson 10/15`
- **objective:** Show that the final vector is **dot-producted against every row of the embedding table** to score all vocab tokens — those scores are the next-token (vocab-)logits. Reuses lesson 2's dot product. Answers "where does 'cat 61%' come from."
- **prereq:** lesson 2 (dot product = similarity score), lesson 8–9 (the final vector).
- **prose beats:**
  1. We now have a final vector for the predicting position. To turn it into a word, ask: which vocabulary token does it point at most? Score it against **every** token by the **same dot product** from lesson 2.
  2. Reuse the embedding table `E` **backwards**: dot the final vector against each row → one **vocab-logit** per word. Softmax them → next-token probabilities. (This is often the very same `E`, transposed — "tied weights.")
  3. **Two different logit lists** (name the distinction here): attention-logits are one-per-token (~7, lessons 6–7); these vocab-logits are one-per-vocabulary-word (7 here, ~50k real). Same softmax, different lists.
- **THE WIDGET — computational spec (rank the vocab, live):**
  - **Inputs:** the final vector `UNEMBED_VEC = [0.9, 0.3, 1.4, 0.1]` shown at top (framed: "the running vector at the last position, `was`, after attention + FFN"). Optional: 4 sliders to nudge it and watch the ranking re-sort.
  - **Computed live:** `logit(t) = dot(UNEMBED_VEC, E[t])` for every `t` in `TOKENS` (reuse `dot`), then `softmax(logits)`. **Recompute, not replay.** Show each dot product worked out for at least the winner (`tired: 0.9·0.3 + 0.3·0.1 + 1.4·1 + 0.1·0 = 1.70`).
  - **What updates on screen:** a ranked bar list of all 7 vocab tokens, longest bar = highest probability, each row `token — logit 1.70 — 30.3%`. The winner (`tired`) highlighted; a caption "the toy predicts **`tired`** for '…it was ___' — sensible; runners-up `cat`/`it` are the sentence's subject." A one-line honesty note: "real vocab is ~50,000 rows, so real logits and this ranking differ — the *mechanism* (one dot product per word) is the real thing."
  - **Formula on screen:** `vocab-logit(word) = final_vector · E[word]` → `softmax` → next-token probabilities.
- **under-the-hood (.hood):** full table (token · dot written out · logit · softmax %): tired 1.70/30.3% · cat 1.10/16.6% · it 1.06/16.0% · was 0.82/12.6% · sat 0.62/10.3% · because 0.40/8.2% · The 0.10/6.1% (Σ = 100%, sum of `e^(z−1.70)` = 3.305). "So 'cat 61%' in any explainer is exactly this: the final vector dotted against every embedding row, softmaxed. **Attention-logits vs vocab-logits:** lesson 6's 7 scores were one-per-token in the context; these 7 are one-per-word in the vocabulary — same softmax, different lists."
- **myth / reality:** *Myth* — "Predicting the next word is a special lookup or a separate classifier head with its own giant table." / *Reality* — "It's the **same dot product** as similarity, run against every embedding row — the model scores its final vector against the whole vocabulary and softmaxes."
- **self-check:** *"Where does 'next token: `cat`, 61%' come from?"* — (a) a stored table of sentence → next-word *(wrong)*; **(b) the final vector dotted against every row of the embedding table, then softmax** *(correct)*; (c) the highest attention weight from lesson 6 *(wrong)*. **Why:** the unembedding scores the output vector against every vocab token with a dot product; softmax turns those vocab-logits into probabilities.
- **prev / next:** prev: `feed-forward.html` · next: `sampling.html` ("Reshape the distribution: temperature, top-k, top-p")

---

## 11 · sampling.html — Logits → probabilities you can reshape: temperature + top-p + top-k

- **title (h1):** Three dials on the <span class="green">next-word</span> distribution
- **eyebrow:** `Part 1 · Lesson 11/15`
- **objective:** Show the vocab-logits becoming a probability distribution you can reshape with **temperature**, **top-k**, and **top-p**. Gives the oft-referenced "temperature widget" a home inside the track.
- **prereq:** lesson 10 (vocab-logits → softmax), canon softmax.
- **prose beats:**
  1. Softmax gives one honest distribution. Before picking a word, you can **reshape** it — this is where "creativity vs. focus" is dialed.
  2. **Temperature** divides every logit by `T`: below 1 sharpens toward the leader, above 1 flattens toward the field. **top-k** keeps only the `k` best. **top-p** keeps the smallest set whose probability adds up to `p`.
  3. top-k/top-p decide *which tokens are eligible*; temperature decides *how peaky* the pick among them is. Same softmax underneath.
- **THE WIDGET — computational spec (reshape the bars, live):**
  - **Inputs:** three controls — a **temperature** slider (`0.1…2.0`, step 0.1, default 1.0), a **top-k** control (`1…8`, default 8 = off), a **top-p** slider (`0.1…1.0`, step 0.05, default 1.0 = off). Data = `NEXT_LOGITS` (8 candidates for "…it was ___").
  - **Computed live, in this order:** (1) apply temperature: `z' = z / T`; (2) `p = softmax(z')`; (3) if top-k < 8, keep the k highest-`p`, zero the rest; (4) if top-p < 1, sort desc, keep the running-sum-≥-p nucleus, zero the rest; (5) **renormalize** the survivors to sum to 1. **Recompute, not replay** — reuse `softmax`; every dial move re-runs steps 1–5.
  - **What updates on screen:** 8 horizontal probability bars (rose/green), each `token — p%`. Eliminated tokens (cut by k or p) go dim/greyed with `0%`; surviving bars renormalize and regrow. A live caption naming the eligible set: e.g. "top-k = 3 → **tired, sleepy, hungry** eligible" / "top-p = 0.9 → **tired, sleepy, hungry, resting** eligible". A running "Σ eligible = 100%" line.
  - **Formula on screen:** `p = softmax(logits / T)`, then keep top-k or the top-p nucleus, then renormalize.
- **under-the-hood (.hood):** at `T = 1`: tired 51.0 · sleepy 22.9 · hungry 10.3 · resting 6.3 · scared 4.2 · purring 2.8 · gone 1.7 · blue 0.8 (%). **top-k = 3** keeps {tired, sleepy, hungry}, renormalized to 60.6 / 27.2 / 12.2. **top-p = 0.9** accumulates 0.510 → 0.740 → 0.843 → 0.906, so it keeps {tired, sleepy, hungry, **resting**} — one more token than k=3, because `resting` was needed to cross 0.9. **Temperature:** `T = 0.5` sharpens `tired` to ~79%; `T = 2` flattens it to ~30%. "All three change the *distribution*, never the *logits* — the model's scores are fixed; you're only reshaping how you sample from them."
- **myth / reality:** *Myth* — "Temperature makes the model smarter or dumber / changes what it knows." / *Reality* — "The logits are fixed. Temperature, top-k, and top-p only **reshape the distribution** you sample from — more peaky and safe, or flatter and more surprising."
- **self-check:** *"You set top-p = 0.9. What does that do?"* — (a) keeps exactly 9 tokens *(wrong)*; **(b) keeps the smallest set of top tokens whose probabilities add up to at least 0.9, then renormalizes** *(correct)*; (c) divides every logit by 0.9 *(wrong)*. **Why:** top-p (nucleus) caps *probability mass*, so the number of eligible tokens floats with the model's confidence — unlike top-k, which caps the *count*.
- **prev / next:** prev: `unembedding.html` · next: `transformer-block.html` ("One block, stacked N times")

---

## 12 · transformer-block.html — Attention + FFN + residual (+ layer-norm) = one block, stacked

- **title (h1):** One <span class="violet">block</span>, stacked N times
- **eyebrow:** `Part 1 · Lesson 12/15`
- **objective:** Assemble the pieces — attention → add back → FFN → add back, with a one-line layer-norm — into a repeatable **block**, and show a token's vector **drifting** over layers.
- **prereq:** lessons 6–9 (attention, residual, FFN).
- **prose beats:**
  1. You've now met every piece. Stack them in order: **attention → add back → FFN → add back**. That whole unit is one **transformer block**.
  2. One honest extra line: between sub-steps a **layer-norm** rescales the vector so the numbers stay in a sane range after all the adding. That's its whole job here.
  3. A model is this same block **repeated N times** (dozens to ~100), each with its own weights. Watch the running vector drift, layer by layer, from raw `it` toward its contextual, `cat`-flavored meaning.
- **THE WIDGET — computational spec (layer slider, watch the drift):**
  - **Inputs:** a **layer slider** `0…6` (default 0). Data = `DRIFT`.
  - **Computed live via `driftTo(L)`:** the running vector at layer `L`, `v_L = v_{L−1} + rate·(attractor − v_{L−1})` (a compact stand-in for one block's attention+FFN+residual — **flag it as illustrative**). **Recompute, not replay** — the slider recomputes `driftTo(L)` each move.
  - **What updates on screen:** (1) the 4-number vector at layer `L` with its dim bars (animacy bar visibly growing); (2) a small line-plot of each dimension across layers 0→6 with a marker at `L`; (3) a schematic of the block — `attention → +resid → FFN → +resid → (layer-norm)` — with a caption "each layer nudges the vector; early layers move it fast, later layers fine-tune." At `L = 6`: "`it` now reads animacy 1.37 — firmly `cat`-flavored; it started at 0.80."
  - **Formula on screen:** `block: v → v + attn(v) → +ffn(...) → layer-norm`, and `(toy) v_L = v_{L−1} + 0.4·(target − v_{L−1})`.
- **under-the-hood (.hood):** trajectory table L0 `[0.80,0.10,0.20,0.30]` → L1 `[1.04,0.18,0.22,0.22]` → L2 `[1.18,0.23,0.23,0.17]` → L3 `[1.27,0.26,0.24,0.14]` → L4 `[1.32,0.27,0.24,0.13]` → L5 `[1.35,0.28,0.25,0.12]` → L6 `[1.37,0.29,0.25,0.11]`. "Animacy 0.80→1.37, function 0.30→0.11 — the vector sheds its raw-`it`-ness and gains context, fast early then fine-tuning. **This is a toy:** real blocks each have their own learned attention + FFN weights and a real layer-norm; we use one fixed attractor so the drift is visible and traceable. Layer-norm (skipped in the toy arithmetic) would rescale each step's numbers back to a standard range so the running total never blows up."
- **myth / reality:** *Myth* — "Each layer is a different kind of operation; deeper means fancier math." / *Reality* — "It's the **same block** — attention, FFN, two add-backs, a layer-norm — repeated with different weights. Depth refines the same running vector over and over."
- **self-check:** *"What is one transformer block?"* — (a) one attention head *(wrong)*; **(b) attention → add back → FFN → add back (with a layer-norm), repeated N times with different weights** *(correct)*; (c) the embedding lookup plus softmax *(wrong)*. **Why:** a block bundles the cross-token step (attention) and the per-token step (FFN), each added back onto the residual stream; stacking blocks is what makes a model deep.
- **prev / next:** prev: `sampling.html` · next: `positional-encoding.html` ("Meaning + position, added")

---

## 13 · positional-encoding.html — Meaning + position, literal element-wise addition

- **title (h1):** <span class="cyan">Meaning</span> + <span class="amber">position</span>, literally added
- **eyebrow:** `Part 1 · Lesson 13/15`
- **objective:** Redeem lesson 1's deferred promise — show position as a literal element-wise addition, and answer "doesn't adding position clobber the meaning?" "dog bites man" vs "man bites dog" diverge live.
- **prereq:** lesson 1 (the embedding lookup, which promised "meaning + position").
- **prose beats:**
  1. Attention has no built-in sense of order — on its own it would treat "dog bites man" and "man bites dog" identically. Position has to be **added in**.
  2. Each position gets its own small vector `POS[pos]`, summed onto the embedding: `vector = meaning + position`, element by element. Not a metaphor — real addition.
  3. **"Doesn't that clobber the meaning?"** No — the position vectors live in **their own directions** (spare slots the meaning barely uses). So a word keeps its meaning coordinates exactly and only picks up a position tag in the leftover dimensions. *(This prose beat is required.)*
- **THE WIDGET — computational spec (toggle position, watch orderings diverge):**
  - **Inputs:** an ordering toggle **"dog bites man" ⟷ "man bites dog"** and a **position ON/OFF** toggle. Data = `POS_WORDS`, `POS`, `POS_ORDERINGS`.
  - **Computed live:** for the chosen ordering, each slot `slot(word, pos) = POS_WORDS[word] + POS[pos]` when position is ON, or just `POS_WORDS[word]` when OFF (element-wise add — **recompute, not replay**).
  - **What updates on screen:** three position slots, each showing its 4-number vector, dims split visually into **meaning (0,1: animacy, action)** and **position (2,3)**. Two callouts: (a) **meaning preserved** — `dog`'s meaning dims stay `1.0, 0.2` at every position, only the position dims change (highlight that dims 0–1 never move when you flip position ON/OFF); (b) **orderings diverge** — with position ON, slot 0 holds `dog`-flavored vs `man`-flavored across the two orderings, so the sequences differ; with position OFF, flipping the ordering just shuffles identical vectors and nothing downstream can tell subject from object.
  - **Formula on screen:** `token_at_position = meaning_vector + POS[position]   (element-wise)`.
- **under-the-hood (.hood):** "dog bites man" with position: slot0 `[1.0,0.2, 0.3,0.1]`, slot1 `[0.1,1.0, 0.0,0.4]`, slot2 `[0.9,0.1, −0.3,0.7]`. "man bites dog": slot0 `[0.9,0.1, 0.3,0.1]`, slot2 `[1.0,0.2, −0.3,0.7]`. "`dog`'s meaning dims are `1.0, 0.2` in **both** orderings and **both** positions — adding `POS` never touched them; the change is entirely in dims 2–3, the position channel. That's why position doesn't clobber meaning: it's written into directions meaning leaves free. Without position, the two orderings are the same multiset of vectors — order is invisible."
- **myth / reality:** *Myth* — "Adding a position vector scrambles or overwrites the word's meaning." / *Reality* — "Position is added into its **own directions**; the meaning coordinates are untouched. The model reads meaning from the meaning dims and order from the position dims."
- **self-check:** *"How does a transformer know 'dog bites man' ≠ 'man bites dog'?"* — (a) attention naturally processes tokens left to right *(wrong)*; **(b) a position vector is added onto each token's embedding, in its own directions, tagging where it sits** *(correct)*; (c) the tokenizer stores the word order *(wrong)*. **Why:** attention is order-blind; position is injected by literal element-wise addition, in dimensions meaning doesn't use, so order becomes readable without clobbering meaning.
- **prev / next:** prev: `transformer-block.html` · next: `hallucination.html` ("Why models hallucinate, mechanically")

---

## 14 · hallucination.html — Prediction picks plausible, not true

- **title (h1):** Why models <span class="rose">hallucinate</span>
- **eyebrow:** `Part 1 · Lesson 14/15`
- **objective:** Show that prediction picks the highest-probability continuation with **no truth-check**, so a plausible-but-false token can outrank the true one.
- **prereq:** lessons 10–11 (vocab-logits → probabilities).
- **prose beats:**
  1. Everything so far ranks tokens by **learned plausibility** — how well a continuation fits the patterns in training. Nowhere in the arithmetic is there a step that asks "is this actually **true**?"
  2. So when a false answer is more *famous* or more *frequent* than the true one, it can get the higher probability — and win.
  3. That's hallucination, mechanically: not the model "lying," just plausibility beating truth, because truth was never a signal in the computation.
- **THE WIDGET — computational spec (confident-wrong beats correct, live):**
  - **Inputs:** the prompt `HALLUCINATE.prompt` ("The capital of Australia is ___") shown fixed. A reveal button "mark the true answer." Data = `HALLUCINATE`.
  - **Computed live:** `softmax(HALLUCINATE.logits.map(o=>o.z))` — **recompute, not replay** — producing the 5 probabilities. The `truth:true` entry (`Canberra`) is flagged only for display; it gets **no boost** in the math.
  - **What updates on screen:** 5 probability bars ranked by probability. The top bar (`Sydney`, 50%) marked with a red "confident — and wrong" tag; the true answer (`Canberra`, rank 3, 15%) marked green "actually correct." A callout with no arrow into the softmax: "**No 'is this true?' step exists anywhere in this computation.**" The winner is chosen purely by probability.
  - **Formula on screen:** `pick = argmax softmax(logits)   —   note: no truth term anywhere`.
- **under-the-hood (.hood):** softmax (sum `e^(z−3.0)` = 1.997): Sydney 50.1% · Melbourne 22.5% · **Canberra 15.1% (true)** · Perth 6.8% · Brisbane 5.5%. "Canberra **is** Australia's capital; Sydney is its largest, most-talked-about city, so training made `Sydney` the stronger association here. The softmax ranks by that association — it has no fact table to consult. Turn up temperature and you don't fix it; you just change *how often* the wrong-but-confident token wins. (Illustrative logits — the mechanism, plausibility-with-no-truth-signal, is the real point.)"
- **myth / reality:** *Myth* — "A hallucination is a bug — the model 'knows' the truth and glitched." / *Reality* — "There's no truth signal in the computation at all. The model ranks by plausibility; when a false token is more plausible than the true one, it wins. Same mechanism as a correct answer."
- **self-check:** *"Mechanically, why can a model state a confident falsehood?"* — (a) a corrupted weight flipped the true answer *(wrong)*; **(b) prediction ranks tokens by learned plausibility with no truth-check, so a plausible false token can outrank the true one** *(correct)*; (c) the temperature was set too high *(wrong)*. **Why:** nothing in the softmax computes truth — only plausibility — so fame/frequency can beat correctness.
- **prev / next:** prev: `positional-encoding.html` · next: `tokenization-edges.html` ("Why it can't count the r's in strawberry")

---

## 15 · tokenization-edges.html — Opaque IDs: why it can't count the r's in "strawberry"

- **title (h1):** Why it can't count the <span class="amber">r</span>'s in "strawberry"
- **eyebrow:** `Part 1 · Lesson 15/15`
- **objective:** Show that the model sees **opaque token IDs**, not letters — so per-letter questions are invisible to it.
- **prereq:** lesson 1 (a token is a chunk with an integer ID).
- **prose beats:**
  1. Back on lesson 1, text became token IDs and the letters were dropped. This page shows the cost of that.
  2. "strawberry" isn't one letter-by-letter thing to the model — it's a few **chunks**, each an opaque ID. The spelling is gone the moment it's tokenized.
  3. So "how many r's in strawberry?" asks about letters *inside* chunks the model can't see into. It's not being dumb — the information literally isn't in its input.
- **THE WIDGET — computational spec (split into chunks + IDs, count what's lost):**
  - **Inputs:** a small set of word buttons (`strawberry`, `cat`, `unbelievable`, `GPT`) — default `strawberry`. Optional free-text field that maps to the hard-coded set (unknown words show "not in this toy tokenizer"). Data = `TOKENIZE`, `COUNT_LETTER`. **This is illustrative data, not computed** — say so on-page.
  - **What updates on screen:** two side-by-side panels. **What the human sees:** the raw letters of the word with each `r` highlighted and a live count ("3 r's"). **What the model sees:** the word split into chunks `["str","aw","berry"]`, each shown as an opaque ID pill `1618 · 707 · 15717`, with the letters faded/removed — and a struck-through "letter count: unavailable — no letters here." A caption: "the model gets the bottom row. The `r`'s live *inside* chunks it reads as single IDs."
  - **Note:** the split and IDs are hard-coded illustrative values (real BPE tokenizers chunk differently and assign different IDs) — state this in the widget and hood.
  - **Formula on screen (conceptual):** `"strawberry" → tokenizer → [1618, 707, 15717] → (letters gone)`.
- **under-the-hood (.hood):** "strawberry" → `["str","aw","berry"]` → IDs `[1618, 707, 15717]` (illustrative, not a real vocabulary). A human counts `r` in s-t-**r**-a-w-b-e-**r**-**r**-y = **3**. The model's input is three integers; nothing in `[1618, 707, 15717]` encodes 'how many r's' — the embedding for each chunk was learned from *usage*, not spelling. "This is why letter puzzles, rhyme edge-cases, and character counts trip models up: it's a tokenization blind spot, not a reasoning failure. Real tokenizers (BPE) chunk by frequency, so the exact split differs — but the opacity is real."
- **myth / reality:** *Myth* — "The model can obviously see the letters in a word it just read." / *Reality* — "It sees opaque token IDs. Once 'strawberry' is three chunk-IDs, the individual letters are gone — so it can't reliably count them."
- **self-check:** *"Why do models miscount the letters in a word?"* — (a) they're bad at arithmetic *(wrong)*; **(b) they read opaque token IDs, not letters — the spelling is gone after tokenization** *(correct)*; (c) the word was misspelled in training *(wrong)*. **Why:** tokenization replaces letters with chunk IDs; per-letter questions ask about information the model's input no longer contains.
- **prev / next:** prev: `hallucination.html` · next: `index.html` (label "Back to the map — you've closed the forward pass, ID → word")

---

# D. FIXES TO EXISTING PAGES (precise, minimal edits for the coding pass)

### D0 · Global renumber + hub growth (7 → 15)

- **All 7 existing lesson pages** — eyebrow `Part 1 · Lesson N/7` → `Part 1 · Lesson N/15` (embeddings-lookup 1/7→1/15, embedding-space 2/7→2/15, analogy 3/7→3/15, neuron 4/7→4/15, attention-qkv 5/7→5/15, attention-scoring 6/7→6/15, attention-heads 7/7→7/15). *(Grep target: `Lesson N/7`.)*
- **attention-heads.html** — pager `next`: change `href="index.html"` + `<div class="ttl">Back to the map</div>` → `href="residual-stream.html"` + `<div class="ttl">Where the blended vector goes</div>`. (It is no longer the last lesson.)
- **index.html (hub)** — grow the lesson list from 7 to 15 cards and update the count pill:
  - Pill: `📚 <b>7 lessons</b>` → `📚 <b>15 lessons</b>`.
  - Meta description: `7-lesson track` → `15-lesson track` (and reword to mention the full forward pass).
  - Add 8 new `.lesson-card` list items (`08`–`15`) after the attention-heads card, matching the existing markup. Color dots: 08 violet (residual/attention-adjacent), 09 rose (FFN/weights), 10 cyan (unembedding/vectors), 11 green (sampling/probabilities), 12 violet (block), 13 cyan (position/vectors) with an amber accent, 14 rose (hallucination), 15 amber (tokens). Blurbs (one line each):
    - 08 residual-stream — "Add attention's blend *back onto* the token's own vector — refine, don't replace."
    - 09 feed-forward — "Run the lesson-4 neuron across a real post-attention token vector."
    - 10 unembedding — "Dot the final vector against every embedding row to rank the next word."
    - 11 sampling — "Reshape the next-word distribution with temperature, top-k, and top-p."
    - 12 transformer-block — "Stack attention + FFN + residual and watch a token's vector drift over layers."
    - 13 positional-encoding — "Add position to meaning, element-wise; 'dog bites man' vs 'man bites dog'."
    - 14 hallucination — "See a confident, false token outrank the true one — no truth-check anywhere."
    - 15 tokenization-edges — "Watch 'strawberry' become opaque IDs the letter-count can't survive."
  - The intro `<p class="lede">` says "Seven short pages" → "Fifteen short pages"; and consider extending the framing sentence to "…where a token's meaning-vector comes from, how attention computes, **and how the vector becomes the next word**." *(Grep target in body: "Seven short pages", "7 lessons".)*
  - Footer count text if any references "7".

### D1 · neuron.html — "sum past the edges" overstatement (review New-confusion #4)

- **`.stage-note`** (line ~165): old — *"Drag <b>x₁</b> all the way up and watch the sum climb past the edges — yet <code>tanh</code> flattens the output near <b>+1</b>."* → new — *"Drag <b>x₁</b> all the way up and watch the sum grow (to about <b>+3</b>) — yet <code>tanh</code> still flattens the output near <b>+1</b>. Flip to <b>no activation</b> and the same neuron becomes a plain straight line whose **output** shoots off the chart, unbounded."* (The sum stays within the ±3.2 domain; it is the un-squished *output* that shoots off — not the sum.)
- **`.hood`** (line ~176): old — *"Drag <code>x₁</code> to a large value and the sum keeps growing, but <code>tanh</code> flattens near <code>±1</code>…"* → new — *"Drag <code>x₁</code> to its max and the sum reaches about <code>+3</code>; <code>tanh</code> flattens the output near <code>+1</code> regardless. With <b>no activation</b> the output <code>= Σxw + b</code> would keep climbing with no ceiling — it's the **output** that blows up without the squish, not the sum."*

### D2 · analogy.html — flag the separate demo vocabulary (review New-confusion #2, question 3)

- Add one line at the end of the setup prose (after line ~105): *"<span class="strong-ink">Heads-up:</span> this page uses a **separate little demo vocabulary** (`man / woman / king / queen …`) with its own axes `[gender, royalty, age]` — not the `cat / sat / it …` sentence tokens from the other pages. It needs clean gender/royalty axes for the analogy to land exactly; `king` is **not** a row in the sentence table."*

### D3 · embeddings-lookup.html — clarify axis labels are made-up-for-readability (review question 2)

- The setup-prose line (line 126) already says "No human labels these axes, but we pretend-label ours." Tighten so it can't be half-read as claiming real names: append — *"— these four names (`animacy · action · state · function`) are **our invention for readability**, not labels the model learned; the real axes have no names."* (Keep the existing sentence; add the clause.)

### D4 · attention-scoring.html — distinguish attention-logits from vocab-logits (review New-confusion #5, question 17)

- After the "logits" sentence in setup prose (line 129, ends "…the logits of the attention softmax."), add: *"(These are **attention-logits** — one score per token in the sentence. Later, next-token prediction produces **vocab-logits** — one score per word in the whole vocabulary. Same softmax, different lists; lesson 10 builds the second.)"*

### D5 · course-data.js / attention-heads.html — shared-Wv-across-heads (review New-confusion #7, question 18)

- **Decision: keep the single shared `Wv`; add an honest note.** (Rationale: the heads page only draws routing arcs — it never blends Values per head — so adding `Wv2`/`Wv3` would be dead, unused code. A one-line note is the minimal correct fix.)
- **course-data.js** — the `Wv` comment (line 27): `// value = embedding with dim3 halved (shared by all heads)` → `// value = embedding with dim3 halved. NOTE: real multi-head attention has a separate Wv per head; this toy shares one because the heads page only draws routing (Q·K) arcs, never a per-head Value blend.`
- **attention-heads.html** — add one line to the `.hood`: *"(Simplification: real heads each have their own `Wv`; ours share one, because this page only shows Q·K routing, not the per-head Value blend.)"*

### D6 · temperature-widget cross-links now resolve inside the track (review New-confusion #6, question 19)

- Only **attention-scoring.html** actually references the temperature widget (confirmed by grep; page 2 does **not** — no edit needed there). Two occurrences, both should point to the new `sampling.html`:
  - Line 130: *"…the classic temperature widget."* → *"…the same softmax that turns next-token logits into probabilities — the one you'll dial with temperature on <a href="sampling.html">lesson 11</a>."*
  - Line 160 (`.softmax-note`): *"…the identical <code>e^z / Σe^z</code> fractions the temperature widget uses…"* → *"…the identical <code>e^z / Σe^z</code> fractions the sampling page (lesson 11) reshapes with temperature…"*
- **course-data.js** — the comment "same softmax shape as the classic temperature widget" (line 37) may stay, or point to sampling.html; cosmetic only.

---

# Builder checklist (no further decisions needed)

- 8 new files cloned from `_template.html`, in `tools/course/`, wired `course-data.js` → `course.js` → page script.
- Section B appended verbatim to `course-data.js`; every widget computes from those globals (recompute, not replay). Verification targets in Section B must reproduce to the stated decimals.
- Page 9 reuses the lesson-4 neuron on `it`'s real residual vector (closes the islanded-neuron blemish). Page 10 reuses lesson-2's `dot`. Page 13 answers "doesn't position clobber meaning?" in prose.
- Pager chain intact: `attention-heads → residual-stream → feed-forward → unembedding → sampling → transformer-block → positional-encoding → hallucination → tokenization-edges → index`.
- All 15 eyebrows read `/15`; hub shows 15 cards and a "15 lessons" pill.
- Section D edits applied to the 7 existing pages + hub + `course-data.js`.
