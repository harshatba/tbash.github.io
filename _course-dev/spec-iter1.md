# Build Spec — Iteration 1
## "Part 1 · Meaning & Attention" (7 lessons + hub)

**For:** coding agents building directly from this spec.
**Design system (do not restyle):** `tools/course/course.css`, `tools/course/course.js`, clone `tools/course/_template.html` for every page. All 8 files live in the same directory as the template so `prev`/`next` relative links resolve.
**The bar to hit:** every widget must **COMPUTE its result live from the data on each interaction — recompute, never replay.** The reference bar is the temperature widget in `tools/llm-visual-course.html` (real `Math.exp((z-m)/t)` softmax, lines ~1236-1269) and the quantization widget in `tools/llm-inference-efficiency.html` (real rounding). The thing we are explicitly killing is the old attention widget's hard-coded `aWeights` table (`llm-visual-course.html` ~line 1110, `7:{1:0.72,...}`) — no answer may be typed as a constant; it must fall out of arithmetic the learner can trace.

**Color canon (from course.css, reuse consistently):** amber = tokens/structure · cyan = vectors/embeddings/memory · violet = attention/generation · rose = weights/parameters · green = training/correct.

**Owner's rule:** ONE core idea + ONE widget per page. Do not merge pages. Do not add pages.

---

# A. CONCEPT CANON (shared vocabulary — reuse verbatim in spirit on every page)

Define each term the FIRST time the track needs it, then reuse the same phrasing. No term may be used before it is introduced.

- **token** — a chunk of text (a word or word-piece) the model reads as one unit. Each distinct token has a fixed integer ID (its row number). *(assumed known from the classic walkthrough; restated once on page 1.)*
- **embedding / embedding vector** — the list of numbers a token's ID is turned into. It is **looked up**, not computed: the model holds one big **lookup table** with one row per token, and the embedding is simply that token's row. Say the words "lookup table."
- **dimension** — one slot (one number) in the vector. Our toy vectors have 4 dimensions; real models use hundreds to thousands. Each dimension is a learned "axis of meaning" — no human labels them, but we pretend-label ours (animacy, action, state, function-word-ness) so the numbers are readable.
- **learned** — the numbers in the table (and in every weight matrix) started random and were slowly tuned during training so the model predicts well. On this track they are **frozen** — we only read them.
- **weight** — one tunable number the model learned. Physically: one floating-point number at one address in memory.
- **weight matrix** — a grid of weights. Multiplying a vector by a weight matrix produces a new vector: each output number is (one row of the matrix) · (the input vector). We use this exact convention everywhere: `out[i] = Σⱼ W[i][j] · in[j]` (row `i` dotted with the input). State this convention wherever a matrix is used so no builder transposes it.
- **dot product** — multiply two equal-length vectors slot-by-slot and add up the products: `a·b = Σ aᵢbᵢ`. It is a **similarity score**: bigger when two vectors point the same way. This single operation shows up as (a) similarity between embeddings and (b) the attention score between a Query and a Key — the SAME operation both times.
- **softmax** — turns a list of raw scores into positive fractions that add up to 1. Bigger score → bigger share; the gaps get exaggerated (a small lead becomes a bigger share). Formula shown once: `share(zᵢ) = e^{zᵢ} / Σⱼ e^{zⱼ}`. It is the **same operation** used for attention weights (this track) and for next-token probabilities (the temperature widget in the classic walkthrough) — call this out explicitly.
- **logits** — the raw scores fed into softmax, before it turns them into fractions. In the classic temperature widget these are the values called `z`/"scores"; name them "logits" here. Attention's raw dot-product scores are the logits of the attention softmax.
- **Query (Q), Key (K), Value (V)** — three DIFFERENT vectors made from the SAME token's embedding by three DIFFERENT weight matrices (Wq, Wk, Wv). Plain gloss, used consistently: the **Query** is what a token is looking for; the **Key** is what a token advertises it offers; the **Value** is the content a token hands over if it is attended to.
- **attention weight** — the softmax'd attention score: the fraction of its attention one token sends to another. All the weights a token hands out sum to 1.
- **attention (one line)** — each token builds a new version of itself by taking a weighted blend of every token's Value, where the weights come from Query·Key similarity run through softmax.
- **head** — one complete Q/K/V attention computation with its own Wq/Wk/Wv. A model runs many heads in parallel; different heads learn to route differently (one tracks reference, another tracks predicate/state). The heads' outputs are combined.
- **position is literally added** — a token's position is encoded as its own vector and **arithmetically summed** onto the embedding (`vector = meaning + position`), element by element. Not a metaphor. *(Mentioned in canon so pages can reference it; positional encoding is not a page in this iteration — see page 1 note.)*
- **activation function ("squish")** — after a neuron sums its weighted inputs it passes the sum through an activation function that bends the straight-line sum into a curve and keeps it in a tidy range (we use `tanh`). Name it "activation function"; "squish" is the friendly alias. Without it, stacking layers would collapse into a single straight-line step and the network could not model anything curved.

**Named gaps from the review this canon must close (verify each appears):** "lookup table" (pg 1), "activation function" named + motivated (pg 4), "logits" named (pg 6), position "literally added" (canon + pg 1 note), dot-product→softmax as the actual attention math (pgs 5-6).

---

# B. PER-PAGE SPEC

> Numbering: eyebrow reads `Part 1 · Lesson N/7`. Hub is the map (no lesson number). `~X min` and `math: none/some` pills per template. Every widget section carries a `.stage-note` with a `TRY →` line naming the exact interaction.

---

## 0 · index.html — Course hub / map

- **title (h1):** How LLMs Work — Meaning & Attention
- **eyebrow:** Interactive deep-dive · Part 1
- **objective:** Orient the learner and route them into the 7-lesson track; link the classic walkthrough as reference.
- **prereq:** none.
- **prose beats:**
  1. This track opens the two hardest boxes in an LLM — *where a token's meaning-vector comes from* and *how attention actually computes* — and makes you trace real numbers, not analogies.
  2. Seven short pages, one idea and one live widget each. Every number on screen is computed in front of you.
  3. Want the wide tour first? Link the classic walkthrough.
- **THE WIDGET — computational spec:** No compute widget; this is the map. Render the 7 lessons as a vertical list of cards (reuse `.pager`/card styling), each: lesson number, title, one-line "what you'll compute" blurb, color dot matching its concept color. Cards link to the 7 pages in order. Below, a separate "The classic walkthrough" block with two links: `/tools/llm-visual-course.html` ("How LLMs Work — the full 12-chapter tour") and `/tools/llm-inference-efficiency.html` ("How LLMs Run Fast — inference & serving"). Label these "the classic walkthrough (overview pages)".
- **under-the-hood:** none.
- **myth / reality:** *Myth* — "LLMs understand words the way we do." / *Reality* — "They turn every token into a list of numbers and do arithmetic on those lists; this track shows the arithmetic."
- **self-check:** *"What will you actually do on the Attention II page?"* — (a) Read an analogy about spotlights *(wrong)*; **(b) Watch dot products and softmax get computed from real vectors** *(correct, data-correct="1")*; (c) Memorize the Q/K/V definitions *(wrong)*. **Why:** every widget on this track computes its answer live from the data.
- **prev / next:** prev: none (`.disabled`) · next: `embeddings-lookup.html`

---

## 1 · embeddings-lookup.html — Token IDs → embeddings: the learned lookup table

- **title (h1):** Where does the <span class="cyan">vector</span> come from?
- **eyebrow:** Part 1 · Lesson 1/7
- **objective:** Show that an embedding is not computed — it is **looked up** as one row of a learned table, one row per token ID.
- **prereq:** knows tokens have integer IDs (classic walkthrough §02, restated here).
- **prose beats:**
  1. A token ID (e.g. `cat` = 2543) carries zero meaning — 2543 is just a row number.
  2. The model keeps one giant **lookup table**: one row per token, each row a list of numbers (the **embedding vector**). "Turning a token into a vector" is literally *go to that row and read it off.* No math, just a lookup.
  3. Those row-numbers were **learned** during training and are now frozen; each **dimension** is a learned axis of meaning (we pretend-label ours so they're readable).
  4. One-line bridge to positional encoding without making it a page: the model then **literally adds** a second "where it sits" vector onto this row — `vector = meaning + position` — element by element (flag as "covered in the classic walkthrough").
- **THE WIDGET — computational spec (lookup, made tangible):**
  - **Data:** the canonical 7 tokens and their 4-dim embeddings from Section C (this is the SAME table page 5-7's attention reads — say so).
  - Render the table as 7 rows: `ID | token | [d0, d1, d2, d3]`, columns headed `animacy · action · state · function` (the pretend labels), cyan-tinted numbers.
  - **Input:** learner clicks a token chip (also a number field to type an ID 0-6). **On click:** highlight that row, and animate a "read head" that copies the four numbers out of the row into a big display `embedding(it) = [0.8, 0.1, 0.2, 0.3]`.
  - **Must recompute, not replay:** the readout is pulled from the data array by index (`E[TOKENS.indexOf(tok)]`), never from a per-token hard-coded string. Changing the array must change the display. (Trivial compute, but the point is the value is *addressed*, not memorized.)
  - **What updates:** highlighted row, the 4 extracted numbers, and a caption "row #4 → this is `it`'s meaning, before anything else happens to it."
- **under-the-hood (.hood):** "A real table is `vocab_size × d_model` — e.g. 50,000 rows × 4,096 numbers ≈ 200 million weights, just for the lookup. Ours is 7 × 4. Row for `it`: `[0.8, 0.1, 0.2, 0.3]`."
- **myth / reality:** *Myth* — "The model computes a word's vector from its letters." / *Reality* — "It looks the vector up by ID in a table it learned. The letters were already gone at tokenization."
- **self-check:** *"Turning token ID 2543 into its embedding is mostly…"* — (a) a calculation over the digits 2-5-4-3 *(wrong)*; **(b) reading row 2543 of a learned table** *(correct)*; (c) asking attention what it means *(wrong)*. **Why:** an embedding is a lookup — the row was learned in training, and reading it is just indexing.
- **prev / next:** prev: `index.html` · next: `embedding-space.html`

---

## 2 · embedding-space.html — Embedding space & similarity

- **title (h1):** Nearby vectors mean <span class="cyan">similar things</span>
- **eyebrow:** Part 1 · Lesson 2/7
- **objective:** Show that closeness between embedding vectors = similarity of meaning, and that closeness is computed by the **dot product** (a similarity score).
- **prereq:** page 1 (an embedding is a row of numbers).
- **prose beats:**
  1. Once every token is a vector, "similar meaning" becomes "vectors that point the same way."
  2. We measure it with a **dot product** — multiply slot-by-slot, add up. Bigger = more alike. (Introduce the term here; it returns as the attention score on page 6.)
  3. Our 4 dimensions can't be drawn, so the scatter is a flattened 2-D *map* for intuition — but the similarity bars are computed from the true 4-dim vectors.
- **THE WIDGET — computational spec (live similarity ranking):**
  - **Data:** same 7 tokens + 4-dim embeddings (Section C). Plus a fixed 2-D map coordinate per token (Section C, `MAP2D`) used ONLY to place dots in the scatter — label the scatter "a flattened map (real distances live in 4-D)."
  - **Left:** SVG scatter of the 7 tokens at `MAP2D` positions, colored by cluster.
  - **Input:** click a token = the "anchor." **On click:** compute, live, the **cosine similarity** of the anchor's 4-dim vector to every other token: `cos(a,b) = (a·b)/(|a||b|)`. (Use cosine so magnitude differences don't dominate; show the raw dot product too.)
  - **What updates:** (1) a ranked bar list, longest bar = most similar, each row `token — cos = 0.93`; (2) in the scatter, draw faint lines from anchor to others with opacity ∝ similarity, and size/brighten the nearest. Anchor `it` must visibly rank **cat** highest.
  - **Must recompute, not replay:** all bars come from the cosine loop over `E`; no similarity is stored. Editing a vector in `E` must reorder the bars.
- **under-the-hood (.hood):** Show the actual numbers for anchor `it`: `it·cat = 0.8·1.0 + 0.1·0.2 + 0.2·0.1 + 0.3·0 = 0.84`, `|it| = √0.78 = 0.883`, `|cat| = √1.05 = 1.025`, `cos = 0.84 / (0.883·1.025) = 0.93`. Compare `cos(it,tired)=0.49`, `cos(it,sat)=0.31`. "`it` sits closest to `cat` — the same reason it will attend to `cat` two pages from now."
- **myth / reality:** *Myth* — "The 2-D dots are the real embedding." / *Reality* — "Real vectors have hundreds of dimensions; the dots are a flattened shadow. The dot-product similarity is computed on the full vector."
- **self-check:** *"Two tokens have vectors that point almost the same way. Their dot product is…"* — (a) near zero *(wrong)*; **(b) large — they're similar in meaning** *(correct)*; (c) negative *(wrong)*. **Why:** the dot product is a similarity score — aligned vectors give a big positive number.
- **prev / next:** prev: `embeddings-lookup.html` · next: `analogy.html`

---

## 3 · analogy.html — Directions in meaning-space: a LIVE analogy calculator

- **title (h1):** Meaning has <span class="cyan">directions</span> you can do arithmetic with
- **eyebrow:** Part 1 · Lesson 3/7
- **objective:** Let the learner compute `A − B + C` on real vectors and watch the nearest word fall out — proving `king − man + woman ≈ queen` instead of asserting it.
- **prereq:** pages 1-2 (vectors, similarity).
- **prose beats:**
  1. Because meanings are vectors, the *difference* between two is a direction — "king → man" is the same direction as "queen → woman": the "royal→commoner" arrow.
  2. So you can add and subtract meanings. `king − man` isolates "royalness"; `+ woman` lands on `queen`.
  3. Nothing here is looked up as an answer — the widget does the subtraction and then searches for the nearest real word.
- **THE WIDGET — computational spec (the analogy calculator):**
  - **Data:** the 8-word analogy vocab in 3 dims `[gender, royalty, age]` from Section C (`VOCAB`). This set is self-contained (not the sentence tokens) because it needs clean semantic axes.
  - **Input:** three dropdowns — A, B, C — each listing all 8 words (default A=king, B=man, C=woman). Optional preset buttons for a few known-good analogies (Section C lists them).
  - **On any change, compute live:** `r = VOCAB[A] − VOCAB[B] + VOCAB[C]` (element-wise, 3 numbers). Then compute cosine similarity of `r` to every vocab word **except A, B, C**, and pick the max = the answer. Show the top-3 with scores.
  - **What updates:** (1) the vector arithmetic spelled out — `[-1,1,0] − [-1,0,0] + [1,0,0] = [1,1,0]`; (2) the winner word big, e.g. `≈ queen`, with `cos = 1.00`; (3) runner-ups with their scores.
  - **Must recompute, not replay:** the winner comes from the arithmetic + nearest-neighbor search over `VOCAB`; there is NO `{king,man,woman}→queen` lookup. Any A/B/C combo returns a computed nearest word. **Note to builder:** every intended analogy in Section C lands exactly on a vocab vector (cos = 1.0), so the demo is crisp; off-target combos still return the honest nearest word.
- **under-the-hood (.hood):** `king − man + woman`: gender `-1−(-1)+1 = 1`, royalty `1−0+0 = 1`, age `0−0+0 = 0` → `[1,1,0]` = `queen` exactly. Then `king − queen + prince`: `[-1,1,0]−[1,1,0]+[-1,1,1] = [-1,1,1]` = `prince`? show it resolves to `princess` `[1,1,1]` — walk the three numbers.
- **myth / reality:** *Myth* — "`king − man + woman = queen` is a cute coincidence someone hard-coded." / *Reality* — "It's real vector subtraction landing near a real row; the widget searches for the nearest word every time."
- **self-check:** *"`king − man + woman` lands near `queen` because…"* — (a) the model memorized that fact *(wrong)*; **(b) subtracting `man` and adding `woman` moves along the male→female direction while keeping 'royal'** *(correct)*; (c) the words rhyme *(wrong)*. **Why:** differences between vectors are directions of meaning, and you can add them.
- **prev / next:** prev: `embedding-space.html` · next: `neuron.html`

---

## 4 · neuron.html — One neuron: squish(Σ xᵢwᵢ + b)

- **title (h1):** One <span class="rose">neuron</span>: multiply, add, squish
- **eyebrow:** Part 1 · Lesson 4/7
- **objective:** Show a single neuron performing an ACTUAL multiply-and-sum, then motivate the **activation function** ("squish"). Just the one node — not the whole forward pass.
- **prereq:** pages 1-3 (vectors, weights hinted). This is the "weights" color (rose) page.
- **prose beats:**
  1. A neuron takes a few input numbers, multiplies each by its own **weight**, adds them up, adds a **bias**, and passes the total through a **squish**.
  2. Every multiply is `input · weight`. That's the whole operation — billions of these, wired together, are the model.
  3. Why squish? Name it: the **activation function** (we use `tanh`). It bends the straight-line sum into a curve and keeps it in a tidy range. Without it, stacking neurons collapses into one straight-line step — the network couldn't model anything curved. *(This is the review's unanswered "why squish?")*
- **THE WIDGET — computational spec (show every multiplication):**
  - **Data:** one neuron, 3 inputs. Defaults `x = [0.90, 0.30, 0.60]`, weights `w = [1.3, −0.9, 0.7]`, bias `b = 0.1`, activation `tanh` (Section C, `NEURON`).
  - **Input:** three sliders for `x₁,x₂,x₃` (range −1…1, step 0.05). Weights/bias fixed and shown (rose).
  - **On any slider move, compute and DISPLAY each step:**
    - three product tiles: `x₁·w₁ = 0.90 × 1.3 = 1.17`, `x₂·w₂ = 0.30 × −0.9 = −0.27`, `x₃·w₃ = 0.60 × 0.7 = 0.42`
    - the sum: `1.17 + (−0.27) + 0.42 + b(0.1) = 1.42`
    - the squish: `tanh(1.42) = 0.89`, drawn as a dot on a small tanh S-curve so the learner sees where the sum lands on the curve.
  - **What updates:** all three products, the running sum, the output number, and the dot's position on the S-curve — all live.
  - **Must recompute, not replay:** output = `Math.tanh(x.reduce((a,xi,i)=>a+xi*w[i],0)+b)`. No lookup. The point the review demanded: **surface the individual `x·w` product**, not just a recolored node.
- **under-the-hood (.hood):** "Drag `x₁` to a large value and watch the sum grow, but `tanh` flattens near ±1 — that's the squish saving the number from blowing up. With no activation, `out = Σxw+b` is just a straight line; ten of them in a row is still one straight line. The curve is what lets layers build up nonlinear meaning."
- **myth / reality:** *Myth* — "A neuron 'decides' or 'thinks.'" / *Reality* — "A neuron multiplies its inputs by weights, adds them, and squishes the total. The intelligence is in the *pattern* of billions of these, not any one."
- **self-check:** *"What does the activation function ('squish') buy you?"* — (a) it makes the math faster *(wrong)*; **(b) it bends the straight-line sum into a curve so stacked layers can model nonlinear patterns** *(correct)*; (c) it stores the weights *(wrong)*. **Why:** without a nonlinear squish, any stack of neurons collapses to a single straight-line transform.
- **prev / next:** prev: `analogy.html` · next: `attention-qkv.html`

---

## 5 · attention-qkv.html — Attention I: Q, K, V are three vectors from one token

- **title (h1):** <span class="violet">Query</span>, <span class="violet">Key</span>, <span class="violet">Value</span>: three vectors from one token
- **eyebrow:** Part 1 · Lesson 5/7
- **objective:** Establish that Q, K, V are three DIFFERENT vectors made from the SAME embedding by three DIFFERENT weight matrices — the setup for the scoring math on page 6. No scoring yet.
- **prereq:** page 4 (weight-matrix multiply = row·vector), page 1 (embedding).
- **prose beats:**
  1. Attention starts by making three new vectors from each token's embedding. Same input row, three different weight matrices: `Wq`, `Wk`, `Wv`.
  2. Plain gloss (canon): **Query** = what this token is looking for; **Key** = what it advertises it offers; **Value** = the content it will hand over if attended to.
  3. Each is just a matrix multiply — `q = Wq·e` — using the row·vector rule from page 4. Nothing is decided yet; we've only prepared the pieces.
- **THE WIDGET — computational spec (make Q, K, V from an embedding, live):**
  - **Data:** the 7 tokens + embeddings, and `Wq`, `Wk`, `Wv` (head 1) from Section C.
  - **Input:** click a token. **On click:** show its embedding `e`, then compute `q = Wq·e`, `k = Wk·e`, `v = Wv·e` using `out[i] = Σⱼ W[i][j]·e[j]`.
  - **What updates:** three colored vectors (Q/K/V, each 4 numbers) appear, each with ONE expanded worked row so the learner sees the matrix multiply, e.g. for `it`, `q[0] = 2·0.8 + 0·0.1 + 0·0.2 + 1·0.3 = 1.9`. Full results for `it`: `q=[1.9,0.2,0.2,0.3]`, `k=[0.8,0.1,0.2,0.09]`, `v=[0.8,0.1,0.2,0.15]`.
  - **Must recompute, not replay:** Q/K/V computed from `E` and the `W` matrices on every click; changing a matrix changes the vectors. State the row·vector convention on-page.
  - Explicitly: this page STOPS before comparing Q to K. "Next page: what happens when `it`'s Query meets every token's Key."
- **under-the-hood (.hood):** Print `Wq`, `Wk`, `Wv` as 4×4 grids and one full multiply for `it`'s Query. Note Wv here is a mild rescale (dim 3 halved) so V ≠ embedding but stays readable.
- **myth / reality:** *Myth* — "Query, Key, and Value are three different tokens." / *Reality* — "They're three different vectors built from the *same* token's embedding by three different learned matrices."
- **self-check:** *"Where do a token's Q, K, and V come from?"* — (a) three different tokens nearby *(wrong)*; **(b) the token's own embedding, multiplied by three different weight matrices** *(correct)*; (c) the softmax output *(wrong)*. **Why:** Q=Wq·e, K=Wk·e, V=Wv·e — same input row, three learned matrices.
- **prev / next:** prev: `neuron.html` · next: `attention-scoring.html`

---

## 6 · attention-scoring.html — Attention II: dot-product → softmax → blend (FLAGSHIP)

- **title (h1):** How <span class="violet">attention</span> actually computes
- **eyebrow:** Part 1 · Lesson 6/7
- **objective:** THE flagship real-math widget. Click a query token; watch Q·K dot products computed against every token, watch softmax turn them into attention weights that sum to 1, watch the Values blend into a new vector. This is the page the whole review is asking for.
- **prereq:** page 5 (Q/K/V), page 2 (dot product = similarity), canon softmax.
- **prose beats:**
  1. To decide how much token A should listen to token B, take A's **Query** and B's **Key** and dot them: `q·k`. Bigger dot = better match = higher raw score (the score's **logits**).
  2. Run all those scores through **softmax** → positive fractions that sum to 1: the **attention weights**. (Same softmax as next-token probabilities — the classic temperature widget — pointed out explicitly.)
  3. Build A's new vector as the **weighted blend of every token's Value**, using those weights. A token literally becomes a mix of what it paid attention to.
- **THE WIDGET — computational spec (the full pipeline, computed):**
  - **Data:** 7 tokens + embeddings + `Wq`,`Wk`,`Wv` (head 1), Section C. Precompute each token's `k` and `v` once; compute the chosen token's `q` on click.
  - **Input:** click a token = the Query token (default `it`). Optional "step" button to reveal the 4 stages one at a time; a "run" button to animate straight through.
  - **Stage 1 — raw scores (logits):** for the chosen `qᵢ`, compute `qᵢ·kⱼ` for **all** j=0..6. Show a row per token: `it · cat → 1.9·1.0 + 0.2·0.2 + 0.2·0.1 + 0.3·0 = 1.96`. Display all 7 raw scores. **Note (builder):** use the RAW dot product — do NOT divide by √d here; the numbers in Section C assume no scaling. (Mention √d scaling only in the hood, see below.)
  - **Stage 2 — softmax → weights:** apply `e^{z}/Σe^{z}` to the 7 scores. Show each score becoming a percentage; render a bar per token; **display the running sum = 100%** so the "fractions that sum to 1" property is visible. Reuse the exact softmax shape from the temperature widget.
  - **Stage 3 — the weights as arcs:** draw violet arcs from the query token to each token, thickness ∝ attention weight (this is the OLD widget's visual — but now the thickness is the computed weight, not `aWeights`). `it→cat` must be the fattest arc.
  - **Stage 4 — blend V:** compute `out = Σⱼ weightⱼ · vⱼ` (element-wise). Show the weighted sum forming the 4-number result and a one-line read: "`it`'s new vector is now mostly `cat`-flavored (animacy high) with a hint of `tired`."
  - **What the learner SEES update on each click:** 7 raw dot-product rows → 7 softmax percentages summing to 100% → arc thicknesses → the blended output vector. All four recompute for whichever token is clicked.
  - **Must recompute, not replay — CRITICAL:** every number flows from `E` and the `W` matrices through `dot()`, `softmax()`, and the weighted-V loop. There is NO stored weight table. Deleting a token or editing a vector must change every downstream number. This page is the direct replacement for the hard-coded `aWeights` demo.
  - **Verified target (must match, see Section C arithmetic):** clicking `it` yields raw scores `cat 1.96, it 1.61, tired 0.79, sat 0.60, was 0.39, because 0.14, The 0.09` → attention weights `cat 36%, it 25%, tired 11%, sat 9%, was 7%, because 6%, The 5%` → blended `out ≈ [0.62, 0.24, 0.25, 0.10]`. If a build shows anything else for `it`, the math is wrong — fix before shipping.
  - **Scope guard:** attention here is **bidirectional** (every token may look at every token) to keep the mechanism pure. Do not add causal masking — that's a later-part topic; one sentence saying so is enough. Note "it also keeps ~25% on itself — self-attention is normal; the largest *outward* share still goes to cat."
- **under-the-hood (.hood):** Full `it` walkthrough: the seven `q·k` dot products written out, the `e^z` values `[1.09, 7.10, 1.82, 1.15, 4.99, 1.47, 2.20]`, sum `19.83`, and the divisions giving the percentages. One line: "Real transformers divide each score by √(key-dimension) before softmax to keep the numbers from getting too large — with d=4 that's ÷2; it changes the sharpness but not which token wins. We skip it here so the arithmetic stays clean."
- **myth / reality:** *Myth* — "The attention weights are looked up or hand-set." / *Reality* — "They're computed every step: dot products of Queries and Keys, run through softmax. Change one vector and every weight changes."
- **self-check:** *"An attention weight of 0.36 from `it` to `cat` means…"* — (a) `cat` is 36% likely to be the next word *(wrong)*; **(b) 36% of `it`'s attention — and 36% of `cat`'s Value — goes into `it`'s new vector** *(correct)*; (c) `cat` and `it` are 36% spelled the same *(wrong)*. **Why:** softmax'd Q·K scores are the fractions of each token's Value that get blended in; they sum to 1.
- **prev / next:** prev: `attention-qkv.html` · next: `attention-heads.html`

---

## 7 · attention-heads.html — Attention III: multiple heads

- **title (h1):** Many <span class="violet">heads</span>, many kinds of attention
- **eyebrow:** Part 1 · Lesson 7/7
- **objective:** Show that a model runs several attention computations in parallel, each with its own Wq/Wk/Wv, and they route differently — toggle heads and watch the arcs re-route.
- **prereq:** page 6 (the full scoring pipeline).
- **prose beats:**
  1. One attention computation is a **head**. A real layer runs many heads at once — same tokens, different Wq/Wk/Wv each.
  2. Because each head has its own matrices, each learns a different job: one tracks *reference* (`it`→`cat`), one tracks *predicate/state* (`it`→`tired`), one tracks *action* (`it`→`sat`).
  3. Their outputs are combined so each token ends up enriched from several angles at once. That's why "attention" is really "attention, in parallel, many ways."
- **THE WIDGET — computational spec (toggle heads, arcs re-route):**
  - **Data:** same 7 tokens + embeddings; three head matrix sets from Section C — Head 1 (`Wq/Wk/Wv`, the reference head from page 6), Head 2 (`Wq2/Wk2`, state head), Head 3 (`Wq3/Wk3`, action head). Shared `Wv` for the blend.
  - **Input:** toggle switches for Head 1 / Head 2 / Head 3 (default Head 1 + Head 2 on). Click a query token (default `it`).
  - **On each toggle/click, compute live per active head:** its `q,k` → dot products → softmax → weights, exactly the page-6 pipeline, independently per head. Draw each head's arcs in a distinct hue (e.g. Head1 violet, Head2 cyan, Head3 amber). Show a tiny per-head weight bar-list.
  - **What updates:** turning on Head 2 adds a second fan of arcs where `it`'s fattest arc now points at `tired`, not `cat`; Head 3 points it at `sat`. The learner SEES the same token routed three different ways.
  - **Must recompute, not replay:** each head runs the real pipeline over its own matrices; no per-head answer is stored.
  - **Verified targets (Section C):** for query `it` — Head 1 top = `cat` (36%), Head 2 top = `tired` (41%), Head 3 top = `sat` (~46% of its softmax). Distinct winners = the whole point.
- **under-the-hood (.hood):** Show Head 2's `it` computation landing on `tired`: `q2_it·k2_tired = 0.24·0.09 + 0.1·0.1 + 0.6·3.0 + 0.09·0 = 1.83` (the biggest score in that head), vs Head 1 where `cat` won. Same token, different matrices, different focus.
- **myth / reality:** *Myth* — "More heads = the model looks harder at the same thing." / *Reality* — "Each head has its own matrices and learns a different relationship; they run in parallel and get combined."
- **self-check:** *"Why do different heads point `it` at different words?"* — (a) randomness *(wrong)*; **(b) each head has its own Wq/Wk/Wv, so its Q·K scores rank tokens differently** *(correct)*; (c) they take turns *(wrong)*. **Why:** different weight matrices produce different Queries and Keys, so different tokens win the dot-product race.
- **prev / next:** prev: `attention-scoring.html` · next: `index.html` (back to the map; label "Back to the map / on to the classic walkthrough").

---

# C. SHARED TOY DATA (JS-ready — every attention page imports THIS, identical)

Put these in a shared inline block on each attention page (pages 1, 5, 6, 7 use the sentence set; pages 2 adds `MAP2D`; page 3 uses `VOCAB`; page 4 uses `NEURON`). Matrix-vector convention everywhere: **`out[i] = Σⱼ W[i][j] · in[j]`** (row `i` dotted with the input vector). Do **not** transpose.

```js
/* ---- Canonical sentence tokens (pages 1, 2, 5, 6, 7) ---- */
const TOKENS = ['The','cat','sat','because','it','was','tired'];

// 4-dim embeddings. Pretend-labeled dims: [animacy, action, state, function]
const E = {
  'The':     [0.0, 0.0, 0.0, 1.0],
  'cat':     [1.0, 0.2, 0.1, 0.0],
  'sat':     [0.2, 1.0, 0.1, 0.0],
  'because': [0.0, 0.1, 0.2, 0.9],
  'it':      [0.8, 0.1, 0.2, 0.3],
  'was':     [0.1, 0.5, 0.4, 0.2],
  'tired':   [0.3, 0.1, 1.0, 0.0]
};

/* ---- Head 1 : the "reference" head (it -> cat). Used on pages 5 & 6. ---- */
const Wq = [[2,0,0,1],[0,2,0,0],[0,0,1,0],[0,0,0,1]];      // query gain on animacy+function
const Wk = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0.3]];    // key advertises raw dims (function damped)
const Wv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0.5]];    // value = embedding with dim3 halved (shared by all heads)

/* ---- Head 2 : "state/predicate" head (it -> tired). Page 7. ---- */
const Wq2 = [[0.3,0,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,0.3]]; // gain on the state dim
const Wk2 = [[0.3,0,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,0.3]];

/* ---- Head 3 : "action" head (it -> sat). Page 7. ---- */
const Wq3 = [[0.3,0,0,0],[0,3,0,0],[0,0,0.3,0],[0,0,0,0.3]]; // gain on the action dim
const Wk3 = [[0.3,0,0,0],[0,3,0,0],[0,0,0.3,0],[0,0,0,0.3]];

/* ---- Core ops (reuse the temperature widget's softmax shape) ---- */
const matvec = (W,v)=> W.map(row => row.reduce((s,w,j)=> s + w*v[j], 0));
const dot    = (a,b)=> a.reduce((s,x,i)=> s + x*b[i], 0);
function softmax(zs){ const m=Math.max(...zs); const ex=zs.map(z=>Math.exp(z-m)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(e=>e/s); }
// attention for one query token index qi under a head {Wq,Wk}:
function attend(qi, Wqh, Wkh){
  const q = matvec(Wqh, E[TOKENS[qi]]);
  const scores = TOKENS.map(t => dot(q, matvec(Wkh, E[t])));   // raw logits (NO /sqrt(d))
  const weights = softmax(scores);
  const V = TOKENS.map(t => matvec(Wv, E[t]));
  const out = [0,1,2,3].map(d => TOKENS.reduce((s,_,j)=> s + weights[j]*V[j][d], 0));
  return {q, scores, weights, out};
}

/* ---- 2-D map coords for the scatter ONLY (page 2). Similarity is computed from E, not these. ---- */
const MAP2D = { 'The':[0.9,3.4],'because':[1.3,3.0],'cat':[3.2,1.0],'it':[3.5,1.4],'sat':[1.6,0.6],'was':[2.0,1.2],'tired':[3.0,2.4] };

/* ---- Analogy vocab (page 3 only). Dims: [gender(-male/+female), royalty, age(young=1)] ---- */
const VOCAB = {
  'man':[-1,0,0],'woman':[1,0,0],'king':[-1,1,0],'queen':[1,1,0],
  'prince':[-1,1,1],'princess':[1,1,1],'boy':[-1,0,1],'girl':[1,0,1]
};
// known-good preset analogies (all land EXACTLY on a vocab vector, cos = 1.00):
//  king - man   + woman  = queen      queen - king + prince = princess
//  king - queen + woman  = man        prince - boy + girl   = princess
//  woman - girl + boy    = man        king - man   + boy     = prince

/* ---- Single neuron (page 4 only) ---- */
const NEURON = { x:[0.90,0.30,0.60], w:[1.3,-0.9,0.7], b:0.1, act: Math.tanh };
```

### Hand-verification (DONE — numbers below must reproduce; if a build differs, the build is wrong)

**Page 6 / Head 1, query = `it`** (`e_it=[0.8,0.1,0.2,0.3]`):
- `q_it = Wq·e_it = [2·0.8+1·0.3, 2·0.1, 1·0.2, 1·0.3] = [1.9, 0.2, 0.2, 0.3]`
- Keys `k=Wk·e` (dim3 ×0.3): The`[0,0,0,0.3]` cat`[1,0.2,0.1,0]` sat`[0.2,1,0.1,0]` because`[0,0.1,0.2,0.27]` it`[0.8,0.1,0.2,0.09]` was`[0.1,0.5,0.4,0.06]` tired`[0.3,0.1,1,0]`
- Raw scores `q_it·k`: **The 0.09 · cat 1.96 · sat 0.60 · because 0.14 · it 1.61 · was 0.39 · tired 0.79**
- `e^z`: `[1.09, 7.10, 1.82, 1.15, 4.99, 1.47, 2.20]`, sum `19.83`
- Attention weights: **The 5.5% · cat 35.8% · sat 9.2% · because 5.8% · it 25.2% · was 7.4% · tired 11.1%** (sum 100%) → **`it`→`cat` is the top arc.** ✓
- Blended `out = Σ wⱼ·vⱼ ≈ [0.62, 0.24, 0.25, 0.10]` (animacy-dominant, from cat; state bump from tired). ✓

**Page 7 / Head 2 (state), query = `it`:** `q2_it=[0.24,0.1,0.6,0.09]`; top raw score `tired = 0.24·0.09+0.1·0.1+0.6·3.0+0 = 1.83`; weights ≈ **tired 41% · was 14% · it 10%** → **`it`→`tired`.** ✓
**Page 7 / Head 3 (action), query = `it`:** `q3_it=[0.24,0.3,0.06,0.09]`; top raw score `sat = 0.24·0.06+0.3·3.0+0.06·0.03 = 0.916`; softmax top ≈ **sat 46%**, then was → **`it`→`sat`.** ✓

**Page 2 cosine (anchor `it`):** cat 0.93 · tired 0.49 · was 0.45 · because 0.39 · The 0.34 · sat 0.31 → nearest = **cat.** ✓
**Page 3 analogy:** `king−man+woman = [1,1,0] = queen` (cos 1.00). ✓  `prince−boy+girl = [1,1,1] = princess`. ✓
**Page 4 neuron:** `0.9·1.3 + 0.3·(−0.9) + 0.6·0.7 + 0.1 = 1.42`; `tanh(1.42) = 0.89`. ✓

Three heads give three different winners for the same token `it` (cat / tired / sat) — the multi-head payoff is real, not staged.
