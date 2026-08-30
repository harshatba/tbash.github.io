# Learner Review — "How LLMs Work" (Part 1) + "How LLMs Run Fast" (Part 2)

*Reviewer persona: junior SWE, comfortable with Python/JS, zero ML background. I read both pages end to end, poked every widget, and read the JS to see what the widgets actually do vs. what the prose claims.*

---

## Overall impression

It mostly clicked, and that surprised me — the "it just predicts the next word, looped" framing in Part 1 §01 is a genuinely great on-ramp, and the color-coding (amber=tokens, cyan=vectors, violet=attention, rose=weights, green=training) gave me a mental filing system that paid off. The single best thing is Part 2's central thesis: "reading your prompt is fast; writing the reply is slow, because for every token the GPU must stream *all* the weights out of memory." That one sentence reframed the entire second half and made quantization, KV cache, and batching feel like answers to a question I actually had. The single most frustrating thing is the **middle of Part 1 (embeddings → neural network → attention)**: this is exactly where a no-math learner needs to see one concrete number move through one concrete operation, and instead the pages hand-wave with analogies ("meaning became geometry," Q/K/V are "roles a token plays") while the interactive widgets underneath are often *decorative* — they animate pre-baked numbers rather than computing anything I can trace. I finished feeling like I could *recite* the vocabulary but couldn't *derive* anything.

---

## Concept-by-concept confusion log

### Part 1 — How LLMs Work

| Concept | Verdict | The specific place I got lost |
|---|---|---|
| Next-token prediction (§01) | ✅ got it | Clean. The "pick one, stick it on the end, ask again" loop is the best sentence in the course. The 61%/15%/11% bars made "ranked list with probabilities" concrete. No complaints. |
| Tokens / tokenizer (§02) | ✅ got it | "contentedly" → `content` + `edly` with IDs is exactly the right demo. One gap: it never says *why* the tokenizer chops that way (frequency? who decided?), so "cat stays whole, contentedly splits" reads as arbitrary magic rather than "common = one token." |
| Embeddings (§03) | ⚠️ shaky | The jump that lost me: the page goes from "ID 2543 carries no meaning" straight to "so the model turns each token into a vector that places it as a point in space" — but it **never explains where that vector comes from**. Is it looked up in a table? Computed? Learned? (It's a learned lookup table, but the course never says the words "lookup table.") The `king − man + woman ≈ queen` box is asserted as a static fact; I can't *do* the subtraction, I just have to believe the arrow. And "hundreds of dimensions" shown as a 2D scatter is stated but the leap from 2 to hundreds is never made tangible. |
| Neural network / forward pass (§04) | ⚠️ shaky | The `out = squish(x₁·w₁ + … + b)` node formula is the most honest moment in the whole course — finally a real operation. But then it's **crammed together with three other big ideas in one section**: the forward pass, "training nudges weights," AND the parameter-scale toy-net→frontier widget. That's three separate lessons. Also "squish" is never named (it's an activation function) and never *motivated* — why squish at all? I dragged the input sliders and watched nodes recolor, but the widget never shows me a single multiplication happening, so I'm trusting the formula, not seeing it. |
| Positional encoding (§05) | ✅ got it | "dog bites man" vs "man bites dog" is a perfect hook, and "embedding = what the word is + where it sits" is a keeper. Minor: "a position tag added in" — *added* how, literally summed onto the vector? It says "added" but I wasn't sure if that's arithmetic or metaphor (it's literal addition, which is genuinely surprising and worth showing). |
| Attention (§06) | ❌ lost | This is the "big breakthrough" section and it's where I fell off hardest. The Q/K/V cards are pure analogy — "Query asks a question, Keys offer a label, Values get pulled in" — and the widget draws pretty violet arcs whose thickness I'm told is the attention weight. But **the arc weights are hard-coded in a table** (`aWeights` in the JS: "it"→cat is literally typed as 0.72). So the demo doesn't *compute* attention, it *replays a prediction*. As a learner I have no idea how "it" decided cat scores 0.72. The missing step: what is the actual math that turns Q and K into a number? (dot product → softmax, never mentioned). "Heads" get one sentence at the end and no visual. Too many ideas here (Q/K/V, weighting, softmax-implied, multi-head) for one page. |
| Transformer block (§07) | ⚠️ shaky | The stacked-block diagram (Attention → Add&Norm → Feed-forward → Add&Norm) is nice, and the layer slider is fun. But three unexplained things are introduced as "safety rails" in one breath: residual shortcuts, layer-norm, and the feed-forward "thinking step." "Keep the numbers stable" is hand-waved — stable from *what*? Why would numbers get unstable? And "feed-forward = think about each token" is doing a lot of undefined work. I understood the *shape* (stack N copies) but not *why each piece is there*. |
| Generation / temperature (§08) | ✅ got it | Best-executed widget in Part 1: the temperature slider **actually recomputes softmax live** (I checked the JS — real `Math.exp((z-m)/t)`), so dragging it and watching the bars flatten genuinely taught me what temperature does. This is the model for what every other widget should be. Only nit: "logits" (the `z` values) are never named; they're just "scores." |
| Training pipeline (§09) | ✅ got it | Three-stage card (pre-train → instruction tune → RLHF) is clear and the "train vs run / weights are frozen" myth-buster is valuable. RLHF is named but the *reinforcement learning* part isn't explained at all — it's just "humans rank, model is nudged." Fine for a first pass. |
| Architectures (dec / enc / enc-dec) (§10) | ⚠️ shaky | The attention-mask grid (lit cell = "can look at") is a great idea and the data-flow animation is the most polished thing on the page. But this section assumes I've fully internalized attention (which I hadn't) and then piles on causal vs bidirectional masking, cross-attention, AND three model families at once. For a beginner this is probably one section too advanced/dense to sit here; it's really three concepts. "Cross-attention" appears in the T5 diagram with zero explanation. |
| Scale & context window (§11) | ✅ got it | Simple, works. The "why a model forgets the start of a long chat" payoff for context window is exactly the kind of real-world hook that makes it stick. |
| Recap glossary (§12) | ✅ got it | Genuinely useful; I'd bookmark this. "Autoregressive" finally gets named here (it should appear back in §01/§08 where the looping is first shown). |

### Part 2 — How LLMs Run Fast

| Concept | Verdict | The specific place I got lost |
|---|---|---|
| Why it's slow / prefill vs decode (§01) | ✅ got it | Outstanding. The two-bar meter (math units near-empty, bandwidth pinned at 100% during decode) made "memory-bandwidth-bound" click instantly. Best single concept in either part. |
| Bits / fp32-16-bf16-fp8 (§02) | ⚠️ shaky | The sign/exponent/mantissa bit-strip is good and interactive. But it assumes I know what a floating-point number *is* under the hood, and "exponent = roughly how big, mantissa = the precise digits" went by fast. The bf16-vs-fp16 "range vs precision" trade is stated correctly but I couldn't *feel* it — I wanted to see a number overflow in fp16 and survive in bf16. |
| Quantization (§03) | ✅ got it | The snapping-to-levels SVG is the best widget in Part 2 — and it genuinely computes (I checked: real rounding, real per-group ranges, real outlier handling with deterministic weights). Toggling "protect outliers" and watching error drop actually taught me *why* GPTQ/AWQ exist. This is how all of Part 1's widgets should work. Slightly too much at once (bit-width + per-group + outliers = three toggles), but each is worth it. |
| KV cache (§04) | ⚠️ shaky | The prose is correct and the "stores K/V *vectors*, not the text" myth-buster is exactly the misconception I had. But the widget conflates two things: the recompute-vs-cache grid animation, and the memory tank with context/concurrency sliders. And critically — **it never shows me what a K or a V actually is.** It caches "keys and values of every earlier token," but since Part 1 never made K and V concrete, I'm caching two things I don't understand. This section depends on §06 of Part 1 having landed, and for me it hadn't. |
| Paging & prefix caching (§05) | ⚠️ shaky | Two distinct concepts (paged attention + prefix caching) stapled into one section, each with its own widget. Paged attention as "OS virtual memory for the cache" is a decent analogy but assumes I know how OS paging works (I sort of do; a non-CS learner wouldn't). The "reserve max wastes 58%" grid is good. Prefix caching clicked better (shared system prompt computed once). Should be two pages. |
| Speculative decoding (§06) | ✅ got it | The draft→verify→accept/reject animation is excellent and the "output is *exactly* what the target would've written" myth-buster resolves the obvious fear. The k-slider and aligned-vs-weak-draft toggle let me build real intuition about acceptance rate. One thing left implicit: *how* the target verifies all k tokens "in one parallel pass" (it's the same parallelism as prefill) — a callback to §01 would nail it. |
| Continuous batching (§07) | ✅ got it | The "throughput ≠ latency" distinction is the key insight and it's stated crisply. The slots-over-time grid is legible. Good. |
| More tricks: Flash/MoE/distill/early-exit (§08) | ⚠️ shaky | "Four ideas, one line each" is honest about being a flyover, but MoE especially deserves more — "route each token to 1-2 of 8 experts" raises the immediate question *who routes, and how?* that never gets answered. FlashAttention's "tiles that stay in on-chip memory" is too compressed to actually understand; it's a name-drop with a tiny SVG. |
| Recap matrix + decision guide (§09) | ✅ got it | The technique×bottleneck matrix and the "so which do you reach for?" if→then list are the most useful reference in either part. Great closer. |

**Cramming flags (should be split):** Part 1 §04 (forward pass + training + param-scale = 3 lessons), §06 (Q/K/V + weighting + softmax + multi-head = 3-4 lessons), §07 (feed-forward + residual + layer-norm = 3 lessons), §10 (three architectures + masking + cross-attention). Part 2 §05 (paging + prefix = 2 lessons), §03 (bit-width + per-group + outliers, though these earn their density).

---

## Genuine questions I still have

*Ordered most-basic → most-advanced. These are the questions the course left ringing.*

1. Where does a token's embedding vector actually *come from* — is it looked up in a giant table, one row per token, and is that table itself learned during training?
2. What *is* a weight, physically, in memory? The course says "billions of tunable knobs" — is each knob literally one floating-point number sitting at one address in the GPU's memory?
3. When §04 says numbers get "squished," what is the squish and why is it needed? What breaks if you don't squish?
4. The position tag is "added in" — is that literal arithmetic addition onto the embedding vector? How can adding a position number not destroy the meaning already encoded in that vector?
5. Attention: when "it" looks at "cat," is that literally a matrix multiply? Show me the numbers — how does Q for "it" and K for "cat" turn into the 0.72 the widget hard-codes?
6. What actually *is* a Query, a Key, and a Value as data? Are they three separate vectors computed from the same token's embedding by three separate weight matrices?
7. What is softmax, concretely? It's implied everywhere (probabilities, temperature, attention weights) but never defined — is it the same operation in all three places?
8. What are "logits"? The temperature widget calls them "scores" (`z`) — are these the raw outputs before softmax, and are they the same numbers coming out of the final layer as next-token scores?
9. A frontier model has one output "per possible token — tens of thousands of them" (§04). So the final layer produces ~50,000 numbers every single step? How does that map back to the embedding table?
10. What does "the model refines the meaning a little more" at each layer actually mean numerically — the vector for a token changes as it goes up the stack? Is the "cat" vector at layer 40 different from layer 1?
11. In feed-forward, what is being "thought about"? Is it per-token (each token's vector transformed independently) while attention is the only step where tokens talk to each other?
12. Why do the numbers become "unstable" without residual shortcuts and layer-norm? What does instability look like — values exploding to infinity?
13. What is a KV cache actually caching — the K vector and V vector for *which* tokens, and why specifically K and V but not Q?
14. Why does the KV cache grow *linearly* with context length? Is it one K and one V vector per token per layer per head?
15. In speculative decoding, how does the big target model check all k draft tokens "in one parallel pass" — is that the same trick that makes prefill fast, applied to the guesses?
16. Why does a bigger batch make each request *cheaper* (per token) but not *faster* (per request)? I get "share the weight-read" but I want the mechanism spelled out.
17. Mixture-of-Experts: *who* decides which 2 of 8 experts a token goes to, and is that router itself a learned network? Does the model pick different experts for different tokens in the same sentence?
18. Quantization rounds weights to 16 levels (int4) and "it still works" — *why* is a neural net so robust to that much rounding when normal software would break?
19. What exactly is fp16 vs bf16 "overflow"? When would a training number actually exceed fp16's range, and what happens when it does?
20. The context window is a "token budget" — but §11 (Part 1) and the KV-cache section (Part 2) both talk about long context costing memory. Is the context-window limit fundamentally a KV-cache-memory limit, or something else?
21. When temperature is 0, output is deterministic — but is it *bit-for-bit* identical every run, given all the floating-point math and batching on a GPU?
22. Is "attention" O(n²) in the number of tokens? The mask grid is n×n, which hints at it — is that why long context is expensive to *compute*, separate from the cache memory cost?
23. Prefix caching reuses a shared system prompt "computed once" — computed into *what*, exactly the KV cache for those prefix tokens? So prefix caching is literally KV-cache reuse across requests?
24. During RLHF, what is the "reinforcement" — the course says humans rank answers and the model is "nudged," but nudged by what signal, and how is that different from the instruction-tuning nudge?
25. Distillation trains a student to "imitate" a teacher — imitate the teacher's *final word choices*, or its full probability distribution over all tokens? Does the student ever see real training data or only the teacher's outputs?

---

## New topics I wish the course covered

- **What a GPU actually does during inference** — the whole of Part 2 hinges on "streaming weights from memory to compute units," but "compute units," VRAM, and memory bandwidth are never grounded. One diagram of a GPU (VRAM ↔ SMs ↔ on-chip SRAM) would make prefill/decode/FlashAttention click at once. *Why it matters: it's the physical stage every trick in Part 2 plays on, and it's invisible.*
- **The dot-product-and-softmax math of attention, with real small numbers** — the single biggest hole. *Why: attention is billed as "the breakthrough" but is currently pure analogy.*
- **Why models hallucinate, mechanically** — a course that nails "it's prediction, not retrieval" is perfectly set up to explain that confident wrong answers are just high-probability tokens with no truth-check. *Why: it's the #1 thing a real user asks, and this framing answers it.*
- **Tool use / function calling** — how a next-token predictor ends up "calling an API." *Why: it's how every modern assistant actually works and feels like magic without an explanation.*
- **Reasoning / "thinking" models** — the course's own myth-buster says "the math for one token is trivial, it's not thinking hard," which directly contradicts the marketing of reasoning models. *Why: I'd want to know what chain-of-thought / test-time compute actually is given what I just learned.*
- **RAG vs fine-tuning vs prompting** — three ways to "give the model new knowledge," and the course's "weights are frozen" point begs this question. *Why: it's the first practical decision an engineer faces, and embeddings (§03) are half of RAG already.*
- **Embeddings as a product / vector search** — §03 builds embeddings and never mentions they're independently useful for search/similarity. *Why: it's the most common thing a SWE actually ships with an LLM API.*
- **Tokenization edge cases** — why LLMs can't count letters in "strawberry," why they're bad at arithmetic, why non-English costs more tokens. *Why: concrete, memorable, and follows directly from §02.*
- **Multimodal (images/audio as tokens)** — one line on how a picture becomes tokens the same transformer eats. *Why: it's where everything is going and the token abstraction makes it explainable.*
- **top-p / top-k sampling** — §08 covers temperature but not the other two dials people actually set. *Why: they ship together in every API and the softmax widget could show all three.*
- **What "parameters" cost to *train*** vs to *run* — Part 2 is all about serving cost; a line on why pre-training costs millions of dollars would complete the picture.

---

## Where I want MORE interactivity (not just prose)

- **§03 Embeddings — a live analogy calculator.** Let me pick any three words from dropdowns and watch `A − B + C` land on a real nearest word, instead of the static `king − man + woman ≈ queen` box. Right now the vectors are invisible; I want to *do* the arithmetic.
- **§04 Neural network — show one multiplication.** The widget recolors nodes but never surfaces a single `x·w` product. I want to hover an edge and see `0.85 × 1.3 = 1.105`, then hover a node and see the full sum → squish. The formula is right there; make the widget *be* the formula.
- **§06 Attention — the actual scoring, not hard-coded arcs.** This is the #1 wanted widget. Let me click "it," see Q·K dot products computed against every other token as raw numbers, watch softmax turn them into the weights, then watch V get pulled in proportionally. The current arcs are a diagram of a conclusion; I want the calculation. (The infrastructure exists — the temperature widget already does live softmax.)
- **§06 — a multi-head toggle.** "Dozens of lenses in parallel" is text-only. Let me flip between 2-3 heads and see the arcs re-route (one tracks grammar, one tracks reference).
- **§02 (Part 2) Bits — an overflow demo.** A number-line where I type a value and watch it round in fp16 vs bf16, and push it big enough to see fp16 overflow to infinity while bf16 holds. Make "range vs precision" a thing I break, not read.
- **§04 (Part 2) KV cache — expose a K/V vector.** Even a fake 4-number vector per token, lighting up as it's cached, would make "it stores vectors not text" tangible instead of asserted.
- **§08 (Part 1) — add top-p / top-k sliders** beside temperature so I can watch the candidate list get truncated, not just reweighted.
- **§08 (Part 2) MoE card — an interactive router.** Type a token, watch the router light up which 2 experts fire. The static "2 of 8 lit" doesn't convey routing.

---

## Pacing & page structure

Both pages are doing far too much per scroll. Each is a single ~2,500-line document with 9-12 dense chapters, and several chapters are themselves 3 lessons wide (see cramming flags above). For the stated goal of *low concept-count per page*, I'd split each chapter into its own page with **one idea and one widget each**, chained with next/prev. Concretely:

**Part 1 → ~16 pages:**
1. The big idea: next-token prediction
2. Generation as a loop (autoregressive) — *pulled out of §01's footnote; it deserves its own page*
3. Tokens & the tokenizer
4. Token IDs → embeddings (the lookup table) — *new page making "where the vector comes from" explicit*
5. Embedding space & similarity (the scatter)
6. Directions in meaning-space (the analogy calculator)
7. One neuron: `squish(Σ x·w + b)` — *just the node*
8. The forward pass (numbers flowing through layers)
9. Weights are learned, not written (training preview) + the param-scale widget
10. Positional encoding
11. Attention I — Q, K, V as three vectors
12. Attention II — scoring with dot-product + softmax (the real math widget)
13. Attention III — multiple heads
14. The transformer block (feed-forward + residual + layer-norm, each briefly motivated)
15. Sampling & temperature (+ top-p/top-k)
16. Architectures — *possibly its own 3-page mini-series (decoder / encoder / enc-dec)*; plus training pipeline, scale/context, and the recap as closers.

**Part 2 → ~11 pages:**
1. Why it's slow: prefill vs decode (keep exactly as-is, it's the anchor)
2. How a weight is stored: bits
3. Quantization
4. (optional) Protecting outliers / per-group scales — *could split from quantization*
5. The KV cache: what it stores
6. KV cache: why it grows (the memory tank)
7. Paged attention
8. Prefix caching
9. Speculative decoding
10. Continuous batching
11. More tricks (Flash/MoE/distill/early-exit) + the recap matrix.

The recaps/glossaries at the end of each part are excellent and should stay as capstone pages. A persistent progress rail across the whole multi-page series (not just within one scroll) would help me feel the map.

---

## Verdict

**Not satisfactory as-is — it needs another design+build pass.** The scaffolding, visual language, and the two anchor concepts (next-token-loop in Part 1, prefill-vs-decode in Part 2) are genuinely strong, and a few widgets (temperature, quantization, speculative decoding) already hit the "poke it and understand it" bar. But the conceptual spine — embeddings and especially attention — is carried by analogy while the widgets underneath animate hard-coded answers instead of computing them, so a motivated no-math learner comes away able to recite the words but not trace a single number through the machine. Fix attention's math, make the mid-Part-1 widgets actually compute, and split the crammed chapters into one-idea pages, and this becomes excellent.
