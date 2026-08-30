/* ============================================================
   Shared toy data + core ops for the "Meaning & Attention" track.
   EVERY lesson page loads this so all widgets compute from the
   SAME numbers. Do not redefine these in a page; just use them.
   Matrix-vector convention EVERYWHERE:  out[i] = Σⱼ W[i][j]·in[j]
   (row i dotted with the input vector). Do not transpose.
   ============================================================ */

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
const DIM_LABELS = ['animacy','action','state','function'];

/* ---- Head 1 : the "reference" head (it -> cat). Pages 5 & 6. ---- */
const Wq = [[2,0,0,1],[0,2,0,0],[0,0,1,0],[0,0,0,1]];      // query gain on animacy+function
const Wk = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0.3]];    // key advertises raw dims (function damped)
const Wv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0.5]];    // value = embedding with dim3 halved. NOTE: real multi-head attention has a separate Wv per head; this toy shares one because the heads page only draws routing (Q·K) arcs, never a per-head Value blend.

/* ---- Head 2 : "state/predicate" head (it -> tired). Page 7. ---- */
const Wq2 = [[0.3,0,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,0.3]]; // gain on the state dim
const Wk2 = [[0.3,0,0,0],[0,1,0,0],[0,0,3,0],[0,0,0,0.3]];

/* ---- Head 3 : "action" head (it -> sat). Page 7. ---- */
const Wq3 = [[0.3,0,0,0],[0,3,0,0],[0,0,0.3,0],[0,0,0,0.3]]; // gain on the action dim
const Wk3 = [[0.3,0,0,0],[0,3,0,0],[0,0,0.3,0],[0,0,0,0.3]];

/* ---- Core ops (same softmax shape as the classic temperature widget) ---- */
const matvec = (W,v)=> W.map(row => row.reduce((s,w,j)=> s + w*v[j], 0));
const dot    = (a,b)=> a.reduce((s,x,i)=> s + x*b[i], 0);
const mag    = (a)=> Math.sqrt(a.reduce((s,x)=> s + x*x, 0));
const cosine = (a,b)=> dot(a,b) / (mag(a)*mag(b) || 1);
function softmax(zs){ const m=Math.max(...zs); const ex=zs.map(z=>Math.exp(z-m)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(e=>e/s); }

// attention for one query token index qi under a head {Wqh,Wkh}. Raw logits (NO /sqrt(d)).
function attend(qi, Wqh, Wkh){
  const q = matvec(Wqh, E[TOKENS[qi]]);
  const scores = TOKENS.map(t => dot(q, matvec(Wkh, E[t])));
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

/* ============================================================
   ITER-2 additions — closing the forward pass (pages 8–15).
   All numbers hand-verified; see _course-dev/spec-iter2.md.
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
  W2: [
    [0.6, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],     // out0 animacy  <- h0,h4
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],     // out1 action   <- h1,h5
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],     // out2 state    <- h2,h7
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.4, 0.0]      // out3 function <- h3,h6
  ],
  b2: [0.0, 0.0, 0.0, 0.0],
  act: Math.tanh
};
// FFN forward (reuses matvec): hidden = tanh(W1·v + b1), out = W2·hidden + b2
function ffnForward(v){
  const pre = matvec(FFN.W1, v).map((s,i)=> s + FFN.b1[i]);
  const hid = pre.map(FFN.act);
  const out = matvec(FFN.W2, hid).map((s,i)=> s + FFN.b2[i]);
  return { pre, hid, out };
}

/* ---- Page 10 (unembedding): a final vector for the predicting position, dotted
   against every row of E -> 7 vocab-logits -> softmax -> ranked next token.     */
const UNEMBED_VEC = [0.9, 0.3, 1.4, 0.1];         // final vector at the predicting position
// vocab-logit(t) = dot(UNEMBED_VEC, E[t]) for every t in TOKENS  (reuses lesson-2 dot)

/* ---- Page 11 (sampling): candidate set for "...it was ___". ---- */
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

/* ---- Page 12 (transformer block): watch it's vector DRIFT over layers. ---- */
const DRIFT = {
  v0: [0.8, 0.1, 0.2, 0.3],                        // e_it, layer 0
  attractor: [1.4, 0.3, 0.25, 0.1],                // contextual "cat-flavored" target
  rate: 0.4,                                       // v_L = v_{L-1} + rate*(attractor - v_{L-1})
  layers: 6
};
function driftTo(L){                               // running vector at layer L
  let v = DRIFT.v0.slice();
  for(let i=0;i<L;i++) v = v.map((x,d)=> x + DRIFT.rate*(DRIFT.attractor[d]-x));
  return v;
}

/* ---- Page 13 (positional encoding): meaning in dims 0,1; position in dims 2,3. ---- */
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

/* ---- Page 14 (hallucination): confident-but-false outranks true. ---- */
const HALLUCINATE = {
  prompt: 'The capital of Australia is',
  logits: [
    { tok: 'Sydney',    z: 3.0, truth: false },
    { tok: 'Melbourne', z: 2.2, truth: false },
    { tok: 'Canberra',  z: 1.8, truth: true  },   // the correct answer, rank 3
    { tok: 'Perth',     z: 1.0, truth: false },
    { tok: 'Brisbane',  z: 0.8, truth: false }
  ]
};

/* ---- Page 15 (tokenization edges): illustrative, NOT computed. ---- */
const TOKENIZE = {
  'strawberry':   { chunks: ['str','aw','berry'],        ids: [1618, 707, 15717] },
  'cat':          { chunks: ['cat'],                     ids: [9246] },
  'unbelievable': { chunks: ['un','bel','iev','able'],   ids: [359, 6667, 12796, 429] },
  'GPT':          { chunks: ['G','PT'],                  ids: [38, 6316] }
};
const COUNT_LETTER = 'r';   // "how many r's in strawberry?" -> a human sees 3; the IDs carry none
