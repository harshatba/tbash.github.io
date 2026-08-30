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
const Wv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0.5]];    // value = embedding with dim3 halved (shared by all heads)

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
