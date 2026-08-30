/* ============================================================
   Part 3 — "Beyond the Basics": shared constants + helpers.
   Loaded AFTER course-data.js AND course-data-p2.js.
   Reuses: E, TOKENS, matvec, dot, cosine, softmax (p1);
           MODEL, GPU, paramCount, weightBytes, kvCacheBytes, BYTES (p2).
   All headline numbers hand-verified — see _course-dev/spec-part3.md.
   ============================================================ */

/* ------------------------------------------------------------------
   Page 1 · GPU anatomy — the memory hierarchy (reference figures).
   ------------------------------------------------------------------ */
const HIER = {
  sms:            108,          // Streaming Multiprocessors (compute units)
  registers_kb:   256,          // register file per SM (KB)
  sram_kb:        192,          // on-chip SRAM / shared+L1 per SM (KB)
  vram_gb:        80,           // HBM (GB)
  vram_bw_tbs:    2.0,          // HBM bandwidth (TB/s)   == GPU.bandwidth
  sram_bw_tbs:    19            // on-chip SRAM bandwidth (TB/s, ~10× HBM) — illustrative
};
function fitsInSram(bytes){ return bytes <= HIER.sram_kb * 1024; }
function fitsInVram(bytes){ return bytes <= HIER.vram_gb * 1e9; }

/* ------------------------------------------------------------------
   Page 2 · Grouped-query attention — KV cache vs n_kv_heads.
   ------------------------------------------------------------------ */
const GQA = {
  configs: [
    { name:'MHA',   n_kv:32, blurb:'every query head has its own K/V (Part-1 lesson 6)' },
    { name:'GQA-8', n_kv:8,  blurb:'8 K/V heads shared across 32 query heads (Llama-3 8B)' },
    { name:'GQA-4', n_kv:4,  blurb:'4 K/V groups' },
    { name:'MQA',   n_kv:1,  blurb:'one K/V head for all 32 query heads (extreme)' }
  ]
};
function gqaKvBytesPerToken(m, n_kv, bytes){ m=m||MODEL; return 2*m.layers*n_kv*m.head_dim*bytes; }
function gqaKvCacheBytes(m, n_kv, seq, batch, bytes){ return gqaKvBytesPerToken(m,n_kv,bytes)*seq*batch; }
function gqaShrink(m, n_kv){ m=m||MODEL; return m.n_heads / n_kv; }

/* ------------------------------------------------------------------
   Page 3 · FlashAttention — tiling model (per attention head).
   ------------------------------------------------------------------ */
const FLASH = { d: 128, bytes: 2, sram_budget: HIER.sram_kb*1024 /* 196608 B */ };
function scoreMatrixBytes(n, bytes){ bytes=bytes||FLASH.bytes; return n*n*bytes; }
function flashTileBytes(B, d, bytes){ d=d||FLASH.d; bytes=bytes||FLASH.bytes; return (3*B*d + B*B)*bytes; }
function flashTilesPerAxis(n, B){ return Math.ceil(n/B); }
function flashMemReduction(n, B){ return (n/B)*(n/B); }
function flashKvPasses(n, B){ return flashTilesPerAxis(n, B); }

/* ------------------------------------------------------------------
   Page 4 · Mixture-of-Experts — router + active-vs-total params.
   Router reuses Part-1 E (4 toy dims) + matvec + softmax.
   ------------------------------------------------------------------ */
const MOE = {
  n_experts: 8,
  top_k: 2,
  d_model: 4096, d_ff: 14336, layers: 32, vocab: 32000,   // Mixtral-8x7B-like
  gate: [
    [ 3.0, 0.0, 0.0, 0.0],   // e0  animacy / entity
    [ 0.0, 3.0, 0.0, 0.0],   // e1  action / verb
    [ 0.0, 0.0, 3.0, 0.0],   // e2  state / adjective
    [ 0.0, 0.0, 0.0, 3.0],   // e3  function / glue
    [ 1.5, 1.5, 0.0, 0.0],   // e4  animate + action
    [ 1.5, 0.0, 1.5, 0.0],   // e5  animate + state
    [ 0.5, 0.5, 0.5, 0.5],   // e6  generalist
    [-1.0,-1.0,-1.0,-1.0]    // e7  cold / rarely picked
  ],
  labels: ['animacy','action','state','function','animate+action','animate+state','generalist','cold']
};
function moeRoute(embed, k){
  k = k || MOE.top_k;
  const logits = matvec(MOE.gate, embed);
  const order  = logits.map((z,i)=>({i,z})).sort((a,b)=>b.z-a.z);
  const top    = order.slice(0,k);
  const w      = softmax(top.map(o=>o.z));
  return { logits, top: top.map((o,j)=>({expert:o.i, label:MOE.labels[o.i], logit:o.z, weight:w[j]})) };
}
function moeParams(p){
  p = p || MOE;
  const embed  = p.vocab * p.d_model;
  const attn   = 4 * p.d_model * p.d_model;
  const router = p.n_experts * p.d_model;
  const expert = 3 * p.d_model * p.d_ff;
  const total  = embed + p.layers * (attn + router + p.n_experts * expert);
  const active = embed + p.layers * (attn + router + p.top_k   * expert);
  return { embed, attn, router, expert, total, active, activeFrac: active/total };
}

/* ------------------------------------------------------------------
   Page 5 · Tool use — scripted stream + a MOCK tool run live.
   ------------------------------------------------------------------ */
const TOOLUSE = {
  userPrompt: "What's the weather in Paris right now?",
  reason:      ['I','need','live','data','—','calling','a','tool','.'],
  callTokens:  ['{','"tool"',':','"get_weather"',',','"args"',':','{','"city"',':','"<CITY>"','}','}'],
  weatherDB: {
    Paris:  { tempC:18, sky:'cloudy' },
    Tokyo:  { tempC:24, sky:'clear'  },
    Cairo:  { tempC:34, sky:'sunny'  },
    Oslo:   { tempC: 6, sky:'rain'   }
  },
  cities: ['Paris','Tokyo','Cairo','Oslo']
};
function runWeather(city){ return TOOLUSE.weatherDB[city] || null; }
function toolResultTokens(city){
  const r = runWeather(city);
  return ['{','"tempC"',':', String(r.tempC), ',','"sky"',':', '"'+r.sky+'"','}'];
}
function toolAnswer(city){
  const r = runWeather(city);
  return `It's ${r.tempC}°C and ${r.sky} in ${city} right now.`;
}

/* ------------------------------------------------------------------
   Page 6 · Reasoning — accuracy vs thinking-tokens (saturating curve).
   ------------------------------------------------------------------ */
const REASON = {
  presets: {
    typical: { floor:0.30, cap:0.85, tau:120, label:'a typical problem' },
    hard:    { floor:0.15, cap:0.80, tau:200, label:'a hard problem'    },
    easy:    { floor:0.70, cap:0.95, tau:60 , label:'an easy problem'   }
  },
  maxTokens: 600
};
function reasonAcc(t, p){ p = p || REASON.presets.typical; return p.floor + (p.cap-p.floor)*(1 - Math.exp(-t/p.tau)); }
function reasonMarginal(t, step, p){ return 100*(reasonAcc(t,p) - reasonAcc(Math.max(0,t-step),p)); }

/* ------------------------------------------------------------------
   Page 7 · RAG — tiny corpus + queries as 4-dim vectors (E-style).
   Retrieval reuses Part-1 cosine. dims: [animals, weather, cooking, space]
   ------------------------------------------------------------------ */
const RAG = {
  dims: ['animals','weather','cooking','space'],
  passages: [
    { id:'P0', text:'Cats purr when content and sleep about 15 hours a day.',      vec:[1.0,0.0,0.1,0.0] },
    { id:'P1', text:'A thunderstorm forms when warm, moist air rises quickly.',    vec:[0.0,1.0,0.0,0.1] },
    { id:'P2', text:'Caramelize onions by cooking them slowly in butter ~40 min.', vec:[0.0,0.1,1.0,0.0] },
    { id:'P3', text:'Jupiter is the largest planet; its Red Spot is a giant storm.',vec:[0.0,0.2,0.0,1.0] },
    { id:'P4', text:'Penguins are flightless birds that huddle for warmth.',       vec:[0.9,0.3,0.0,0.1] },
    { id:'P5', text:'Sourdough rises from wild yeast fermenting the dough.',       vec:[0.1,0.0,0.9,0.0] }
  ],
  queries: [
    { q:'How long do cats sleep?',          vec:[1.0,0.0,0.0,0.0] },
    { q:'What temperature to cook onions?', vec:[0.0,0.1,1.0,0.0] },
    { q:'Which is the biggest planet?',     vec:[0.0,0.1,0.0,1.0] }
  ]
};
function ragRetrieve(qvec, topN){
  topN = topN || 2;
  return RAG.passages
    .map(p => ({ id:p.id, text:p.text, vec:p.vec, cos: cosine(qvec, p.vec) }))
    .sort((a,b) => b.cos - a.cos)
    .slice(0, topN);
}

/* ------------------------------------------------------------------
   Page 8 · Multimodal — 4×4 toy image → 2×2 patches → 4-dim embeddings.
   ------------------------------------------------------------------ */
const IMG = [
  [0.9, 0.1, 0.1, 0.1],
  [0.1, 0.9, 0.1, 0.1],
  [0.1, 0.1, 0.9, 0.1],
  [0.1, 0.1, 0.1, 0.9]
];
const PATCH = 2;
const WPATCH = [
  [ 0.25, 0.25, 0.25, 0.25],   // e0 brightness (mean)
  [ 0.5 ,-0.5 , 0.5 ,-0.5 ],   // e1 left-right edge
  [ 0.5 , 0.5 ,-0.5 ,-0.5 ],   // e2 top-bottom edge
  [ 0.5 ,-0.5 ,-0.5 , 0.5 ]    // e3 diagonal
];
const PATCH_DIMS = ['brightness','LR-edge','TB-edge','diagonal'];
const PATCH_POS = [ [0,0,0,0.1],[0,0,0,0.2],[0,0,0,0.3],[0,0,0,0.4] ];
function patchesOf(img, p){
  const out = []; const g = img.length;
  for(let pr=0; pr<g; pr+=p){ for(let pc=0; pc<g; pc+=p){
    const px = [];
    for(let r=0;r<p;r++) for(let c=0;c<p;c++) px.push(img[pr+r][pc+c]);
    out.push({ row:pr/p, col:pc/p, pixels:px });
  }}
  return out;
}
function patchEmbed(pixels){ return matvec(WPATCH, pixels); }
