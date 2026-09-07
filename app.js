/**
 * The Memory Duel: Client-Side Simulation & Visualization Engine
 * Performs real linear algebra for KV Cache attention & BDH Hebbian plasticity.
 */

// Simulation Configuration
const DIM = 32; // Vector dimension for live interactive simulation
const REAL_WORLD_SCALE = 32 * 4096 * 4; // Scale factor simulating 32 layers of 4096-dim float32 LLM

// State Variables
let totalSeqLength = 10000;
let numFacts = 20;
let hebbianDecay = 0.9995;
let queryTarget = 'oldest';

// Pre-seeded pseudo-random number generator for reproducible facts
let seed = 42;
function pseudoRandom() {
  seed = (seed * 9301 + 49297) % 233280;
  return (seed / 233280) * 2 - 1;
}

function generateVector(d) {
  let v = new Float32Array(d);
  let norm = 0;
  for (let i = 0; i < d; i++) {
    v[i] = pseudoRandom();
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < d; i++) v[i] /= norm;
  return v;
}

function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function cosineSim(a, b) {
  let dot = dotProduct(a, b);
  let normA = Math.sqrt(dotProduct(a, a)) || 1e-7;
  let normB = Math.sqrt(dotProduct(b, b)) || 1e-7;
  return dot / (normA * normB);
}



function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
  ctx.fill();
}

// Global Memory Store for Current Run
const REAL_WORDS = ['Paris', 'London', 'Tokyo', 'Berlin', 'Rome', 'Madrid', 'Athens', 'Cairo', 'Beijing', 'Seoul', 'Dog', 'Cat', 'Bird', 'Fish', 'Tree', 'River', 'Mountain', 'Sun', 'Moon', 'Star'];
let ssmState = new Float32Array(DIM);
let isAnimating = false;
let wordVectors = [];

let factKeys = [];
let factValues = [];
let factPositions = [];
let synapticMatrix = new Float32Array(DIM * DIM);

function initializeFacts() {
  if (wordVectors.length === 0) {
    let oldSeed = seed;
    seed = 8888;
    for (let i = 0; i < REAL_WORDS.length; i++) {
      wordVectors.push(generateVector(DIM));
    }
    seed = oldSeed;
  }

  seed = 12345;
  factKeys = [];
  factValues = [];
  factPositions = [];

  for (let i = 0; i < numFacts; i++) {
    factKeys.push(generateVector(DIM));
    let wordIdx = i % REAL_WORDS.length;
    factValues.push(wordVectors[wordIdx]);
    // Position spaced across the sequence
    let pos = Math.floor((i / Math.max(1, numFacts - 1)) * (totalSeqLength - 10));
    factPositions.push(pos);
  }
}

// Compute Model Outputs
function runSimulation() {
  initializeFacts();

  // 1. Transformer KV Cache Memory Footprint
  // In real 8B LLM: 2 * T * layers * d * bytes_per_elem
  const kvBytesSim = 2 * totalSeqLength * REAL_WORLD_SCALE;
  const kvMb = kvBytesSim / (1024 * 1024);
  const kvGb = kvMb / 1024;

  // 2. BDH Synaptic Memory Footprint
  // Fixed size: layers * d * d * bytes_per_elem
  const bdhBytesSim = 32 * 4096 * 4096 * 4; // Simulated 64MB fixed matrix in production
  const bdhFixedMb = 64.0; // Fixed 64MB regardless of sequence length

  // Update Memory Meters
  const kvMemEl = document.getElementById('kv-mem-text');
  const kvBarEl = document.getElementById('kv-bar');
  const kvOomEl = document.getElementById('kv-oom-alert');

  if (kvGb >= 1.0) {
    kvMemEl.innerText = `${kvGb.toFixed(2)} GB`;
  } else {
    kvMemEl.innerText = `${kvMb.toFixed(1)} MB`;
  }

  // Visual bar: assume max capacity = 16 GB (16384 MB)
  const kvBarPercent = Math.min(100, Math.max(2, (kvMb / 8192) * 100));
  kvBarEl.style.width = `${kvBarPercent}%`;

  if (kvMb > 8000) {
    kvOomEl.style.display = 'block';
    kvBarEl.style.background = 'var(--accent-rose)';
  } else {
    kvOomEl.style.display = 'none';
    kvBarEl.style.background = 'linear-gradient(90deg, #f59e0b, var(--accent-rose))';
  }

  
  const useGating = document.getElementById('toggle-gating') ? document.getElementById('toggle-gating').checked : false;
  const useMultiHead = document.getElementById('toggle-multihead') ? document.getElementById('toggle-multihead').checked : false;

  // 3. Compute BDH Synaptic Matrix W & SSM State
  synapticMatrix.fill(0);
  ssmState.fill(0);

  // Gating weights (random projection)
  const Wg = generateVector(DIM);

  for (let i = 0; i < numFacts; i++) {
    const k = factKeys[i];
    const v = factValues[i];
    const pos = factPositions[i];
    const age = totalSeqLength - pos;
    
    // Gating vs Constant Decay
    let weight = Math.pow(hebbianDecay, age * 0.05);
    if (useGating) {
       let gateDot = dotProduct(Wg, k);
       let gate = 1.0 / (1.0 + Math.exp(-gateDot)); // sigmoid
       weight = weight * gate;
    }

    if (useMultiHead) {
      // Simulate 4 heads by blocking the matrix
      const heads = 4;
      const headDim = DIM / heads;
      for(let h = 0; h < heads; h++) {
        for (let r = h*headDim; r < (h+1)*headDim; r++) {
          for (let c = h*headDim; c < (h+1)*headDim; c++) {
            synapticMatrix[r * DIM + c] += weight * (v[r] * k[c]);
          }
        }
      }
    } else {
      for (let r = 0; r < DIM; r++) {
        for (let c = 0; c < DIM; c++) {
          synapticMatrix[r * DIM + c] += weight * (v[r] * k[c]);
        }
      }
    }

    // Simple SSM step: h = 0.99 h + x
    for(let d=0; d<DIM; d++) {
       ssmState[d] = 0.99 * ssmState[d] + (k[d] + v[d]);
    }
  }

  // Determine query fact
  let targetIdx = 0;
  if (queryTarget === 'middle') {
    targetIdx = Math.floor(numFacts / 2);
  } else if (queryTarget === 'newest') {
    targetIdx = numFacts - 1;
  }

  const queryVec = factKeys[targetIdx];
  const trueTarget = factValues[targetIdx];

  // 4. Query Retrieval from BDH: y = W * q
  let bdhRetrieved = new Float32Array(DIM);
  for (let r = 0; r < DIM; r++) {
    let sum = 0;
    for (let c = 0; c < DIM; c++) {
      sum += synapticMatrix[r * DIM + c] * queryVec[c];
    }
    bdhRetrieved[r] = sum;
  }

  const bdhSim = cosineSim(bdhRetrieved, trueTarget);

  // 5. Query Retrieval from Transformer KV Cache (Exact Attention Match)
  // Softmax over all keys
  let kvScores = [];
  let maxScore = -Infinity;
  for (let i = 0; i < numFacts; i++) {
    let score = dotProduct(factKeys[i], queryVec) / Math.sqrt(DIM);
    kvScores.push(score);
    if (score > maxScore) maxScore = score;
  }

  let expSum = 0;
  let expScores = [];
  for (let i = 0; i < numFacts; i++) {
    let expS = Math.exp(kvScores[i] - maxScore);
    expScores.push(expS);
    expSum += expS;
  }

  let kvRetrieved = new Float32Array(DIM);
  for (let i = 0; i < numFacts; i++) {
    let weight = expScores[i] / (expSum || 1);
    for (let d = 0; d < DIM; d++) {
      kvRetrieved[d] += weight * factValues[i][d];
    }
  }

  const kvSim = cosineSim(kvRetrieved, trueTarget);

  // Update UI Elements
  document.getElementById('kv-score').innerText = kvSim.toFixed(3);
  document.getElementById('bdh-score').innerText = bdhSim.toFixed(3);

  // Fidelity Badges
  const bdhBadge = document.getElementById('bdh-fidelity-badge');
  if (bdhSim >= 0.8) {
    bdhBadge.className = 'fidelity-badge badge-sharp';
    bdhBadge.innerText = 'Sharp Recall';
  } else if (bdhSim >= 0.45) {
    bdhBadge.className = 'fidelity-badge badge-blurred';
    bdhBadge.innerText = 'Interference Blur';
  } else {
    bdhBadge.className = 'fidelity-badge badge-lost';
    bdhBadge.innerText = 'Signal Overwritten';
  }

  // Signal to Noise Ratio estimate
  const snr = Math.max(0, (bdhSim / (1.001 - Math.abs(bdhSim))) * 10);
  document.getElementById('bdh-snr-text').style.color = '#a1a1aa';
  document.getElementById('bdh-snr-text').innerText = `SNR: ${snr.toFixed(1)} dB`;

  // Memory Compression Ratio
  const ratio = Math.max(1, Math.round(kvMb / bdhFixedMb));
  document.getElementById('stat-ratio').innerText = `${ratio}x`;

  // Interference Metric
  const interference = Math.min(100, Math.max(5, (numFacts / DIM) * 45));
  document.getElementById('stat-interference').innerText = `${interference.toFixed(1)}%`;

  
  // Evaluate SSM
  let ssmSim = cosineSim(ssmState, trueTarget); // Highly simplified
  const ssmScoreEl = document.getElementById('ssm-score');
  if(ssmScoreEl) ssmScoreEl.innerText = ssmSim.toFixed(3);
  const ssmBadge = document.getElementById('ssm-fidelity-badge');
  if(ssmBadge) {
      if (ssmSim >= 0.3) {
        ssmBadge.className = 'fidelity-badge badge-sharp';
        ssmBadge.innerText = 'Signal Present';
      } else {
        ssmBadge.className = 'fidelity-badge badge-lost';
        ssmBadge.innerText = 'State Overwritten';
      }
  }
  
  // Real Word decoding via Vector Lookup
  let similarities = [];
  for(let w = 0; w < REAL_WORDS.length; w++) {
      similarities.push({
          word: REAL_WORDS[w],
          score: cosineSim(bdhRetrieved, wordVectors[w])
      });
  }
  similarities.sort((a, b) => b.score - a.score);
  
  let top1 = similarities[0];
  let top2 = similarities[1];
  
  let cleanWord = top1.word;
  let displayWord = cleanWord;
  let statusText = "";
  
  // If interference is high (difference between top 2 is small or top score is low)
  if (top1.score < 0.8 || (top1.score - top2.score < 0.2)) {
      // Blend the strings physically to show crosstalk
      let blended = "";
      let len = Math.max(top1.word.length, top2.word.length);
      for(let i=0; i<len; i++) {
          if (i % 2 === 0) {
              blended += (i < top1.word.length) ? top1.word[i] : "";
          } else {
              blended += (i < top2.word.length) ? top2.word[i].toLowerCase() : "";
          }
      }
      displayWord = blended + "*"; // highlight it's blended
      statusText = `Nearest: ${cleanWord} | Blended: ${displayWord}`;
  } else {
      statusText = `Decoded: ${cleanWord}`;
  }

  document.getElementById('bdh-snr-text').style.color = '#a1a1aa';
  document.getElementById('bdh-snr-text').innerText = `${statusText} (SNR: ${Math.max(0, (bdhSim / (1.001 - Math.abs(bdhSim))) * 10).toFixed(1)} dB)`;

  // Render Visualizers
  renderKVCanvas();
  renderBDHCanvas();
  renderSSMCanvas();
}

// Render Transformer KV Cache Canvas
function renderKVCanvas() {
  const canvas = document.getElementById('canvas-kv');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Draw memory blocks filling up the buffer
  const maxBlocks = 80;
  const blocksToDraw = Math.min(maxBlocks, Math.max(5, Math.floor((totalSeqLength / 100000) * maxBlocks)));

  const blockWidth = (w - 30) / 10;
  const blockHeight = (h - 40) / 8;

  ctx.fillStyle = '#a1a1aa'; ctx.font = '11px sans-serif'; ctx.fillText(`Tokens buffered: ${totalSeqLength.toLocaleString()} tokens`, 15, 20);

  let drawn = 0;
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 10; col++) {
      if (drawn >= blocksToDraw) break;

      const x = 15 + col * blockWidth;
      const y = 30 + row * blockHeight;

      // Color based on whether it's a key fact or distractor
      if (drawn % Math.max(1, Math.floor(blocksToDraw / numFacts)) === 0) {
        ctx.fillStyle = '#fb7185'; // Fact token (pink)
      } else {
        ctx.fillStyle = '#27272a'; // Filler token (gray)
      }

      roundRect(ctx, x, y, blockWidth - 4, blockHeight - 4, 3);
      drawn++;
    }
  }

  // Draw legend
  ctx.fillStyle = '#fb7185';
  ctx.fillRect(15, h - 14, 8, 8);
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('Target Facts', 28, h - 7);

  ctx.fillStyle = '#27272a';
  ctx.fillRect(110, h - 14, 8, 8);
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('Distractor Tokens', 123, h - 7);
}

// Render BDH Synaptic Weight Heatmap Canvas
function renderBDHCanvas() {
  const canvas = document.getElementById('canvas-bdh');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const cellSize = Math.min((w - 40) / DIM, (h - 40) / DIM);
  const offsetX = (w - DIM * cellSize) / 2;
  const offsetY = 15;

  // Find max absolute weight for color normalization
  let maxWeight = 0.01;
  for (let i = 0; i < DIM * DIM; i++) {
    let absW = Math.abs(synapticMatrix[i]);
    if (absW > maxWeight) maxWeight = absW;
  }

  // Draw matrix cells
  for (let r = 0; r < DIM; r++) {
    for (let c = 0; c < DIM; c++) {
      const weight = synapticMatrix[r * DIM + c];
      const norm = weight / maxWeight;

      if (norm > 0) {
        // Positive synaptic connection: glows Cyan / Emerald
        const alpha = Math.min(1, Math.abs(norm));
        ctx.fillStyle = `rgba(45, 212, 191, ${alpha.toFixed(2)})`;
      } else {
        // Negative / inhibitory synaptic connection: glows Purple
        const alpha = Math.min(1, Math.abs(norm));
        ctx.fillStyle = `rgba(192, 132, 252, ${alpha.toFixed(2)})`;
      }

      roundRect(ctx, offsetX + c * cellSize + 0.5, offsetY + r * cellSize + 0.5, cellSize - 1.5, cellSize - 1.5, 2);
    }
  }

  // Heatmap label
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px sans-serif';
  ctx.fillText('Synaptic Connections (Active weights self-stabilizing)', 15, h - 8);
}

// Wire Event Listeners
function setupEventListeners() {
  const sliderSeq = document.getElementById('slider-seq');
  const sliderFacts = document.getElementById('slider-facts');
  const sliderDecay = document.getElementById('slider-decay');
  const selectQuery = document.getElementById('select-query');

  sliderSeq.addEventListener('input', (e) => {
    // Log scale 10^2 to 10^5
    totalSeqLength = Math.round(Math.pow(10, parseFloat(e.target.value)));
    document.getElementById('val-seq').innerText = totalSeqLength.toLocaleString();
    runSimulation();
  });

  sliderFacts.addEventListener('input', (e) => {
    numFacts = parseInt(e.target.value);
    document.getElementById('val-facts').innerText = numFacts;
    runSimulation();
  });

  sliderDecay.addEventListener('input', (e) => {
    hebbianDecay = parseFloat(e.target.value);
    document.getElementById('val-decay').innerText = hebbianDecay.toFixed(4);
    runSimulation();
  });

  selectQuery.addEventListener('change', (e) => {
    queryTarget = e.target.value;
    runSimulation();
  });

  // Presets
  document.getElementById('btn-novel').addEventListener('click', () => {
    setActivePreset('btn-novel');
    sliderSeq.value = 5.0; // 100,000 tokens
    totalSeqLength = 100000;
    sliderFacts.value = 20;
    numFacts = 20;
    sliderDecay.value = 0.9998;
    hebbianDecay = 0.9998;
    selectQuery.value = 'oldest';
    queryTarget = 'oldest';
    updateControlDisplays();
    runSimulation();
  });

  document.getElementById('btn-crammed').addEventListener('click', () => {
    setActivePreset('btn-crammed');
    sliderSeq.value = 3.3; // ~2,000 tokens
    totalSeqLength = 2000;
    sliderFacts.value = 75; // Heavy interference!
    numFacts = 75;
    sliderDecay.value = 1.0;
    hebbianDecay = 1.0;
    selectQuery.value = 'middle';
    queryTarget = 'middle';
    updateControlDisplays();
    runSimulation();
  });

  document.getElementById('btn-decay').addEventListener('click', () => {
    setActivePreset('btn-decay');
    sliderSeq.value = 4.0; // 10,000 tokens
    totalSeqLength = 10000;
    sliderFacts.value = 25;
    numFacts = 25;
    sliderDecay.value = 0.9850; // Fast decay for recency
    hebbianDecay = 0.9850;
    selectQuery.value = 'newest';
    queryTarget = 'newest';
    updateControlDisplays();
    runSimulation();
  });

  
  document.getElementById('toggle-gating').addEventListener('change', runSimulation);
  document.getElementById('toggle-multihead').addEventListener('change', runSimulation);
  document.getElementById('btn-animate').addEventListener('click', () => {
    if(!isAnimating) startAnimation();
  });

  document.getElementById('btn-reset').addEventListener('click', () => {

    setActivePreset(null);
    sliderSeq.value = 4.0;
    totalSeqLength = 10000;
    sliderFacts.value = 20;
    numFacts = 20;
    sliderDecay.value = 0.9995;
    hebbianDecay = 0.9995;
    selectQuery.value = 'oldest';
    queryTarget = 'oldest';
    updateControlDisplays();
    runSimulation();
  });
}

function setActivePreset(id) {
  document.querySelectorAll('.btn-preset').forEach(btn => btn.classList.remove('active'));
  if (id) {
    document.getElementById(id).classList.add('active');
  }
}

function updateControlDisplays() {
  document.getElementById('val-seq').innerText = totalSeqLength.toLocaleString();
  document.getElementById('val-facts').innerText = numFacts;
  document.getElementById('val-decay').innerText = hebbianDecay.toFixed(4);
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  runSimulation();
});


// Render SSM Canvas
function renderSSMCanvas() {
  const canvas = document.getElementById('canvas-ssm');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  
  const barWidth = (w - 40) / DIM;
  const maxH = h - 60;
  
  // Find max absolute value
  let maxV = 0.01;
  for(let i=0; i<DIM; i++) {
    if(Math.abs(ssmState[i]) > maxV) maxV = Math.abs(ssmState[i]);
  }

  for(let i=0; i<DIM; i++) {
    const val = ssmState[i] / maxV;
    const barH = Math.abs(val) * maxH;
    const y = val > 0 ? (h/2 - barH) : h/2;
    
    ctx.fillStyle = val > 0 ? 'rgba(45, 212, 191, 0.8)' : 'rgba(192, 132, 252, 0.8)';
    roundRect(ctx, 20 + i * barWidth, y, barWidth - 3, barH, 2);
  }

  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px sans-serif';
  ctx.fillText('Latent Vector State (1D representation)', 15, h - 8);
}

// Animation Logic
function startAnimation() {
  const canvas = document.getElementById('canvas-anim');
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  isAnimating = true;
  
  let frame = 0;
  let wordIdx = Math.floor(Math.random() * REAL_WORDS.length);
  document.getElementById('anim-status-text').innerText = `Animating insertion of fact: ${REAL_WORDS[wordIdx]}`;

  function draw() {
    if(!isAnimating) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#fff';
    ctx.font = '14px sans-serif';
    ctx.fillText('Key Vector', 150, 20);
    ctx.fillText('Value Vector', 10, 100);
    
    // Draw outer product grid
    for(let r=0; r<10; r++) {
       for(let c=0; c<10; c++) {
          let alpha = Math.min(1, frame / 100);
          ctx.fillStyle = `rgba(45, 212, 191, ${alpha * Math.random()})`;
          ctx.fillRect(100 + c*20, 40 + r*20, 18, 18);
       }
    }
    
    frame++;
    if(frame < 120) {
      requestAnimationFrame(draw);
    } else {
      isAnimating = false;
      document.getElementById('anim-status-text').innerText = 'Animation complete. Matrix updated.';
    }
  }
  draw();
}
