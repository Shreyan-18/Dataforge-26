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

// Global Memory Store for Current Run
let factKeys = [];
let factValues = [];
let factPositions = [];
let synapticMatrix = new Float32Array(DIM * DIM);

function initializeFacts() {
  seed = 12345;
  factKeys = [];
  factValues = [];
  factPositions = [];

  for (let i = 0; i < numFacts; i++) {
    factKeys.push(generateVector(DIM));
    factValues.push(generateVector(DIM));
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

  // 3. Compute BDH Synaptic Matrix W
  synapticMatrix.fill(0);

  // Write facts into synaptic matrix using Hebbian rule: W = sum lambda^(T - t_i) * (v_i x k_i^T)
  for (let i = 0; i < numFacts; i++) {
    const k = factKeys[i];
    const v = factValues[i];
    const pos = factPositions[i];
    const age = totalSeqLength - pos;
    // Effective decay weight
    const weight = Math.pow(hebbianDecay, age * 0.05);

    for (let r = 0; r < DIM; r++) {
      for (let c = 0; c < DIM; c++) {
        synapticMatrix[r * DIM + c] += weight * (v[r] * k[c]);
      }
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
  document.getElementById('bdh-snr-text').innerText = `SNR: ${snr.toFixed(1)} dB`;

  // Memory Compression Ratio
  const ratio = Math.max(1, Math.round(kvMb / bdhFixedMb));
  document.getElementById('stat-ratio').innerText = `${ratio}x`;

  // Interference Metric
  const interference = Math.min(100, Math.max(5, (numFacts / DIM) * 45));
  document.getElementById('stat-interference').innerText = `${interference.toFixed(1)}%`;

  // Render Visualizers
  renderKVCanvas();
  renderBDHCanvas();
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

  ctx.fillStyle = '#1e293b';
  ctx.font = '11px sans-serif';
  ctx.fillText(`Tokens buffered: ${totalSeqLength.toLocaleString()} tokens`, 15, 20);

  let drawn = 0;
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 10; col++) {
      if (drawn >= blocksToDraw) break;

      const x = 15 + col * blockWidth;
      const y = 30 + row * blockHeight;

      // Color based on whether it's a key fact or distractor
      if (drawn % Math.max(1, Math.floor(blocksToDraw / numFacts)) === 0) {
        ctx.fillStyle = '#f43f5e'; // Fact token (pink)
      } else {
        ctx.fillStyle = '#334155'; // Filler token (gray)
      }

      ctx.fillRect(x, y, blockWidth - 3, blockHeight - 3);
      drawn++;
    }
  }

  // Draw legend
  ctx.fillStyle = '#f43f5e';
  ctx.fillRect(15, h - 14, 8, 8);
  ctx.fillStyle = '#94a3b8';
  ctx.fillText('Target Facts', 28, h - 7);

  ctx.fillStyle = '#334155';
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
        ctx.fillStyle = `rgba(6, 182, 212, ${alpha.toFixed(2)})`;
      } else {
        // Negative / inhibitory synaptic connection: glows Purple
        const alpha = Math.min(1, Math.abs(norm));
        ctx.fillStyle = `rgba(168, 85, 247, ${alpha.toFixed(2)})`;
      }

      ctx.fillRect(offsetX + c * cellSize, offsetY + r * cellSize, cellSize - 0.5, cellSize - 0.5);
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
