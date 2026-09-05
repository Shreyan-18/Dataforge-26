# The Memory Duel: KV Cache vs. BDH Synaptic Plasticity
### DataForge 2026: Pathway Track | NeurIPS 2026 Education Track Submission

> **One-Sentence Falsifiable Claim:**  
> *"A fixed-shape synaptic weight matrix processes sequences of unbounded duration without increasing memory footprint O(d²), but trades exact token-for-token recall for associative interference blur when fact density exceeds network rank capacity."*

---

## 🚀 Live Interactive Artifact
* **Web Demonstration:** Open `index.html` directly in any web browser (or deploy to GitHub Pages with 0 configuration).
* **Instant Start:** Zero dependencies, pure client-side vanilla JavaScript/HTML5 Canvas with sub-millisecond reactive computation.

---

## 🎯 Educational Profile
* **Target Audience:** Graduate students, data scientists, machine learning engineers, and NLP researchers seeking to understand the post-Transformer memory landscape.
* **Prerequisites:** Basic linear algebra (vectors, matrices, outer products, dot products), fundamental familiarity with Transformer attention.
* **Learning Objectives:**
  1. Understand why standard Key–Value (KV) caching causes an unavoidable memory explosion $\mathcal{O}(T \cdot d)$ on long-context tasks.
  2. Discover how **Dragon Hatchling (BDH)** reformulates attention into biological synaptic plasticity using Hebbian associative writes $W_t = \lambda W_{t-1} + v_t \otimes k_t^\top$.
  3. Test the fundamental trade-off: unbounded sequence processing in constant $\mathcal{O}(d^2)$ memory versus associative interference blur when fact density exceeds matrix capacity.
  4. Master how Hebbian retention decay ($\lambda$) mitigates crosstalk noise to restore sharp recall for recent associations.

---

## ⚡ 60-Second Guided Tour
1. **The Infinite Novel Test (Length Scaling):** Click the **"1. The Infinite Novel"** preset button. Drag sequence length from 100 to 100,000 tokens. Observe how the Transformer memory gauge skyrockets from 50 KB to >25 GB (triggering Out of Memory warnings), while BDH's memory footprint remains locked at a constant 16 KB.
2. **The Crammed Exam Test (Interference Breakdown):** Click the **"2. The Crammed Exam"** preset. Increase the number of packed facts to 75. Observe the live Synaptic Heatmap saturate with crosstalk, and watch retrieval fidelity transition from sharp recall ($>0.85$) into interference blur ($\approx 0.35$).
3. **Synaptic Hygiene (Decay Calibration):** Click the **"3. Synaptic Hygiene"** preset. Tune the retention factor $\lambda$ down to 0.98. Observe older distracting facts fade out on the heatmap, immediately restoring sharp recall for recent queries.

---

## 🔬 Mathematical Substrate: Real Client-Side Computation

Unlike animated infographics or static slide decks, every control in this explainer directly modifies live mathematical variables:

### 1. Transformer Key-Value Caching
* **State:** Unbounded buffer of key-value vectors:
  $$K_{1:T} \in \mathbb{R}^{T \times d}, \quad V_{1:T} \in \mathbb{R}^{T \times d}$$
* **Memory Complexity:** $\mathcal{O}(T \cdot d)$
* **Exact Attention Retrieval:**
  $$y = \text{Softmax}\left(\frac{q K^\top}{\sqrt{d}}\right) V$$

### 2. Dragon Hatchling (BDH) Synaptic Memory
* **State:** Fixed-dimension synaptic connection matrix:
  $$W_t \in \mathbb{R}^{d \times d}$$
* **Memory Complexity:** $\mathcal{O}(d^2)$ (Constant across all $T$)
* **Hebbian Plasticity Write Rule:**
  $$W_t = \lambda W_{t-1} + \eta (v_t \otimes k_t^\top)$$
* **Linear Associative Read Rule:**
  $$y = W_t \, q$$

---

## 📦 Project Architecture
```
memory-duel-bdh/
├── index.html            # Standalone interactive web playground
├── styles.css            # Dark-theme responsive UI styling
├── app.js                # Real-time matrix computation & live canvas rendering engine
├── simulator.py          # Standalone Python CLI verification script & benchmarks
├── generate_summary.py   # ReportLab script generating the official 1-page PDF
├── concept_summary.pdf   # The generated authoritative 1-page concept briefing
└── README.md             # Complete submission documentation & disclosures
```

---

## 🛠️ Reproduction & Local Execution

### Quick Start (Web App)
Simply double-click `index.html` or open it in any modern browser:
```bash
# Optional: Launch via local HTTP server
python -m http.server 8000
# Then visit: http://localhost:8000
```

### Run Python Verification Benchmarks
```bash
python simulator.py
```
Expected output confirms linear $\mathcal{O}(T)$ growth for KV Cache vs. strictly fixed 16.0 KB for BDH:
```
================================================================================
THE MEMORY DUEL: KV CACHE vs. BDH SYNAPTIC MEMORY BENCHMARK
Dimension: 64 | Key-Value Facts Stored: 25
================================================================================
Total Seq Len   | KV Memory      | BDH Memory     | KV Recall  | BDH Recall
--------------------------------------------------------------------------------
100             |      50.0 KB   |      16.0 KB   |    0.355   |    0.864
1000            |     500.0 KB   |      16.0 KB   |    0.236   |    0.853
10000           |    5000.0 KB   |      16.0 KB   |    0.225   |    0.377
50000           |   25000.0 KB   |      16.0 KB   |    0.110   |   -0.073
================================================================================
```

### Regenerate 1-Page Concept Summary PDF
```bash
python generate_summary.py
```

---

## 📚 Primary Scientific Citations (2022–2026)
1. **Pathway Research (2025/2026)** — *The Dragon Hatchling (BDH) Architecture: Biological Synaptic Plasticity as Post-Transformer Session Memory*. Technical Report.
2. **Schlag, I., Irie, K., & Schmidhuber, J.** — *Linear Transformers Are Secretly Fast Weight Programmers*. Advances in Neural Information Processing Systems (NeurIPS).
3. **Sun, Y., Dong, L., Huang, S., et al. (2023)** — *Retentive Network: A Successor to Transformer for Large Language Models*. arXiv:2307.08621.

---

## ⚖️ Disclosures, Provenance & License
* **Code & Architecture:** Original implementation built for DataForge 2026.
* **Component Classification:** 
  - Real Compute: Matrix operations, dot products, cosines, and Hebbian updates run live in `app.js` and `simulator.py`.
  - Scaled Display: Memory gauges depict real-world 8B parameter GPU allocations ($d=4096, L=32$) alongside the active $d=32/64$ mathematical simulation.
* **AI Assistance:** Developed with AI pair-programming assistance for accelerated development, with all equations, code paths, and claims verified.
* **License:** MIT License.
