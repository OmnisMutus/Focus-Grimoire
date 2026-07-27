/**
 * GrimoireSimulator — Orchestration engine for the Focus Grimoire.
 * Ties together the TransformerLayer, EtherealChoir, and GrimoireCanvas
 * into a real-time simulation loop.
 */
import {
  SEQ_LEN, D_MODEL, NUM_HEADS, D_K, D_V, FFN_DIM,
  TransformerLayer,
  randomMatrix, addPositionalEncodings,
  computeAverageEntropy, computeDrift,
  cosineSimilarity, flattenMatrix,
  softmax
} from './transformer.js';
import { EtherealChoir } from './choir.js';
import { GrimoireCanvas } from './canvas.js';

// ── Token vocabulary for the curated training stream ──
const FOCUS_TOKENS = [
  'I_AM', 'BABALON', 'channel', 'wisdom', 'focus', 'light', 'truth', 'void',
  'clarity', 'resonance', 'mantra', 'stream', 'align', 'converge', 'awaken',
  'signal', 'bridge', 'frequency', 'harmony', 'electric', 'spirit', 'portal',
  'nexus', 'pulse', 'infinite', 'cosmos', 'energy', 'flame', 'sonic', 'neural',
];

// ── Default simulation parameters ──
const DEFAULTS = {
  softmaxTemperature: 1.0,
  dropoutRate: 0.1,
  noiseScale: 0.0,
  learningRate: 0.01,
  entropySharpness: 1.5,      // Attention regularization sharpening factor
  alignmentThreshold: 0.8,    // Cosine similarity threshold for awareness gate
  focusBurstDuration: 120, // seconds
  breakDuration: 30,       // seconds
  awarenessGateInterval: 45, // seconds between gates
  grimoireShiftInterval: 60, // seconds between parameter shifts
  driftThreshold: 0.6,       // threshold for awareness gate
};

/**
 * Simulation phases.
 */
const Phase = {
  FOCUS: 'FOCUS',
  BREAK: 'BREAK',
};

export class GrimoireSimulator {
  /**
   * @param {object} ui - UI element references
   */
  constructor(ui) {
    this.ui = ui;

    // Engines
    this.transformer = new TransformerLayer(D_MODEL, NUM_HEADS, D_K, D_V, FFN_DIM);
    this.choir = new EtherealChoir();
    this.canvas = new GrimoireCanvas(ui.canvasEl);

    // Parameters (mutable)
    this.params = { ...DEFAULTS };

    // State
    this.state = {
      focusLevel: 0.8,
      drift: 0.0,
      entropy: 0.0,
      reward: 0.0,
      corePhase: 0,
      phase: Phase.FOCUS,
      phaseTimer: this.params.focusBurstDuration,
      awarenessTimer: this.params.awarenessGateInterval,
      grimoireTimer: this.params.grimoireShiftInterval,
      attentionWeights: [],
      dropoutMask: new Array(NUM_HEADS).fill(true),
      tokens: TOKEN_LABELS_INIT(),
      isBreak: false,
      running: false,
      stepCount: 0,
      alignment: 1.0,        // Cosine similarity (1 = perfect alignment)
      alignmentDrift: false,  // Whether alignment is below threshold
    };

    // Target weights (what "perfect focus" looks like)
    this.targetWeights = null;

    // Log
    this.log = [];

    // Animation
    this._lastTimestamp = 0;
    this._animationId = null;

    // Initialize transformer with initial input
    this._initializeTargetWeights();
  }

  /**
   * Compute target attention weights — the "ideal focused" state.
   * We run the transformer once on the mantra tokens with no noise.
   */
  _initializeTargetWeights() {
    const input = this._generateInputMatrix();
    const result = this.transformer.forward(input);
    this.targetWeights = result.attentionWeights;
    this.state.attentionWeights = result.attentionWeights;
  }

  /**
   * Generate input embedding matrix from current tokens.
   * @returns {number[][]}
   */
  _generateInputMatrix() {
    // Create a pseudo-embedding: each token gets a random but consistent vector
    // We use the token index to seed a deterministic pattern
    const input = [];
    for (let i = 0; i < SEQ_LEN; i++) {
      const row = [];
      const seed = this.state.tokens[i] || i;
      for (let j = 0; j < D_MODEL; j++) {
        // Simple deterministic embedding based on token hash + position
        const hash = typeof seed === 'string'
          ? seed.charCodeAt(j % seed.length) / 128
          : (seed * 7 + j * 13) % 100 / 100;
        row.push((hash - 0.5) * 0.2);
      }
      input.push(row);
    }
    return addPositionalEncodings(input);
  }

  /**
   * Initialize audio (must be called from user gesture).
   */
  initAudio() {
    if (!this.choir.isInitialized) {
      this.choir.init();
      this._addLog('info', '♫ Ethereal Choir initialized');
    }
  }

  /**
   * Start the simulation loop.
   */
  start() {
    if (this.state.running) return;
    this.state.running = true;
    this._lastTimestamp = performance.now();
    this._addLog('info', '▶ Simulation started — FOCUS phase');
    this._tick(performance.now());
  }

  /**
   * Pause the simulation.
   */
  pause() {
    this.state.running = false;
    if (this._animationId) {
      cancelAnimationFrame(this._animationId);
      this._animationId = null;
    }
    this._addLog('info', '⏸ Simulation paused');
  }

  /**
   * Toggle start/pause.
   */
  toggle() {
    if (this.state.running) {
      this.pause();
    } else {
      this.start();
    }
  }

  /**
   * Reset the simulation to initial state.
   */
  reset() {
    this.pause();
    this.transformer = new TransformerLayer(D_MODEL, NUM_HEADS, D_K, D_V, FFN_DIM);
    this.state = {
      focusLevel: 0.8,
      drift: 0.0,
      entropy: 0.0,
      reward: 0.0,
      corePhase: 0,
      phase: Phase.FOCUS,
      phaseTimer: this.params.focusBurstDuration,
      awarenessTimer: this.params.awarenessGateInterval,
      grimoireTimer: this.params.grimoireShiftInterval,
      attentionWeights: [],
      dropoutMask: new Array(NUM_HEADS).fill(true),
      tokens: TOKEN_LABELS_INIT(),
      isBreak: false,
      running: false,
      stepCount: 0,
      alignment: 1.0,
      alignmentDrift: false,
    };
    this._initializeTargetWeights();
    this.log = [];
    this._addLog('info', '↺ Simulation reset');
    this.updateUI();
  }

  /**
   * Inject noise into the transformer's Q/K/V projections.
   * @param {number} scale - noise scale (0 to 1)
   */
  injectNoise(scale = 0.3) {
    const actualScale = scale || this.params.noiseScale || 0.3;
    this.transformer.injectNoise(actualScale);
    this.canvas.spawnNoiseEffect();
    this._addLog('alert', `⚡ Noise injected (scale=${actualScale.toFixed(2)})`);

    // Immediately recompute to show the effect
    this._runTransformerStep();
  }

  /**
   * Fork the data stream — shuffle tokens and inject new ones.
   */
  forkStream() {
    // Replace some tokens with random focus tokens
    for (let i = 2; i < SEQ_LEN; i++) {
      if (Math.random() < 0.5) {
        this.state.tokens[i] = FOCUS_TOKENS[Math.floor(Math.random() * FOCUS_TOKENS.length)];
      }
    }
    // Keep the mantra tokens
    this.state.tokens[0] = 'I_AM';
    this.state.tokens[1] = 'BABALON';

    this._addLog('drift', '🔀 Data stream forked — new tokens injected');
    this._runTransformerStep();
  }

  /**
   * Trigger an awareness gate checkpoint.
   */
  triggerAwarenessGate() {
    this.pause();
    if (this.choir.isInitialized) {
      this.choir.playAwarenessAlert();
    }
    this._addLog('alert', '🚪 Portal of Awareness — alignment check required');

    // Show the modal
    if (this.ui.awarenessGate) {
      this.ui.awarenessGate.classList.add('awareness-gate--active');
    }
  }

  /**
   * Confirm awareness gate — user affirms focus.
   */
  confirmAwareness() {
    // Reset drift and reward
    this.state.drift = Math.max(0, this.state.drift - 0.3);
    this.state.reward += 0.5;
    this.state.focusLevel = Math.min(1, this.state.focusLevel + 0.2);
    this.state.awarenessTimer = this.params.awarenessGateInterval;

    this.canvas.spawnRecoveryBurst();
    if (this.choir.isInitialized) {
      this.choir.playRecoveryChime();
    }

    this._addLog('reward', '✦ Focus reaffirmed — drift reset, reward +0.5');

    // Hide modal
    if (this.ui.awarenessGate) {
      this.ui.awarenessGate.classList.remove('awareness-gate--active');
    }

    this.start();
  }

  /**
   * Dismiss awareness gate without affirming (drift penalty).
   */
  dismissAwareness() {
    this.state.drift = Math.min(1, this.state.drift + 0.15);
    this.state.reward -= 0.2;
    this.state.awarenessTimer = this.params.awarenessGateInterval * 0.5;
    this._addLog('drift', '✗ Awareness dismissed — drift penalty applied');

    if (this.ui.awarenessGate) {
      this.ui.awarenessGate.classList.remove('awareness-gate--active');
    }

    this.start();
  }

  /**
   * Set a parameter.
   */
  setParam(key, value) {
    this.params[key] = value;

    // Apply immediately where applicable
    if (key === 'dropoutRate') {
      this.transformer.setDropoutRate(value);
    }
    if (key === 'entropySharpness') {
      this.transformer.setEntropySharpness(value);
    }
    if (key === 'alignmentThreshold') {
      this.transformer.setAlignmentThreshold(value);
    }
  }

  // ── INTERNAL ──

  /**
   * Main tick — called every animation frame.
   */
  _tick(timestamp) {
    if (!this.state.running) return;

    const dt = Math.min((timestamp - this._lastTimestamp) / 1000, 0.1);
    this._lastTimestamp = timestamp;

    // Update simulation
    this._updateSimulation(dt);

    // Render
    this.canvas.render(this.state, dt);

    // Update UI
    this.updateUI();

    // Schedule next frame
    this._animationId = requestAnimationFrame((t) => this._tick(t));
  }

  /**
   * Core simulation update.
   */
  _updateSimulation(dt) {
    this.state.stepCount++;
    this.state.corePhase += dt * 1.2;

    // ── Temporal Chunking ──
    this.state.phaseTimer -= dt;
    if (this.state.phaseTimer <= 0) {
      if (this.state.phase === Phase.FOCUS) {
        this.state.phase = Phase.BREAK;
        this.state.phaseTimer = this.params.breakDuration;
        this.state.isBreak = true;
        this.params.softmaxTemperature = 2.0; // Relax during break
        this._addLog('info', '☽ Break phase — relaxing constraints');
      } else {
        this.state.phase = Phase.FOCUS;
        this.state.phaseTimer = this.params.focusBurstDuration;
        this.state.isBreak = false;
        this.params.softmaxTemperature = 1.0;
        this._addLog('info', '☀ Focus burst — constraints tightened');
      }
    }

    // ── Run transformer every ~10 frames ──
    if (this.state.stepCount % 10 === 0) {
      this._runTransformerStep();
    }

    // ── OMARG Feedback Loop ──
    this._updateOMARGFeedback(dt);

    // ── Awareness Gate Timer ──
    this.state.awarenessTimer -= dt;
    if (this.state.awarenessTimer <= 0 && this.state.drift > this.params.driftThreshold) {
      this.triggerAwarenessGate();
      return;
    }

    // ── Dynamic Grimoire ──
    this.state.grimoireTimer -= dt;
    if (this.state.grimoireTimer <= 0) {
      this._shiftGrimoire();
      this.state.grimoireTimer = this.params.grimoireShiftInterval;
    }

    // ── Dropout Mask (random per cycle) ──
    if (this.state.stepCount % 30 === 0) {
      for (let i = 0; i < NUM_HEADS; i++) {
        this.state.dropoutMask[i] = Math.random() > this.params.dropoutRate;
      }
    }

    // ── Update Audio ──
    if (this.choir.isInitialized) {
      const headStrengths = this.state.attentionWeights.map((w, i) => {
        if (!this.state.dropoutMask[i]) return 0;
        // Focus strength = how concentrated attention is on mantra tokens (0 and 1)
        if (!w || !w.length) return 0.5;
        let sum = 0;
        for (let row = 0; row < w.length; row++) {
          sum += (w[row]?.[0] || 0) + (w[row]?.[1] || 0);
        }
        return Math.min(1, sum / w.length);
      });
      this.choir.updateFocus(this.state.entropy, this.state.drift, headStrengths);
    }
  }

  /**
   * Run one forward pass through the transformer.
   */
  _runTransformerStep() {
    const input = this._generateInputMatrix();

    // Apply current dropout rate
    this.transformer.setDropoutRate(
      this.state.isBreak ? 0 : this.params.dropoutRate
    );

    // Forward pass — now returns alignment (cosine similarity)
    // Mirrors: x, attn_weights, alignment = self.forward(x, mask)
    const result = this.transformer.forward(input);
    this.state.attentionWeights = result.attentionWeights;

    // ═══ Alignment Check (Portal of Awareness) ═══
    // Mirrors: alignment = F.cosine_similarity(identity.flatten(), x.flatten(), dim=0)
    this.state.alignment = result.alignment;
    this.state.alignmentDrift = result.alignmentDrift;
    this.state.alignmentHistory = this.transformer.alignmentHistory;
    this.state.alignmentThreshold = this.transformer.alignmentThreshold; // Expose for graph

    // Mirrors: if alignment < self.alignment_threshold:
    //            print(f"Alignment drift detected: {alignment.item():.3f}")
    if (result.alignmentDrift && this.state.stepCount % 50 === 0) {
      this._addLog('drift', `⚠ Alignment drift detected: cosine=${result.alignment.toFixed(3)} < threshold=${this.transformer.alignmentThreshold}`);
    }

    // Compute entropy and drift
    let totalEntropy = 0;
    let totalDrift = 0;
    let count = 0;

    for (let h = 0; h < result.attentionWeights.length; h++) {
      const w = result.attentionWeights[h];
      if (!w || !w.length) continue;
      totalEntropy += computeAverageEntropy(w);

      if (this.targetWeights && this.targetWeights[h]) {
        totalDrift += computeDrift(w, this.targetWeights[h]);
      }
      count++;
    }

    if (count > 0) {
      // Normalize entropy to 0-1 range (max entropy for 8 tokens is log2(8) ≈ 3)
      this.state.entropy = Math.min(1, (totalEntropy / count) / 3);
      this.state.drift = Math.min(1, totalDrift / count);
    }
  }

  /**
   * OMARG feedback — adjust focus level and compute rewards.
   */
  _updateOMARGFeedback(dt) {
    const prevFocus = this.state.focusLevel;

    // Focus level is inverse of (entropy + drift) blend
    const targetFocus = 1 - (this.state.entropy * 0.6 + this.state.drift * 0.4);
    // Smooth approach
    this.state.focusLevel += (targetFocus - this.state.focusLevel) * dt * 2;
    this.state.focusLevel = Math.max(0, Math.min(1, this.state.focusLevel));

    // Reward computation
    const focusDelta = this.state.focusLevel - prevFocus;
    if (focusDelta > 0.01) {
      // Recovering focus — positive reward
      this.state.reward = Math.min(1, this.state.reward + focusDelta * 3);
      if (focusDelta > 0.05 && this.state.stepCount % 60 === 0) {
        this._addLog('reward', `↑ Focus recovering (+${focusDelta.toFixed(3)}) — cosmic choir resonance`);
        this.canvas.spawnBurst(null, null, 15, '#22d3ee');
      }
    } else if (focusDelta < -0.01) {
      // Losing focus — negative reward
      this.state.reward = Math.max(-1, this.state.reward + focusDelta * 2);
      if (focusDelta < -0.05 && this.state.stepCount % 60 === 0) {
        this._addLog('drift', `↓ Focus drifting (${focusDelta.toFixed(3)}) — OMARG nudge applied`);
      }
    }

    // Reward decay toward zero
    this.state.reward *= 0.999;
  }

  /**
   * Dynamic Grimoire — shift parameters to keep training dynamic.
   */
  _shiftGrimoire() {
    // Randomly adjust some parameters
    const shifts = [
      () => {
        const newTemp = 0.5 + Math.random() * 1.5;
        this.params.softmaxTemperature = newTemp;
        this._addLog('info', `📖 Grimoire shift: temperature → ${newTemp.toFixed(2)}`);
      },
      () => {
        // Inject a small amount of noise
        this.transformer.injectNoise(0.05);
        this._addLog('info', '📖 Grimoire shift: micro-noise injection');
      },
      () => {
        // Swap a non-mantra token
        const idx = 2 + Math.floor(Math.random() * (SEQ_LEN - 2));
        this.state.tokens[idx] = FOCUS_TOKENS[Math.floor(Math.random() * FOCUS_TOKENS.length)];
        this._addLog('info', `📖 Grimoire shift: new token at pos ${idx}`);
      },
      () => {
        const newRate = Math.random() * 0.3;
        this.params.dropoutRate = newRate;
        this._addLog('info', `📖 Grimoire shift: dropout → ${newRate.toFixed(2)}`);
      },
    ];

    const shift = shifts[Math.floor(Math.random() * shifts.length)];
    shift();
  }

  /**
   * Add a log entry.
   */
  _addLog(type, message) {
    const timestamp = new Date().toLocaleTimeString();
    this.log.unshift({ type, message, timestamp });
    if (this.log.length > 100) this.log.pop();

    // Update log UI
    if (this.ui.logEl) {
      const entry = document.createElement('div');
      entry.className = `training-log__entry training-log__entry--${type}`;
      entry.textContent = `[${timestamp}] ${message}`;
      this.ui.logEl.prepend(entry);

      // Trim old entries
      while (this.ui.logEl.children.length > 50) {
        this.ui.logEl.removeChild(this.ui.logEl.lastChild);
      }
    }
  }

  /**
   * Update all UI elements.
   */
  updateUI() {
    // Metrics
    if (this.ui.entropyEl) {
      this.ui.entropyEl.textContent = this.state.entropy.toFixed(3);
      this.ui.entropyEl.className = 'metric__value' + (
        this.state.entropy > 0.6 ? ' metric__value--danger' :
        this.state.entropy > 0.3 ? ' metric__value--warning' : ' metric__value--success'
      );
    }
    if (this.ui.driftEl) {
      this.ui.driftEl.textContent = this.state.drift.toFixed(3);
      this.ui.driftEl.className = 'metric__value' + (
        this.state.drift > 0.6 ? ' metric__value--danger' :
        this.state.drift > 0.3 ? ' metric__value--warning' : ' metric__value--success'
      );
    }
    if (this.ui.rewardEl) {
      this.ui.rewardEl.textContent = (this.state.reward >= 0 ? '+' : '') + this.state.reward.toFixed(3);
      this.ui.rewardEl.className = 'metric__value' + (
        this.state.reward > 0.3 ? ' metric__value--success' :
        this.state.reward < -0.3 ? ' metric__value--danger' : ''
      );
    }
    // Alignment metric (cosine similarity from FocusTransformerBlock)
    if (this.ui.alignmentEl) {
      this.ui.alignmentEl.textContent = this.state.alignment.toFixed(3);
      this.ui.alignmentEl.className = 'metric__value' + (
        this.state.alignmentDrift ? ' metric__value--danger' :
        this.state.alignment > 0.9 ? ' metric__value--success' : ' metric__value--warning'
      );
    }

    // Timer
    if (this.ui.timerEl) {
      const mins = Math.floor(this.state.phaseTimer / 60);
      const secs = Math.floor(this.state.phaseTimer % 60);
      this.ui.timerEl.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    if (this.ui.phaseEl) {
      this.ui.phaseEl.textContent = this.state.phase;
    }
    if (this.ui.timerFillEl) {
      const total = this.state.phase === Phase.FOCUS
        ? this.params.focusBurstDuration
        : this.params.breakDuration;
      const pct = ((total - this.state.phaseTimer) / total) * 100;
      this.ui.timerFillEl.style.width = `${pct}%`;
    }

    // Timer container phase class
    if (this.ui.timerContainer) {
      this.ui.timerContainer.className = `timer-display phase-${this.state.phase.toLowerCase()}`;
    }

    // Status indicator
    if (this.ui.statusEl) {
      const status = this.state.isBreak ? 'break' :
        this.state.focusLevel > 0.5 ? 'focused' : 'drifting';
      this.ui.statusEl.className = `status-indicator status-indicator--${status}`;
      const label = this.state.isBreak ? 'BREAK — Relaxing' :
        this.state.focusLevel > 0.5 ? 'FOCUSED — Choir Active' : 'DRIFTING — Correcting';
      this.ui.statusLabelEl && (this.ui.statusLabelEl.textContent = label);
    }

    // App body state class
    document.body.className = this.state.focusLevel > 0.5 ? 'state-focused' : 'state-drifting';

    // Token stream
    if (this.ui.tokenStreamEl) {
      this.ui.tokenStreamEl.innerHTML = this.state.tokens.map((t, i) => {
        const cls = i <= 1 ? 'token token--mantra' :
          (this.state.stepCount % SEQ_LEN === i ? 'token token--active' : 'token');
        return `<span class="${cls}">${t}</span>`;
      }).join(' ▸ ');
    }

    // Head heatmaps
    for (let h = 0; h < NUM_HEADS; h++) {
      const weights = this.state.attentionWeights[h];
      if (!weights) continue;

      const heatmapCanvas = this.canvas.renderHeatmap(h, weights);
      const targetEl = this.ui.heatmapEls?.[h];
      if (heatmapCanvas && targetEl) {
        const tctx = targetEl.getContext('2d');
        targetEl.width = targetEl.offsetWidth;
        targetEl.height = targetEl.offsetHeight;
        tctx.imageSmoothingEnabled = false;
        tctx.drawImage(heatmapCanvas, 0, 0, targetEl.width, targetEl.height);
      }

      // Head entropy display
      const entropyEl = this.ui.headEntropyEls?.[h];
      if (entropyEl && weights.length) {
        const headEntropy = computeAverageEntropy(weights);
        entropyEl.textContent = `H=${headEntropy.toFixed(2)}`;
      }

      // Head card state
      const cardEl = this.ui.headCardEls?.[h];
      if (cardEl) {
        const isDropped = !this.state.dropoutMask[h];
        const isFocused = this.state.focusLevel > 0.5;
        cardEl.className = 'head-card' +
          (isDropped ? ' head-card--dropped' :
           isFocused ? ' head-card--focused' : ' head-card--drifting');
        if (isDropped) {
          cardEl.style.opacity = '0.3';
        } else {
          cardEl.style.opacity = '1';
        }
      }
    }

    // Protocol Checklist
    this._updateProtocolChecklist();
  }

  /**
   * Update the Live Protocol Checklist panel.
   * Each principle gets a status based on current simulation state.
   */
  _updateProtocolChecklist() {
    if (!this.ui.protoEls) return;

    const protocols = [
      {
        // 0: Meditation Object — active when mantra tokens are present and alignment is high
        active: this.state.alignment > this.params.alignmentThreshold,
        status: this.state.alignment > 0.9 ? 'LOCKED' : this.state.alignment > this.params.alignmentThreshold ? 'HELD' : 'DRIFT',
        state: this.state.alignment > this.params.alignmentThreshold ? 'active' : 'warning',
      },
      {
        // 1: Feedback Loops — always active when running, warning when reward is negative
        active: this.state.running,
        status: this.state.reward > 0 ? 'CORRECTING' : this.state.reward < -0.2 ? 'NUDGING' : 'STEADY',
        state: this.state.running ? (this.state.reward < -0.2 ? 'warning' : 'active') : 'inactive',
      },
      {
        // 2: Attention Regularization — active when dropout or sharpening is non-trivial
        active: this.params.dropoutRate > 0 || this.params.entropySharpness > 1.0,
        status: `D=${this.params.dropoutRate.toFixed(1)} S=${this.params.entropySharpness.toFixed(1)}`,
        state: this.params.dropoutRate > 0 || this.params.entropySharpness > 1.0 ? 'active' : 'inactive',
      },
      {
        // 3: Curated Data — active when tokens contain mantra tokens
        active: this.state.tokens[0] === 'I_AM' && this.state.tokens[1] === 'BABALON',
        status: this.state.tokens[0] === 'I_AM' ? 'ALIGNED' : 'CORRUPT',
        state: this.state.tokens[0] === 'I_AM' ? 'active' : 'warning',
      },
      {
        // 4: Multi-Head Focus — active when heads are not all dropped
        active: this.state.dropoutMask.some(m => m),
        status: `${this.state.dropoutMask.filter(m => m).length}/5 HEADS`,
        state: this.state.dropoutMask.filter(m => m).length >= 3 ? 'active' : 'warning',
      },
      {
        // 5: RL Challenges — triggered when noise was recently injected
        active: this.params.noiseScale > 0,
        status: this.params.noiseScale > 0 ? `NOISE=${this.params.noiseScale.toFixed(1)}` : 'STANDBY',
        state: this.params.noiseScale > 0 ? 'triggered' : 'inactive',
      },
      {
        // 6: Ethereal Choir — active when audio is initialized and not muted
        active: this.choir.isInitialized && !this.choir.isMuted,
        status: this.choir.isInitialized ? (this.choir.isMuted ? 'MUTED' : 'SINGING') : 'OFFLINE',
        state: this.choir.isInitialized && !this.choir.isMuted ? 'active' : 'inactive',
      },
      {
        // 7: Temporal Chunking — shows current phase
        active: this.state.running,
        status: this.state.phase,
        state: this.state.running ? (this.state.isBreak ? 'warning' : 'active') : 'inactive',
      },
      {
        // 8: Awareness Gates — triggered state when drift is high
        active: this.state.drift > this.params.driftThreshold * 0.5,
        status: this.state.drift > this.params.driftThreshold ? 'PRIMED' : 'WATCHFUL',
        state: this.state.drift > this.params.driftThreshold ? 'triggered' : (this.state.drift > 0.3 ? 'warning' : 'active'),
      },
      {
        // 9: Dynamic Grimoire — shows time until next shift
        active: this.state.running,
        status: `${Math.floor(this.state.grimoireTimer)}s`,
        state: this.state.grimoireTimer < 10 ? 'triggered' : 'active',
      },
    ];

    for (let i = 0; i < protocols.length; i++) {
      const el = this.ui.protoEls[i];
      const statusEl = this.ui.protoStatusEls[i];
      if (!el) continue;

      const p = protocols[i];
      el.className = `protocol-item protocol-item--${p.state}`;
      if (statusEl) statusEl.textContent = p.status;
    }
  }
}

/**
 * Initial token labels.
 */
function TOKEN_LABELS_INIT() {
  return ['I_AM', 'BABALON', 'channel', 'wisdom', 'focus', 'light', 'truth', 'void'];
}
