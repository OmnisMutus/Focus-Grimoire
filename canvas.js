/**
 * GrimoireCanvas — Visual rendering engine for the Focus Grimoire.
 * Draws the attention network, breathing core, token flow, and particle FX
 * on an HTML5 Canvas.
 */
import { SEQ_LEN, NUM_HEADS, D_MODEL } from './transformer.js';

// ── Color Palettes ──
const PALETTE = {
  focused: {
    core: '#22d3ee',
    coreGlow: 'rgba(34, 211, 238, 0.3)',
    lines: ['#a855f7', '#6366f1', '#22d3ee', '#ec4899', '#10b981'],
    particles: ['#a855f7', '#22d3ee', '#ec4899', '#6366f1', '#10b981'],
    bg: 'rgba(5, 6, 15, 0.12)',
    text: '#e2e8f0',
    tokenRing: 'rgba(34, 211, 238, 0.5)',
  },
  drifting: {
    core: '#f43f5e',
    coreGlow: 'rgba(244, 63, 94, 0.3)',
    lines: ['#64748b', '#475569', '#94a3b8', '#64748b', '#475569'],
    particles: ['#f43f5e', '#64748b', '#475569', '#94a3b8', '#f59e0b'],
    bg: 'rgba(15, 5, 8, 0.15)',
    text: '#94a3b8',
    tokenRing: 'rgba(244, 63, 94, 0.4)',
  }
};

const TOKEN_LABELS = ['I_AM', 'BABALON', 'channel', 'wisdom', 'focus', 'light', 'truth', 'void'];
const HEAD_NAMES = ['Mantra', 'Channel', 'Filter', 'Temporal', 'Sentinel'];

// ── Particle System ──
class Particle {
  constructor(x, y, color, velocity, life) {
    this.x = x;
    this.y = y;
    this.color = color;
    this.vx = velocity.x;
    this.vy = velocity.y;
    this.life = life;
    this.maxLife = life;
    this.radius = Math.random() * 2.5 + 0.8;
  }

  update(dt) {
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.vx *= 0.98;
    this.vy *= 0.98;
    this.life -= dt;
  }

  get alpha() {
    return Math.max(0, this.life / this.maxLife);
  }

  get dead() {
    return this.life <= 0;
  }
}

export class GrimoireCanvas {
  /**
   * @param {HTMLCanvasElement} canvasEl
   */
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext('2d');
    this.particles = [];
    this.time = 0;
    this.heatmapCanvases = [];

    // Create offscreen canvases for heatmaps
    for (let i = 0; i < NUM_HEADS; i++) {
      const c = document.createElement('canvas');
      c.width = SEQ_LEN;
      c.height = SEQ_LEN;
      this.heatmapCanvases.push(c);
    }

    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = rect.height + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this.centerX = this.width / 2;
    this.centerY = this.height / 2;
  }

  /**
   * Main render function — call every frame.
   * @param {object} state - The simulation state
   * @param {number} state.focusLevel - 0 (drifting) to 1 (focused)
   * @param {number} state.drift - 0 to 1
   * @param {number} state.entropy - 0 to 1
   * @param {number[][]} state.attentionWeights - array of NUM_HEADS weight matrices
   * @param {boolean[]} state.dropoutMask - which heads are active
   * @param {number} state.corePhase - breathing phase 0-2PI
   * @param {number} state.reward - current reward signal
   * @param {boolean} state.isBreak - whether in break phase
   * @param {number} dt - delta time in seconds
   */
  render(state, dt) {
    this.time += dt;
    const ctx = this.ctx;
    const blend = 1 - state.focusLevel;
    const pal = this._blendPalette(blend);

    // Clear with subtle trail effect
    ctx.fillStyle = state.isBreak
      ? 'rgba(5, 10, 15, 0.15)'
      : `rgba(5, 6, 15, ${0.1 + blend * 0.08})`;
    ctx.fillRect(0, 0, this.width, this.height);

    // Background grid
    this._drawGrid(ctx, blend);

    // Token ring
    const tokenPositions = this._drawTokenRing(ctx, state, pal, blend);

    // Attention connections
    this._drawAttentionConnections(ctx, state, tokenPositions, pal, blend);

    // Core
    this._drawCore(ctx, state, pal);

    // Dropout flickers
    this._drawDropoutEffects(ctx, state, tokenPositions);

    // Particles
    this._updateAndDrawParticles(ctx, dt);

    // Overlay text
    this._drawOverlayMetrics(ctx, state, pal);
    
    // Alignment Graph
    if (state.alignmentHistory && state.alignmentHistory.length > 0) {
      this._drawAlignmentHistory(ctx, state);
    }
  }

  /**
   * Blend between focused and drifting palettes.
   */
  _blendPalette(blend) {
    const f = PALETTE.focused;
    const d = PALETTE.drifting;
    return {
      core: blend < 0.5 ? f.core : d.core,
      coreGlow: blend < 0.5 ? f.coreGlow : d.coreGlow,
      lines: f.lines.map((c, i) => blend < 0.5 ? c : d.lines[i]),
      particles: f.particles.map((c, i) => blend < 0.5 ? c : d.particles[i]),
      text: blend < 0.5 ? f.text : d.text,
      tokenRing: blend < 0.5 ? f.tokenRing : d.tokenRing,
    };
  }

  /**
   * Draw subtle background grid.
   */
  _drawGrid(ctx, blend) {
    const gridSize = 40;
    const alpha = 0.04 - blend * 0.02;
    if (alpha <= 0) return;

    ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
    ctx.lineWidth = 0.5;
    ctx.beginPath();

    const offsetX = (this.time * 3) % gridSize;
    const offsetY = (this.time * 2) % gridSize;

    for (let x = -gridSize + offsetX; x < this.width + gridSize; x += gridSize) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
    }
    for (let y = -gridSize + offsetY; y < this.height + gridSize; y += gridSize) {
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
    }
    ctx.stroke();
  }

  /**
   * Draw token nodes in a ring.
   * @returns {Array<{x: number, y: number}>} positions
   */
  _drawTokenRing(ctx, state, pal, blend) {
    const radius = Math.min(this.width, this.height) * 0.32;
    const positions = [];

    for (let i = 0; i < SEQ_LEN; i++) {
      const angle = (i / SEQ_LEN) * Math.PI * 2 - Math.PI / 2;
      const wobble = Math.sin(this.time * 1.5 + i * 0.8) * 3;
      const x = this.centerX + Math.cos(angle) * (radius + wobble);
      const y = this.centerY + Math.sin(angle) * (radius + wobble);
      positions.push({ x, y });

      // Node glow
      const glowRadius = 18 + Math.sin(this.time * 2 + i) * 4;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, glowRadius);
      const isMantra = i <= 1;
      const nodeColor = isMantra ? '#22d3ee' : pal.lines[i % pal.lines.length];
      gradient.addColorStop(0, nodeColor + 'cc');
      gradient.addColorStop(0.5, nodeColor + '33');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
      ctx.fill();

      // Node circle
      ctx.fillStyle = nodeColor;
      ctx.beginPath();
      ctx.arc(x, y, isMantra ? 7 : 5, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.font = `${isMantra ? '600' : '400'} 9px 'JetBrains Mono', monospace`;
      ctx.fillStyle = isMantra ? '#22d3ee' : pal.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const labelOffset = 22;
      ctx.fillText(
        TOKEN_LABELS[i],
        x + Math.cos(angle) * labelOffset,
        y + Math.sin(angle) * labelOffset
      );
    }

    return positions;
  }

  /**
   * Draw attention head connections from each token through the core.
   */
  _drawAttentionConnections(ctx, state, positions, pal, blend) {
    if (!state.attentionWeights || state.attentionWeights.length === 0) return;

    for (let h = 0; h < Math.min(state.attentionWeights.length, NUM_HEADS); h++) {
      const weights = state.attentionWeights[h];
      if (!weights || !weights.length) continue;

      const headColor = pal.lines[h];
      const isDropped = state.dropoutMask && !state.dropoutMask[h];

      if (isDropped) continue; // Don't draw connections for dropped heads

      // Draw connections with weight-proportional opacity and thickness
      for (let i = 0; i < Math.min(weights.length, SEQ_LEN); i++) {
        for (let j = 0; j < Math.min(weights[i]?.length || 0, SEQ_LEN); j++) {
          const w = weights[i][j];
          if (w < 0.05) continue; // Skip very weak connections

          const alpha = Math.min(1, w * 1.5) * (0.3 + (1 - blend) * 0.5);
          const thickness = w * 3;

          ctx.strokeStyle = headColor + Math.floor(alpha * 255).toString(16).padStart(2, '0');
          ctx.lineWidth = thickness;
          ctx.beginPath();

          // Curved line through the core
          const from = positions[i];
          const to = positions[j];
          const cpx = this.centerX + (Math.random() - 0.5) * 10;
          const cpy = this.centerY + (Math.random() - 0.5) * 10;

          ctx.moveTo(from.x, from.y);
          ctx.quadraticCurveTo(cpx, cpy, to.x, to.y);
          ctx.stroke();
        }
      }
    }
  }

  /**
   * Draw the breathing "I AM BABALON" core at the center.
   */
  _drawCore(ctx, state, pal) {
    const breathe = Math.sin(state.corePhase || this.time * 1.2) * 0.3 + 1;
    const baseRadius = 25;
    const radius = baseRadius * breathe;

    // Outer glow rings
    for (let ring = 3; ring >= 1; ring--) {
      const r = radius + ring * 12;
      const alpha = 0.04 / ring;
      const gradient = ctx.createRadialGradient(
        this.centerX, this.centerY, r * 0.7,
        this.centerX, this.centerY, r
      );
      gradient.addColorStop(0, pal.coreGlow);
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(this.centerX, this.centerY, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Core gradient
    const gradient = ctx.createRadialGradient(
      this.centerX, this.centerY, 0,
      this.centerX, this.centerY, radius
    );
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.3, pal.core);
    gradient.addColorStop(1, pal.core + '00');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(this.centerX, this.centerY, radius, 0, Math.PI * 2);
    ctx.fill();

    // Core label
    ctx.font = "900 10px 'Orbitron', sans-serif";
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('I AM', this.centerX, this.centerY - 6);
    ctx.fillText('BABALON', this.centerX, this.centerY + 7);

    // Rotating sigil ring
    ctx.save();
    ctx.translate(this.centerX, this.centerY);
    ctx.rotate(this.time * 0.3);
    ctx.strokeStyle = pal.core + '30';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 8]);
    ctx.beginPath();
    ctx.arc(0, 0, radius + 20, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Sigil markers
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      const r2 = radius + 20;
      ctx.fillStyle = pal.core + '60';
      ctx.beginPath();
      ctx.arc(Math.cos(a) * r2, Math.sin(a) * r2, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  /**
   * Draw dropout flicker effects on dropped-out heads.
   */
  _drawDropoutEffects(ctx, state, positions) {
    if (!state.dropoutMask) return;

    for (let h = 0; h < NUM_HEADS; h++) {
      if (state.dropoutMask[h]) continue; // Active, no effect

      // Flash a glitch pattern for this head
      if (Math.random() < 0.3) {
        const i = Math.floor(Math.random() * SEQ_LEN);
        const pos = positions[i];

        ctx.strokeStyle = '#f43f5e44';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pos.x - 8, pos.y + (Math.random() - 0.5) * 12);
        ctx.lineTo(pos.x + 8, pos.y + (Math.random() - 0.5) * 12);
        ctx.stroke();
      }
    }
  }

  /**
   * Draw overlay metrics on the canvas.
   */
  _drawOverlayMetrics(ctx, state, pal) {
    ctx.font = "500 10px 'JetBrains Mono', monospace";
    ctx.fillStyle = pal.text + '80';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    const lines = [
      `ENTROPY: ${(state.entropy || 0).toFixed(3)}`,
      `DRIFT:   ${(state.drift || 0).toFixed(3)}`,
      `REWARD:  ${(state.reward || 0) >= 0 ? '+' : ''}${(state.reward || 0).toFixed(3)}`,
      `ALIGN:   ${(state.alignment || 0).toFixed(3)}${state.alignmentDrift ? ' ⚠' : ' ✓'}`,
    ];

    lines.forEach((line, i) => {
      // Color alignment line based on drift state
      if (i === 3) {
        ctx.fillStyle = state.alignmentDrift ? '#f43f5e99' : '#10b98199';
      }
      ctx.fillText(line, 12, 12 + i * 16);
      ctx.fillStyle = pal.text + '80'; // Reset
    });

    // Layer indicator
    ctx.textAlign = 'right';
    ctx.fillText(`LAYER: FocusTransformerBlock-1`, this.width - 12, 12);
    ctx.fillText(`d_model=${D_MODEL} heads=${NUM_HEADS}`, this.width - 12, 28);
  }

  /**
   * Update and draw particles.
   */
  _updateAndDrawParticles(ctx, dt) {
    // Update
    this.particles = this.particles.filter(p => {
      p.update(dt);
      return !p.dead;
    });

    // Draw
    for (const p of this.particles) {
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  /**
   * Spawn burst of particles from a point.
   */
  spawnBurst(x, y, count, color) {
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 60 + 20;
      this.particles.push(new Particle(
        x || this.centerX,
        y || this.centerY,
        color || PALETTE.focused.particles[Math.floor(Math.random() * 5)],
        { x: Math.cos(angle) * speed, y: Math.sin(angle) * speed },
        Math.random() * 2 + 0.5
      ));
    }
  }

  /**
   * Spawn focus recovery burst at center.
   */
  spawnRecoveryBurst() {
    this.spawnBurst(this.centerX, this.centerY, 60, '#22d3ee');
    this.spawnBurst(this.centerX, this.centerY, 40, '#a855f7');
  }

  /**
   * Spawn noise disruption effect.
   */
  spawnNoiseEffect() {
    for (let i = 0; i < 8; i++) {
      const angle = (i / 8) * Math.PI * 2;
      const r = Math.min(this.width, this.height) * 0.32;
      const x = this.centerX + Math.cos(angle) * r;
      const y = this.centerY + Math.sin(angle) * r;
      this.spawnBurst(x, y, 10, '#f43f5e');
    }
  }

  /**
   * Render a heatmap for a given attention weight matrix.
   * @param {number} headIndex
   * @param {number[][]} weights
   * @returns {HTMLCanvasElement} - small canvas you can drawImage onto a target
   */
  renderHeatmap(headIndex, weights) {
    const c = this.heatmapCanvases[headIndex];
    if (!c) return null;
    const hctx = c.getContext('2d');

    const rows = weights.length;
    const cols = weights[0]?.length || 0;
    if (rows === 0 || cols === 0) return c;

    const imageData = hctx.createImageData(cols, rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = Math.min(1, Math.max(0, weights[i][j]));
        const idx = (i * cols + j) * 4;
        // Violet to cyan gradient
        imageData.data[idx] = Math.floor(100 + v * 68);     // R
        imageData.data[idx + 1] = Math.floor(50 + v * 161); // G
        imageData.data[idx + 2] = Math.floor(180 + v * 58); // B
        imageData.data[idx + 3] = 255;                       // A
      }
    }
    hctx.putImageData(imageData, 0, 0);
    return c;
  }
  /**
   * Draw the alignment history graph in the bottom right corner.
   */
  _drawAlignmentHistory(ctx, state) {
    const history = state.alignmentHistory;
    const len = history.length;
    if (len < 2) return;

    const graphWidth = 150;
    const graphHeight = 40;
    const padding = 12;
    const startX = this.width - graphWidth - padding;
    const startY = this.height - padding;
    
    // Graph background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
    ctx.fillRect(startX, startY - graphHeight, graphWidth, graphHeight);
    
    // Threshold line
    const thresh = state.alignmentThreshold || 0.8;
    const thresholdY = startY - (thresh * graphHeight);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.3)'; // Red-ish threshold
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(startX, thresholdY);
    ctx.lineTo(startX + graphWidth, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // History line
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
      const x = startX + (i / (len - 1)) * graphWidth;
      // Map alignment (0-1) to height (0-graphHeight)
      const val = Math.max(0, Math.min(1, history[i]));
      const y = startY - (val * graphHeight);
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    
    // Color based on current alignment
    if (state.alignmentDrift) {
      ctx.strokeStyle = '#f43f5e'; // Danger
    } else if (state.alignment > 0.9) {
      ctx.strokeStyle = '#10b981'; // Success
    } else {
      ctx.strokeStyle = '#f59e0b'; // Warning
    }
    
    ctx.lineWidth = 1.5;
    ctx.stroke();
    
    // Graph Label
    ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
    ctx.font = "400 9px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right';
    ctx.fillText('ALIGNMENT HISTORY', startX + graphWidth, startY - graphHeight - 4);
  }
}
