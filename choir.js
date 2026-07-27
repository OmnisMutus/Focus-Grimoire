/**
 * A synesthetic audio feedback engine using the Web Audio API to sonify AI focus state.
 * Creates an "Ethereal Choir" that adapts to the AI's entropy and drift.
 */
export class EtherealChoir {
  /**
   * Initializes the EtherealChoir state.
   * Does not create AudioContext immediately to comply with browser autoplay policies.
   */
  constructor() {
    this.initialized = false;
    this.ctx = null;
    this.masterGain = null;
    this.oscillators = [];
    this.gains = [];
    this.reverb = null;
    this.filter = null;
    this.lfo = null;
    this.lfoGain = null;
    this.volume = 1.0;
    this.isMuted = false;
    
    // Frequencies based on A=220Hz
    // FOCUSED chords (consonant): A Major Add 9 (A2, C#3, E3, A3, B3) -> 110, 138.59, 164.81, 220, 246.94
    this.focusedFrequencies = [110.0, 138.59, 164.81, 220.0, 246.94];
    
    // DRIFTING chords (dissonant): Tritones/Minor 2nds (Bb2, E3, F3, B3, C4) -> 116.54, 164.81, 174.61, 246.94, 261.63
    this.driftingFrequencies = [116.54, 164.81, 174.61, 246.94, 261.63];
  }

  /**
   * Initializes the Web Audio API context and signal chain.
   * Must be called after a user gesture.
   */
  init() {
    if (this.initialized) return;

    // Create AudioContext
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AudioContext();

    // Create master gain node
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.setValueAtTime(this.volume, this.ctx.currentTime);
    this.masterGain.connect(this.ctx.destination);

    // Create Reverb
    this.reverb = this.ctx.createConvolver();
    this.reverb.buffer = this._createReverbIR(3, 2);

    // Create Filter
    this.filter = this.ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.setValueAtTime(2000, this.ctx.currentTime);
    this.filter.Q.setValueAtTime(1, this.ctx.currentTime);

    // Create LFO for filter modulation
    this.lfo = this.ctx.createOscillator();
    this.lfo.type = 'sine';
    this.lfo.frequency.setValueAtTime(0.5, this.ctx.currentTime);
    
    this.lfoGain = this.ctx.createGain();
    this.lfoGain.gain.setValueAtTime(400, this.ctx.currentTime); // Modulation depth in cents
    
    this.lfo.connect(this.lfoGain);
    this.lfoGain.connect(this.filter.detune);
    this.lfo.start();

    // Connect effects chain: filter -> reverb -> masterGain
    this.filter.connect(this.reverb);
    this.reverb.connect(this.masterGain);

    // Create 5 oscillators (one per attention head)
    for (let i = 0; i < 5; i++) {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(this.focusedFrequencies[i], this.ctx.currentTime);
      
      // Start silent
      gain.gain.setValueAtTime(0.0001, this.ctx.currentTime);

      osc.connect(gain);
      gain.connect(this.filter);
      
      osc.start();
      
      this.oscillators.push(osc);
      this.gains.push(gain);
    }

    this.initialized = true;
  }

  /**
   * Generates a synthetic reverb impulse response.
   * @param {number} duration - Duration in seconds.
   * @param {number} decay - Decay factor.
   * @returns {AudioBuffer|null} The generated impulse response buffer.
   * @private
   */
  _createReverbIR(duration = 3, decay = 2) {
    if (!this.ctx) return null;
    const sampleRate = this.ctx.sampleRate;
    const length = sampleRate * duration;
    const impulse = this.ctx.createBuffer(2, length, sampleRate);
    const left = impulse.getChannelData(0);
    const right = impulse.getChannelData(1);

    for (let i = 0; i < length; i++) {
      // Exponential decay
      const factor = Math.exp(-i / (sampleRate / decay));
      // Random noise -1 to 1
      left[i] = (Math.random() * 2 - 1) * factor;
      right[i] = (Math.random() * 2 - 1) * factor;
    }
    return impulse;
  }

  /**
   * Main update method - call this every frame to update audio based on focus state.
   * @param {number} entropy - 0 (perfectly focused) to 1 (maximum entropy).
   * @param {number} drift - 0 (on target) to 1 (fully drifted).
   * @param {number[]} [headWeights=[]] - Array of 5 numbers (0-1) for each head.
   */
  updateFocus(entropy, drift, headWeights = []) {
    if (!this.initialized || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime + 0.3; // 0.3s ramp for smooth transitions

    // Blend factor = (entropy + drift) / 2, clamped 0-1
    let blend = (entropy + drift) / 2;
    blend = Math.max(0, Math.min(1, blend));

    for (let i = 0; i < 5; i++) {
      const osc = this.oscillators[i];
      const gain = this.gains[i];
      
      const weight = headWeights[i] !== undefined ? headWeights[i] : 0.2;

      // Base frequency interpolation
      const freqFocused = this.focusedFrequencies[i];
      const freqDrifting = this.driftingFrequencies[i];
      const targetFreq = freqFocused * (1 - blend) + freqDrifting * blend;

      // Detune up to +-30Hz when drifting
      const detuneAmount = (Math.random() * 60 - 30) * blend;
      
      osc.frequency.exponentialRampToValueAtTime(Math.max(20, targetFreq + detuneAmount), t);
      
      // Switch to 'sawtooth' when significantly drifting
      if (blend > 0.6) {
        osc.type = 'sawtooth';
      } else {
        osc.type = 'sine';
      }

      // Gain proportional to headWeights
      // Scale down overall volume per oscillator to avoid clipping
      const targetGain = weight * 0.2; 
      gain.gain.exponentialRampToValueAtTime(Math.max(0.0001, targetGain), t);
    }

    // Master filter cutoff: focused=2000Hz, drifting=400Hz
    const targetCutoff = 2000 * (1 - blend) + 400 * blend;
    this.filter.frequency.exponentialRampToValueAtTime(Math.max(10, targetCutoff), t);

    // LFO rate: focused=0.5Hz, drifting=4Hz
    const targetLfoRate = 0.5 * (1 - blend) + 4 * blend;
    this.lfo.frequency.exponentialRampToValueAtTime(Math.max(0.1, targetLfoRate), t);
  }

  /**
   * Plays a triumphant recovery chord swell.
   */
  playRecoveryChime() {
    if (!this.initialized || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    // C5, E5, G5, C6 (523.25, 659.25, 783.99, 1046.50)
    const chordFreqs = [523.25, 659.25, 783.99, 1046.50];
    
    chordFreqs.forEach((freq) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      
      osc.type = 'triangle';
      osc.detune.value = Math.random() * 10 - 5; // Slight detune for richness
      osc.frequency.value = freq;
      
      // Envelope
      gain.gain.setValueAtTime(0.0001, t);
      // Quick attack (0.05s), sustain at 0.3 gain
      gain.gain.exponentialRampToValueAtTime(0.3, t + 0.05);
      // Long release (2s)
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.05 + 2.0);
      
      osc.connect(gain);
      // Connect directly to reverb for a lush tail
      gain.connect(this.reverb);
      
      osc.start(t);
      osc.stop(t + 2.1);
      
      // Auto-cleanup after fade
      osc.onended = () => {
        osc.disconnect();
        gain.disconnect();
      };
    });
  }

  /**
   * Plays a warning tone when awareness gate triggers.
   */
  playAwarenessAlert() {
    if (!this.initialized || this.ctx.state !== 'running') return;

    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    
    osc.type = 'square';
    // Two-tone alert: ascending minor third (A4 to C5 -> 440Hz to 523.25Hz)
    osc.frequency.setValueAtTime(440, t);
    osc.frequency.setValueAtTime(523.25, t + 0.2); 
    
    // Envelope: 0.1s attack, 0.3s sustain, 0.2s release
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.2, t + 0.1);
    gain.gain.setValueAtTime(0.2, t + 0.4);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.6);
    
    osc.connect(gain);
    gain.connect(this.masterGain);
    
    osc.start(t);
    osc.stop(t + 0.6);
    
    osc.onended = () => {
      osc.disconnect();
      gain.disconnect();
    };
  }

  /**
   * Sets the master volume.
   * @param {number} v - Volume level (0-1).
   */
  setVolume(v) {
    this.volume = Math.max(0, Math.min(1, v));
    if (this.initialized && !this.isMuted) {
      this.masterGain.gain.exponentialRampToValueAtTime(Math.max(0.0001, this.volume), this.ctx.currentTime + 0.1);
    }
  }

  /**
   * Toggles mute state.
   */
  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.initialized) {
      const t = this.ctx.currentTime + 0.1;
      if (this.isMuted) {
        this.masterGain.gain.exponentialRampToValueAtTime(0.0001, t);
      } else {
        this.masterGain.gain.exponentialRampToValueAtTime(Math.max(0.0001, this.volume), t);
      }
    }
  }

  /**
   * Gets initialization status.
   * @returns {boolean} True if initialized.
   */
  get isInitialized() {
    return this.initialized;
  }
}
