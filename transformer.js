/**
 * Dimensions
 */
export const SEQ_LEN = 8;
export const D_MODEL = 20;
export const NUM_HEADS = 5;
export const D_K = 4;
export const D_V = 4;
export const FFN_DIM = 40;

/**
 * Creates a matrix of size rows x cols, filled using fillFn(i, j)
 * @param {number} rows
 * @param {number} cols
 * @param {Function} fillFn
 * @returns {number[][]}
 */
export function createMatrix(rows, cols, fillFn) {
  const mat = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(fillFn(i, j));
    }
    mat.push(row);
  }
  return mat;
}

/**
 * Generates Gaussian noise using Box-Muller transform
 * @returns {number}
 */
function randomGaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Creates a matrix of Gaussian random values
 * @param {number} rows
 * @param {number} cols
 * @param {number} scale
 * @returns {number[][]}
 */
export function gaussianNoise(rows, cols, scale = 1.0) {
  return createMatrix(rows, cols, () => randomGaussian() * scale);
}

/**
 * Creates a randomly initialized matrix (Xavier-like)
 * @param {number} rows
 * @param {number} cols
 * @param {number} scale
 * @returns {number[][]}
 */
export function randomMatrix(rows, cols, scale = 0.1) {
  return gaussianNoise(rows, cols, scale);
}

/**
 * Creates a matrix filled with zeros
 * @param {number} rows
 * @param {number} cols
 * @returns {number[][]}
 */
export function zerosMatrix(rows, cols) {
  return createMatrix(rows, cols, () => 0);
}

/**
 * Matrix multiplication A * B
 * @param {number[][]} A
 * @param {number[][]} B
 * @returns {number[][]}
 */
export function matMul(A, B) {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const result = createMatrix(rowsA, colsB, () => 0);
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * Transpose a matrix
 * @param {number[][]} M
 * @returns {number[][]}
 */
export function transpose(M) {
  const rows = M.length;
  const cols = M[0].length;
  return createMatrix(cols, rows, (i, j) => M[j][i]);
}

/**
 * Element-wise addition of two matrices
 * @param {number[][]} A
 * @param {number[][]} B
 * @returns {number[][]}
 */
export function addMatrices(A, B) {
  return createMatrix(A.length, A[0].length, (i, j) => A[i][j] + B[i][j]);
}

/**
 * Multiply matrix by a scalar
 * @param {number[][]} M
 * @param {number} s
 * @returns {number[][]}
 */
export function scaleMatrix(M, s) {
  return createMatrix(M.length, M[0].length, (i, j) => M[i][j] * s);
}

/**
 * Add a bias vector to each row of a matrix
 * @param {number[][]} M
 * @param {number[]} v
 * @returns {number[][]}
 */
export function addVecToRows(M, v) {
  return createMatrix(M.length, M[0].length, (i, j) => M[i][j] + v[j]);
}

/**
 * Element-wise ReLU activation
 * @param {number[][]} M
 * @returns {number[][]}
 */
export function relu(M) {
  return createMatrix(M.length, M[0].length, (i, j) => Math.max(0, M[i][j]));
}

/**
 * Row-wise softmax with temperature
 * @param {number[][]} M
 * @param {number} temperature
 * @returns {number[][]}
 */
export function softmax(M, temperature = 1.0) {
  const result = [];
  for (let i = 0; i < M.length; i++) {
    const row = M[i];
    let maxVal = -Infinity;
    for (let j = 0; j < row.length; j++) {
      if (row[j] > maxVal) maxVal = row[j];
    }
    
    let sumExp = 0;
    const expRow = [];
    for (let j = 0; j < row.length; j++) {
      const val = Math.exp((row[j] - maxVal) / temperature);
      expRow.push(val);
      sumExp += val;
    }
    
    const outRow = [];
    for (let j = 0; j < row.length; j++) {
      outRow.push(expRow[j] / sumExp);
    }
    result.push(outRow);
  }
  return result;
}

/**
 * Randomly zero elements with probability rate
 * @param {number[][]} M
 * @param {number} rate
 * @returns {number[][]}
 */
export function dropout(M, rate) {
  if (rate <= 0) return M;
  const scale = 1 / (1 - rate);
  return createMatrix(M.length, M[0].length, (i, j) => {
    if (Math.random() < rate) return 0;
    return M[i][j] * scale;
  });
}

/**
 * Normalize a 1D vector to zero mean, unit variance
 * @param {number[]} v
 * @returns {number[]}
 */
export function layerNorm(v) {
  const n = v.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += v[i];
  const mean = sum / n;
  
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    varSum += (v[i] - mean) * (v[i] - mean);
  }
  const variance = varSum / n;
  const std = Math.sqrt(variance + 1e-6);
  
  const result = [];
  for (let i = 0; i < n; i++) {
    result.push((v[i] - mean) / std);
  }
  return result;
}

/**
 * Apply layerNorm to each row of a matrix
 * @param {number[][]} M
 * @returns {number[][]}
 */
export function layerNormMatrix(M) {
  const result = [];
  for (let i = 0; i < M.length; i++) {
    result.push(layerNorm(M[i]));
  }
  return result;
}

/**
 * Positional Encoding for a specific position and dimension
 * @param {number} pos
 * @param {number} dim
 * @returns {Float32Array}
 */
export function positionalEncoding(pos, dim) {
  const pe = new Float32Array(dim);
  for (let i = 0; i < dim; i += 2) {
    const divTerm = Math.pow(10000, i / dim);
    pe[i] = Math.sin(pos / divTerm);
    if (i + 1 < dim) {
      pe[i + 1] = Math.cos(pos / divTerm);
    }
  }
  return pe;
}

/**
 * Add PEs to each row of a matrix
 * @param {number[][]} M
 * @returns {number[][]}
 */
export function addPositionalEncodings(M) {
  const dim = M[0].length;
  return createMatrix(M.length, dim, (i, j) => {
    const pe = positionalEncoding(i, dim);
    return M[i][j] + pe[j];
  });
}

/**
 * Scaled Dot-Product Attention
 * @param {number[][]} Q
 * @param {number[][]} K
 * @param {number[][]} V
 * @param {number} dk
 * @param {number[][]} mask
 * @param {number} dropoutRate
 * @returns {Object} { output, weights }
 */
export function scaledDotProductAttention(Q, K, V, dk, mask = null, dropoutRate = 0) {
  // Q: (seq, dk), K: (seq, dk), V: (seq, dv)
  // scores: Q * K^T / sqrt(dk)
  const Kt = transpose(K);
  let scores = matMul(Q, Kt);
  scores = scaleMatrix(scores, 1 / Math.sqrt(dk));
  
  if (mask) {
    scores = createMatrix(scores.length, scores[0].length, (i, j) => {
      return mask[i][j] === 0 ? scores[i][j] - 1e9 : scores[i][j];
    });
  }
  
  let weights = softmax(scores);
  
  if (dropoutRate > 0) {
    weights = dropout(weights, dropoutRate);
  }
  
  const output = matMul(weights, V);
  return { output, weights };
}

/**
 * Flatten a 2D matrix into a 1D array.
 * @param {number[][]} M
 * @returns {number[]}
 */
export function flattenMatrix(M) {
  const flat = [];
  for (let i = 0; i < M.length; i++) {
    for (let j = 0; j < M[i].length; j++) {
      flat.push(M[i][j]);
    }
  }
  return flat;
}

/**
 * Cosine similarity between two 1D vectors.
 * Mirrors F.cosine_similarity(identity.flatten(), x.flatten(), dim=0)
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number} - cosine similarity in range [-1, 1]
 */
export function cosineSimilarity(a, b) {
  const n = Math.min(a.length, b.length);
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  return dot / (normA * normB);
}

/**
 * Entropy sharpening — reduce entropy by scaling logits before softmax.
 * Mirrors attention regularization (entropy sharpening) in FocusTransformerBlock.
 * @param {number[][]} weights - attention weight matrix (post-softmax)
 * @param {number} sharpness - sharpening factor (>1 sharpens, <1 smooths)
 * @returns {number[][]} - re-normalized sharpened weights
 */
export function entropySharpening(weights, sharpness = 2.0) {
  // Raise weights to a power and re-normalize (temperature sharpening in probability space)
  return weights.map(row => {
    const sharpened = row.map(w => Math.pow(Math.max(w, 1e-10), sharpness));
    const sum = sharpened.reduce((a, b) => a + b, 0);
    return sum > 0 ? sharpened.map(w => w / sum) : sharpened;
  });
}

/**
 * AttentionHead Class
 */
export class AttentionHead {
  constructor(dModel, dk, dv) {
    this.dModel = dModel;
    this.dk = dk;
    this.dv = dv;
    this.dropoutRate = 0;
    this.reset();
  }
  
  reset() {
    this.Wq = randomMatrix(this.dModel, this.dk);
    this.Wk = randomMatrix(this.dModel, this.dk);
    this.Wv = randomMatrix(this.dModel, this.dv);
  }
  
  injectNoise(scale) {
    this.Wq = addMatrices(this.Wq, gaussianNoise(this.dModel, this.dk, scale));
    this.Wk = addMatrices(this.Wk, gaussianNoise(this.dModel, this.dk, scale));
    this.Wv = addMatrices(this.Wv, gaussianNoise(this.dModel, this.dv, scale));
  }
  
  forward(X) {
    const Q = matMul(X, this.Wq);
    const K = matMul(X, this.Wk);
    const V = matMul(X, this.Wv);
    
    return scaledDotProductAttention(Q, K, V, this.dk, null, this.dropoutRate);
  }
}

/**
 * TransformerLayer Class
 */
/**
 * FocusTransformerBlock — mirrors the PyTorch FocusTransformerBlock.
 * Integrates Portal of Awareness (cosine alignment), entropy sharpening,
 * and alignment drift detection into the standard Transformer layer.
 */
export class TransformerLayer {
  constructor(dModel, numHeads, dk, dv, ffnDim) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.dk = dk;
    this.dv = dv;
    this.ffnDim = ffnDim;
    this.dropoutRate = 0;
    
    // Alignment threshold — cosine similarity threshold for drift detection
    // Mirrors: self.alignment_threshold = 0.8
    this.alignmentThreshold = 0.8;
    
    // Entropy sharpening factor (>1 = sharper attention, 1 = no effect)
    this.entropySharpness = 1.5;
    
    // Track alignment history for feedback
    this.alignmentHistory = [];
    this.maxAlignmentHistory = 100;
    
    this.heads = [];
    for (let i = 0; i < numHeads; i++) {
      this.heads.push(new AttentionHead(dModel, dk, dv));
    }
    
    this.Wo = randomMatrix(numHeads * dv, dModel);
    
    this.W1 = randomMatrix(dModel, ffnDim);
    this.b1 = new Array(ffnDim).fill(0);
    this.W2 = randomMatrix(ffnDim, dModel);
    this.b2 = new Array(dModel).fill(0);
  }
  
  setDropoutRate(rate) {
    this.dropoutRate = rate;
    for (const head of this.heads) {
      head.dropoutRate = rate;
    }
  }
  
  setAlignmentThreshold(threshold) {
    this.alignmentThreshold = Math.max(0, Math.min(1, threshold));
  }
  
  setEntropySharpness(sharpness) {
    this.entropySharpness = Math.max(0.1, sharpness);
  }
  
  injectNoise(scale) {
    for (const head of this.heads) {
      head.injectNoise(scale);
    }
    this.Wo = addMatrices(this.Wo, gaussianNoise(this.numHeads * this.dv, this.dModel, scale));
    this.W1 = addMatrices(this.W1, gaussianNoise(this.dModel, this.ffnDim, scale));
    this.W2 = addMatrices(this.W2, gaussianNoise(this.ffnDim, this.dModel, scale));
    
    // add noise to biases
    const b1Noise = gaussianNoise(1, this.ffnDim, scale)[0];
    const b2Noise = gaussianNoise(1, this.dModel, scale)[0];
    for (let i = 0; i < this.ffnDim; i++) this.b1[i] += b1Noise[i];
    for (let i = 0; i < this.dModel; i++) this.b2[i] += b2Noise[i];
  }
  
  forward(X) {
    // ═══ Portal of Awareness (Pre-Attention) ═══
    // Store identity for alignment checking
    // Mirrors: identity = x
    const identity = X.map(row => [...row]);
    
    // ═══ 1. Multi-Head Attention (Dynamic Feedback Loops) ═══
    const headOutputs = [];
    const attentionWeights = [];
    
    for (const head of this.heads) {
      const { output, weights } = head.forward(X);
      headOutputs.push(output);
      // Apply Attention Regularization: entropy sharpening
      // Mirrors: # Apply Attention Regularization here (e.g., entropy sharpening)
      const sharpenedWeights = this.entropySharpness > 1.0
        ? entropySharpening(weights, this.entropySharpness)
        : weights;
      attentionWeights.push(sharpenedWeights);
    }
    
    // Concat along last dimension
    const seqLen = X.length;
    const concatOut = createMatrix(seqLen, this.numHeads * this.dv, () => 0);
    
    for (let i = 0; i < seqLen; i++) {
      let colIdx = 0;
      for (let h = 0; h < this.numHeads; h++) {
        for (let j = 0; j < this.dv; j++) {
          concatOut[i][colIdx++] = headOutputs[h][i][j];
        }
      }
    }
    
    let mhaOut = matMul(concatOut, this.Wo);
    if (this.dropoutRate > 0) mhaOut = dropout(mhaOut, this.dropoutRate);
    
    // ═══ 2. Residual + LayerNorm ═══
    const res1 = addMatrices(X, mhaOut);
    const ln1 = layerNormMatrix(res1);
    
    // ═══ Check Alignment (Meditation Object Check) ═══
    // Mirrors: alignment = F.cosine_similarity(identity.flatten(), x.flatten(), dim=0)
    const identityFlat = flattenMatrix(identity);
    const postAttnFlat = flattenMatrix(ln1);
    const alignment = cosineSimilarity(identityFlat, postAttnFlat);
    const alignmentDrift = alignment < this.alignmentThreshold;
    
    // Track alignment history
    this.alignmentHistory.push(alignment);
    if (this.alignmentHistory.length > this.maxAlignmentHistory) {
      this.alignmentHistory.shift();
    }
    
    // ═══ 3. FFN Phase (Temporal Chunking) ═══
    let ffnOut = relu(addVecToRows(matMul(ln1, this.W1), this.b1));
    if (this.dropoutRate > 0) ffnOut = dropout(ffnOut, this.dropoutRate);
    
    let ffnOut2 = addVecToRows(matMul(ffnOut, this.W2), this.b2);
    if (this.dropoutRate > 0) ffnOut2 = dropout(ffnOut2, this.dropoutRate);
    
    // ═══ 4. Residual + LayerNorm ═══
    const res2 = addMatrices(ln1, ffnOut2);
    const finalOut = layerNormMatrix(res2);
    
    // Mirrors: return x, attn_weights, alignment
    return {
      output: finalOut,
      attentionWeights,
      alignment,
      alignmentDrift
    };
  }
}

/**
 * Compute Shannon entropy of a single attention weight row
 * @param {number[]} weights
 * @returns {number}
 */
export function computeEntropy(weights) {
  let entropy = 0;
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    if (w > 0) {
      entropy -= w * Math.log2(w);
    }
  }
  return entropy;
}

/**
 * Average entropy across all rows of an attention weight matrix
 * @param {number[][]} weightsMatrix
 * @returns {number}
 */
export function computeAverageEntropy(weightsMatrix) {
  let sum = 0;
  for (let i = 0; i < weightsMatrix.length; i++) {
    sum += computeEntropy(weightsMatrix[i]);
  }
  return sum / weightsMatrix.length;
}

/**
 * KL divergence between two weight distributions (row-averaged)
 * @param {number[][]} currentWeights
 * @param {number[][]} targetWeights
 * @returns {number}
 */
export function computeDrift(currentWeights, targetWeights) {
  let totalKl = 0;
  const rows = currentWeights.length;
  const cols = currentWeights[0].length;
  
  for (let i = 0; i < rows; i++) {
    let rowKl = 0;
    for (let j = 0; j < cols; j++) {
      const p = targetWeights[i][j];
      const q = currentWeights[i][j];
      if (p > 0 && q > 0) {
        rowKl += p * Math.log(p / q);
      }
    }
    totalKl += rowKl;
  }
  
  return totalKl / rows;
}
