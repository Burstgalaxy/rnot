// R-NOT digit generator in plain JavaScript: two steps on the sphere S^63, then the auto-encoder's decoder (28x28).
// Weights come from weights.js (window.RNOT_WEIGHTS = {manifest, data: base64 float32}); works in a browser and in Node.
(function (root) {
  'use strict';

  function loadWeights(pack) {
    const bin = typeof atob === 'function' ? atob(pack.data) : Buffer.from(pack.data, 'base64').toString('binary');
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const all = new Float32Array(bytes.buffer);
    const w = {};
    for (const [name, [offset, shape]] of Object.entries(pack.manifest)) {
      const size = shape.reduce((a, b) => a * b, 1);
      w[name] = { a: all.subarray(offset, offset + size), shape };
    }
    return w;
  }

  // y = x W^T (+ b): x [n, in], W [out, in] (torch Linear layout)
  function linear(x, n, W, b) {
    const [out, inp] = W.shape, y = new Float32Array(n * out), wa = W.a;
    for (let r = 0; r < n; r++) {
      const xo = r * inp, yo = r * out;
      for (let o = 0; o < out; o++) {
        let s = b ? b.a[o] : 0;
        const wo = o * inp;
        for (let i = 0; i < inp; i++) s += x[xo + i] * wa[wo + i];
        y[yo + o] = s;
      }
    }
    return y;
  }
  const silu = v => v / (1 + Math.exp(-v));
  function layerNorm(x, n, d, g, b) {
    const y = new Float32Array(n * d);
    for (let r = 0; r < n; r++) {
      let m = 0, v = 0;
      for (let i = 0; i < d; i++) m += x[r * d + i];
      m /= d;
      for (let i = 0; i < d; i++) { const t = x[r * d + i] - m; v += t * t; }
      const inv = 1 / Math.sqrt(v / d + 1e-5);
      for (let i = 0; i < d; i++) y[r * d + i] = (x[r * d + i] - m) * inv * (g ? g.a[i] : 1) + (b ? b.a[i] : 0);
    }
    return y;
  }
  function normalizeRows(x, n, d) {
    for (let r = 0; r < n; r++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += x[r * d + i] ** 2;
      const inv = 1 / Math.max(Math.sqrt(s), 1e-12);
      for (let i = 0; i < d; i++) x[r * d + i] *= inv;
    }
    return x;
  }

  // R-NOT field: random Fourier features + FiLM blocks conditioned on (digit, step)
  function field(w, x, n, labels, step) {
    const D = 64, F = w['field.frequencies'].shape[0], H = w['field.in_proj.weight'].shape[0];
    const ctx = Math.sqrt(H / 2), half = H / 2;
    const feats = new Float32Array(n * (2 * F + D + H)), c = new Float32Array(n * H);
    const fr = w['field.frequencies'].a, ss = w['field.sin_scale'].a, cm = w['field.cos_mean'].a, cs = w['field.cos_scale'].a;
    const ce = w['field.class_emb.weight'].a, se = w['field.step_emb.weight'].a;
    const W = 2 * F + D + H;
    for (let r = 0; r < n; r++) {
      const o = r * W;
      for (let f = 0; f < F; f++) {
        let ph = 0;
        for (let i = 0; i < D; i++) ph += x[r * D + i] * fr[f * D + i];
        feats[o + f] = Math.sin(ph) / ss[f];
        feats[o + F + f] = (Math.cos(ph) - cm[f]) / cs[f];
      }
      for (let i = 0; i < D; i++) feats[o + 2 * F + i] = x[r * D + i] * Math.sqrt(D);
      for (let i = 0; i < half; i++) {
        c[r * H + i] = ce[labels[r] * half + i] * ctx;
        c[r * H + half + i] = se[step * half + i] * ctx;
      }
      for (let i = 0; i < H; i++) feats[o + 2 * F + D + i] = c[r * H + i];
    }
    let h = linear(feats, n, w['field.in_proj.weight']).map(silu);
    for (let b = 0; b < 3; b++) {
      const gb = linear(c, n, w[`field.films.${b}.weight`]);
      const nh = layerNorm(h, n, H, w[`field.norms.${b}.weight`], w[`field.norms.${b}.bias`]);
      for (let r = 0; r < n; r++) for (let i = 0; i < H; i++)
        nh[r * H + i] = nh[r * H + i] * (1 + Math.tanh(gb[r * 2 * H + i])) + gb[r * 2 * H + H + i];
      const up = linear(nh, n, w[`field.ups.${b}.weight`]).map(silu);
      const down = linear(up, n, w[`field.downs.${b}.weight`]);
      for (let i = 0; i < h.length; i++) h[i] += down[i];
    }
    return linear(h, n, w['field.out.weight']);
  }

  // geodesic step: move x along the great circle by angle pi * tanh(|u_tan| / pi)
  function expStep(x, u, n, d, maxStep) {
    const y = new Float32Array(n * d);
    for (let r = 0; r < n; r++) {
      let dot = 0;
      for (let i = 0; i < d; i++) dot += x[r * d + i] * u[r * d + i];
      let nn = 0;
      const t = new Float32Array(d);
      for (let i = 0; i < d; i++) { t[i] = u[r * d + i] - dot * x[r * d + i]; nn += t[i] * t[i]; }
      nn = Math.max(Math.sqrt(nn), 1e-6);
      const ang = Math.tanh(nn / maxStep) * maxStep, cs = Math.cos(ang), sn = Math.sin(ang);
      for (let i = 0; i < d; i++) y[r * d + i] = cs * x[r * d + i] + sn * t[i] / nn;
    }
    return normalizeRows(y, n, d);
  }

  // transposed convolution, kernel 3, stride 2, padding 1, output padding 1 (H -> 2H); W [cin, cout, 3, 3]
  function convT(x, n, cin, hgt, W, b) {
    const cout = W.shape[1], H2 = hgt * 2, y = new Float32Array(n * cout * H2 * H2), wa = W.a;
    for (let s = 0; s < n; s++) {
      const yo = s * cout * H2 * H2, xo = s * cin * hgt * hgt;
      for (let co = 0; co < cout; co++) { const bb = b ? b.a[co] : 0; for (let p = 0; p < H2 * H2; p++) y[yo + co * H2 * H2 + p] = bb; }
      for (let ci = 0; ci < cin; ci++) for (let iy = 0; iy < hgt; iy++) for (let ix = 0; ix < hgt; ix++) {
        const v = x[xo + ci * hgt * hgt + iy * hgt + ix];
        if (v === 0) continue;
        for (let co = 0; co < cout; co++) for (let ky = 0; ky < 3; ky++) {
          const oy = iy * 2 - 1 + ky;
          if (oy < 0 || oy >= H2) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ox = ix * 2 - 1 + kx;
            if (ox < 0 || ox >= H2) continue;
            y[yo + co * H2 * H2 + oy * H2 + ox] += v * wa[((ci * cout + co) * 3 + ky) * 3 + kx];
          }
        }
      }
    }
    return y;
  }

  function decode(w, z, n) {
    let h = linear(z, n, w['dec.0.weight'], w['dec.0.bias']);
    h = layerNorm(h, n, 256, w['dec.1.weight'], w['dec.1.bias']).map(silu);
    h = linear(h, n, w['dec.3.weight'], w['dec.3.bias']).map(silu);
    h = convT(h, n, 64, 7, w['dec.6.weight'], w['dec.6.bias']);
    const g = w['dec.7.weight'].a, bb = w['dec.7.bias'].a, m = w['dec.7.running_mean'].a, v = w['dec.7.running_var'].a;
    for (let s = 0; s < n; s++) for (let c = 0; c < 32; c++) {
      const sc = g[c] / Math.sqrt(v[c] + 1e-5);
      for (let p = 0; p < 196; p++) { const i = (s * 32 + c) * 196 + p; h[i] = silu((h[i] - m[c]) * sc + bb[c]); }
    }
    h = convT(h, n, 32, 14, w['dec.9.weight'], w['dec.9.bias']);
    return h.map(v2 => 1 / (1 + Math.exp(-v2)));
  }

  // z: Float32Array [n, k] of normal noise; labels: digits. Returns n pictures 28x28 in [0, 1].
  function generate(w, z, labels) {
    const n = labels.length, k = w.basis.shape[1], D = 64, B = w.basis.a;
    const x = new Float32Array(n * D);
    normalizeRows(z, n, k);
    for (let r = 0; r < n; r++) for (let i = 0; i < D; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += z[r * k + j] * B[i * k + j];
      x[r * D + i] = s;
    }
    normalizeRows(x, n, D);
    const first = expStep(x, field(w, x, n, labels, 0), n, D, Math.PI);   // network call 1
    const second = expStep(first, field(w, first, n, labels, 1), n, D, Math.PI);  // network call 2
    return decode(w, second, n);
  }

  function gaussian(n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const u = 1 - Math.random(), v = Math.random(), r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  }

  const api = { loadWeights, generate, gaussian };
  if (typeof module !== 'undefined') module.exports = api; else root.RNOT = api;
})(typeof window !== 'undefined' ? window : globalThis);
