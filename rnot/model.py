"""R-NOT: a generator on the unit hypersphere that makes exactly two network calls.

    first  = Exp_z(field(z, class, step 0))             coarse jump from the noise point z
    second = Exp_first(field(first, class, step 1))      fine jump to the result

Training (one network for both steps):
    step 1: cross-entropy of first . codes / T against the balanced optimal-transport (Laguerre) cell of z
    step 2: geodesic angle between second and the code nearest to first
"""
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import exp_step
from .transport import cells, fit_potentials


class FourierField(nn.Module):
    """Tangent field u = field(x, class, step): random Fourier features of x, FiLM conditioning on (class, step)."""

    def __init__(self, dim, num_classes, hidden=512, num_freqs=256, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        bands = 2.0 ** torch.linspace(0.0, 4.0, num_freqs)
        self.register_buffer('frequencies', F.normalize(torch.randn(num_freqs, dim, generator=g), dim=-1) * bands[:, None])
        var = self.frequencies.square().sum(-1) / dim
        self.register_buffer('sin_scale', (-.5 * torch.expm1(-2 * var)).sqrt().clamp_min(1e-5))
        self.register_buffer('cos_mean', torch.exp(-.5 * var))
        self.register_buffer('cos_scale', (-torch.expm1(-var) / math.sqrt(2)).clamp_min(1e-5))
        self.context_scale = math.sqrt(hidden / 2.)
        self.class_emb = nn.Embedding(num_classes, hidden // 2)
        self.step_emb = nn.Embedding(2, hidden // 2)
        nn.init.orthogonal_(self.class_emb.weight)
        nn.init.orthogonal_(self.step_emb.weight)
        self.in_proj = nn.Linear(2 * num_freqs + dim + hidden, hidden, bias=False)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(3))
        self.films = nn.ModuleList(nn.Linear(hidden, 2 * hidden, bias=False) for _ in range(3))
        self.ups = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.downs = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.out = nn.Linear(hidden, dim, bias=False)
        nn.init.orthogonal_(self.out.weight, gain=0.02)

    def forward(self, x, labels, step):
        phase = x @ self.frequencies.T
        c = torch.cat([self.class_emb(labels), self.step_emb(torch.full_like(labels, step))], -1) * self.context_scale
        h = F.silu(self.in_proj(torch.cat([phase.sin() / self.sin_scale, (phase.cos() - self.cos_mean) / self.cos_scale,
                                           x * math.sqrt(self.dim), c], -1)))
        for norm, film, up, down in zip(self.norms, self.films, self.ups, self.downs):
            gamma, beta = film(c).chunk(2, -1)
            h = h + down(F.silu(up(norm(h) * (1 + torch.tanh(gamma)) + beta)))
        return self.out(h)


class RNOT(nn.Module):
    """Two-step generator of unit vectors (latent codes) on S^(dim-1), one model per set of classes.

    gen = RNOT(dim=64, num_classes=10)
    gen.fit(codes, labels, seconds=300)      # codes: [N, dim]; labels: [N] in 0..num_classes-1
    z = gen.sample(labels)                   # [len(labels), dim] unit vectors, exactly 2 network calls
    """

    def __init__(self, dim, num_classes=1, k=16, hidden=512, temperature=.1, max_step=math.pi, seed=0):
        super().__init__()
        self.dim, self.num_classes, self.k, self.temperature, self.max_step = dim, num_classes, k, temperature, max_step
        self.field = FourierField(dim, num_classes, hidden, seed=seed)
        self.register_buffer('basis', torch.eye(dim)[:, :k])  # noise subspace, set from the codes in fit()
        self.config = dict(dim=dim, num_classes=num_classes, k=k, hidden=hidden, temperature=temperature,
                           max_step=max_step, seed=seed)

    # ---------------------------------------------------------------- generation
    def noise(self, n, generator=None):
        z = torch.randn(n, self.k, device=self.basis.device, generator=generator)
        return F.normalize(z, dim=-1)

    def steps(self, z, labels):
        """z: [N, k] noise on the k-sphere. Returns (first, second)."""
        x = F.normalize(z @ self.basis.T, dim=-1)
        first, _ = exp_step(x, self.field(x, labels, 0), self.max_step)
        second, _ = exp_step(first.detach(), self.field(first.detach(), labels, 1), self.max_step)
        return first, second

    @torch.no_grad()
    def sample(self, labels, generator=None):
        labels = torch.as_tensor(labels, device=self.basis.device)
        return self.steps(self.noise(len(labels), generator), labels)[1]

    # ---------------------------------------------------------------- training
    def _bank(self, codes, labels):
        """Codes grouped by class into [classes, n, dim]; classes are cut to the size of the smallest one."""
        groups = [codes[labels == c] for c in range(self.num_classes)]
        n = min(len(g) for g in groups)
        if n == 0:
            raise ValueError('every class needs at least one code')
        return torch.stack([g[:n] for g in groups])

    def loss(self, z, labels, bank, psi):
        first, second = self.steps(z, labels)
        first, second = first.float(), second.float()
        loss1, target2 = 0., torch.empty_like(second)
        for c in labels.unique().tolist():
            m = labels == c
            with torch.no_grad():
                cell = cells(z[m].float(), self._keys[c], psi[c])
                target2[m] = bank[c][(first[m] @ bank[c].T).argmax(-1)]
            loss1 = loss1 + F.cross_entropy(first[m] @ bank[c].T / self.temperature, cell, reduction='sum')
        return loss1 / len(labels) + (second * target2).sum(-1).clamp(-.9999, .9999).acos().mean()

    def fit(self, codes, labels=None, seconds=300., batch=192, lr=3e-4, ema=.995, log=print):
        """Train on unit codes [N, dim] for a wall-clock budget. Returns self (weights = exponential moving average)."""
        dev = self.basis.device
        codes = F.normalize(torch.as_tensor(codes, dtype=torch.float32, device=dev), dim=-1)
        labels = torch.zeros(len(codes), dtype=torch.long, device=dev) if labels is None else torch.as_tensor(labels, device=dev).long()
        bank = self._bank(codes, labels)
        flat = bank.reshape(-1, self.dim)
        self.basis = torch.linalg.svd(flat - flat.mean(0), full_matrices=False)[2].T[:, :self.k].contiguous()
        self._keys = F.normalize(bank @ self.basis, dim=-1)
        t0 = time.time()
        psi = fit_potentials(self._keys, log=log)
        log and log(f'optimal transport cells for {bank.shape[0]} x {bank.shape[1]} codes: {time.time() - t0:.0f}s')
        shadow = copy.deepcopy(self).eval().requires_grad_(False)
        opt = torch.optim.AdamW(self.field.parameters(), lr=lr, weight_decay=1e-5)
        amp = dev.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        steps, t0, last = 0, time.time(), time.time()
        while time.time() - t0 < seconds:
            y = torch.randint(0, self.num_classes, (batch,), device=dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = self.loss(self.noise(batch), y, bank, psi)
            steps += 1
            for gr in opt.param_groups:
                gr['lr'] = lr * min(1., steps / 100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(self.field.parameters(), 2.)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                for a, b in zip(shadow.field.parameters(), self.field.parameters()):
                    a.lerp_(b, 1 - ema)
            if log and time.time() - last > 30:
                last = time.time(); log(f'R-NOT: {time.time() - t0:.0f}/{seconds:.0f}s, {steps} updates, loss {loss.item():.3f}')
        self.field.load_state_dict(shadow.field.state_dict())
        del self._keys
        self.updates = steps
        return self.eval()

    # ---------------------------------------------------------------- saving
    def save(self, path):
        torch.save(dict(config=self.config, state=self.state_dict()), path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        d = torch.load(path, map_location=map_location)
        m = cls(**d['config']); m.load_state_dict(d['state'])
        return m.eval()

    def export_onnx(self, path, decoder=None, opset=17):
        """ONNX graph z [N, k] (normal noise), labels [N] -> codes [N, dim], or decoder(codes) if a decoder module
        (nn.Module mapping codes to e.g. pictures) is given. Run it in a browser with onnxruntime-web."""

        class Wrapped(nn.Module):
            def __init__(self, gen, dec):
                super().__init__()
                self.gen, self.dec = gen, dec

            def forward(self, z, labels):
                out = self.gen.steps(F.normalize(z, dim=-1), labels)[1]
                return self.dec(out) if self.dec is not None else out

        w = Wrapped(copy.deepcopy(self).cpu(), copy.deepcopy(decoder).cpu() if decoder is not None else None)
        w = w.eval().requires_grad_(False)
        z, y = torch.randn(2, self.k), torch.zeros(2, dtype=torch.long)
        kw = dict(input_names=['z', 'labels'], output_names=['out'], opset_version=opset,
                  dynamic_axes={'z': {0: 'n'}, 'labels': {0: 'n'}, 'out': {0: 'n'}})
        try:
            torch.onnx.export(w, (z, y), path, dynamo=False, **kw)
        except TypeError:  # torch < 2.5 has no 'dynamo' argument
            torch.onnx.export(w, (z, y), path, **kw)
