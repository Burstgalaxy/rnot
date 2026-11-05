"""Full R-NOT example on handwritten digits (MNIST), built on the rnot library.

1. Spherical auto-encoder: a digit <-> a point on the hypersphere S^63 (the latent representation).
2. A closed teacher: flow matching in the latent space, 128 steps, gives out only finished pictures.
3. R-NOT student (rnot.RNOT) learns from 30 000 teacher pictures and draws digits in exactly 2 network calls.
4. Scores (classifier accuracy, Frechet distance to real digits, novelty, time) and pictures.
5. Export: rnot_digits.onnx = 2 R-NOT steps + the VAE decoder, noise in -> 28x28 picture out (for the web page).
Env: BUDGET (s per trained model, default 300), SMOKE=1 (tiny run), OUT (output folder), HIDDEN (R-NOT width).
"""
import json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rnot import RNOT

DEV = 'cuda'
SMOKE = bool(os.environ.get('SMOKE'))
BUDGET = 3. if SMOKE else float(os.environ.get('BUDGET', 300))
LAT, NCLS, K = 64, 10, 16
PER_CLASS = 64 if SMOKE else 3000  # teacher pictures per digit for R-NOT
N_EVAL = 1000 if SMOKE else 10000
OUT = os.environ.get('OUT', '.')
t_start = time.time()


def log(*a):
    print(f'[{time.time() - t_start:5.0f}s]', *a, flush=True)


# ---------------------------------------------------------------- data, VAE, classifier
def load_mnist():
    from torchvision import datasets
    tr, te = datasets.MNIST('data', train=True, download=True), datasets.MNIST('data', train=False, download=True)
    f = lambda d: (d.data.float().div(255).unsqueeze(1).to(DEV), d.targets.to(DEV))
    return f(tr), f(te)


class SphericalVAE(nn.Module):
    """The user's spherical image auto-encoder (latent = unit vector)."""
    def __init__(self, latent=LAT, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 7, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 64 * 7 * 7), nn.SiLU(),
            nn.Unflatten(1, (64, 7, 7)), nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1))

    def encode(self, x):
        return F.normalize(self.encoder(x).float(), dim=-1)

    def decode(self, z):
        return torch.sigmoid(self.decoder(F.normalize(z.float(), dim=-1)).float())


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(1, 32, 3, 1, 1), nn.SiLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, 1, 1), nn.SiLU(),
                                  nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.SiLU())
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        h = self.body(x)
        return self.head(h), h


def batches(n, bs, gen):
    perm = torch.randperm(n, device=DEV, generator=gen)
    return [perm[i:i + bs] for i in range(0, n, bs)]


def train_vae(X, epochs):
    vae = SphericalVAE().to(DEV)
    opt = torch.optim.AdamW(vae.parameters(), lr=2e-3, weight_decay=1e-5)
    g = torch.Generator(device=DEV).manual_seed(0)
    for ep in range(epochs):
        tot = 0.
        for idx in batches(len(X), 512, g):
            x = X[idx]
            loss = F.binary_cross_entropy_with_logits(vae.decoder(vae.encode(x)), x)  # fp32: small and fast enough
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item()
        log(f'VAE epoch {ep + 1}/{epochs}: loss {tot / math.ceil(len(X) / 512):.4f}')
    return vae.eval()


def train_classifier(X, Y, epochs):
    clf = Classifier().to(DEV)
    opt = torch.optim.AdamW(clf.parameters(), lr=2e-3)
    g = torch.Generator(device=DEV).manual_seed(1)
    for ep in range(epochs):
        for idx in batches(len(X), 256, g):
            loss = F.cross_entropy(clf(X[idx])[0], Y[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return clf.eval()


@torch.no_grad()
def encode_all(vae, X):
    return torch.cat([vae.encode(X[i:i + 4096]) for i in range(0, len(X), 4096)])


@torch.no_grad()
def decode_all(vae, Z):
    return torch.cat([vae.decode(Z[i:i + 4096]) for i in range(0, len(Z), 4096)])


# ---------------------------------------------------------------- flow matching teacher (latent space, R^64)
class FMNet(nn.Module):
    """Class-conditional velocity v(x, t, s) with sinusoidal time / step-size embeddings and residual MLP blocks."""
    def __init__(self, dim=LAT, hidden=512, blocks=4):
        super().__init__()
        self.cls = nn.Embedding(NCLS, hidden)
        self.register_buffer('freq', torch.exp(-math.log(1000.) * torch.arange(32) / 31) * 100.)
        self.time = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
                                    for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        self.scale = math.sqrt(dim)

    def forward(self, x, y, t, s=None):
        emb = lambda u: torch.cat([(u[:, None] * self.freq).sin(), (u[:, None] * self.freq).cos()], -1)
        s = torch.zeros_like(t) if s is None else s
        h = self.inp(x / self.scale) + self.cls(y) + self.time(torch.cat([emb(t), emb(s)], -1))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


@torch.no_grad()
def euler(net, x, y, n, keep_half=False):
    half = None
    for i in range(n):
        if keep_half and i == n // 2:
            half = x.clone()
        t = torch.full((len(x),), i / n, device=DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            x = x + net(x, y, t).float() / n
    return (x, half) if keep_half else x


class EMA:
    def __init__(self, net, decay=.999):
        import copy
        self.net, self.decay = copy.deepcopy(net).eval().requires_grad_(False), decay

    @torch.no_grad()
    def update(self, net):
        for a, b in zip(self.net.parameters(), net.parameters()):
            a.lerp_(b, 1 - self.decay)


def train_fm(Z, Y, label, init=None, traj=None):
    """mode fm (traj None): flow matching on latents x sqrt(64) with Gaussian x0. distill: each of the 2 Euler steps
    regresses the teacher's half path (x0 -> x_half -> x1), starting from the teacher's weights."""
    import copy
    torch.manual_seed(2)
    net = copy.deepcopy(init).train().requires_grad_(True) if init is not None else FMNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    ema = EMA(net)
    scaler = torch.amp.GradScaler('cuda')
    g = torch.Generator(device=DEV).manual_seed(3)
    steps, last, t0 = 0, time.time(), time.time()
    scale = math.sqrt(LAT)
    while time.time() - t0 < BUDGET:
        if traj is None:
            i = torch.randint(0, len(Z), (512,), device=DEV, generator=g)
            x1, y = Z[i] * scale, Y[i]
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=DEV, generator=g)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = (net((1 - t[:, None]) * x0 + t[:, None] * x1, y, t).float() - (x1 - x0)).square().mean()
        else:
            i = torch.randint(0, len(traj['x0']), (512,), device=DEV, generator=g)
            x0, xh, x1, y = traj['x0'][i], traj['xh'][i], traj['x1'][i], traj['y'][i]
            with torch.autocast('cuda', dtype=torch.float16):
                v0 = net(x0, y, torch.zeros(len(y), device=DEV)).float()
                v1 = net(xh, y, torch.full((len(y),), .5, device=DEV)).float()
            loss = (v0 - 2 * (xh - x0)).square().mean() + (v1 - 2 * (x1 - xh)).square().mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); scaler.step(opt); scaler.update(); ema.update(net)
        steps += 1
        if time.time() - last > 30:
            last = time.time(); log(f'{label}: {time.time() - t0:.0f}/{BUDGET:.0f}s, {steps} updates, loss {loss.item():.4f}')
    log(f'{label}: done, {steps} updates')
    return ema.net, steps


class ClosedTeacher:
    """What a closed model offers: a digit in, a finished picture out. Noise, path and weights stay inside."""
    def __init__(self, fm, vae, steps=128):
        self._fm, self._vae, self._steps = fm, vae, steps
        self._g = torch.Generator(device=DEV).manual_seed(12345)

    @torch.no_grad()
    def generate(self, digits):
        x0 = torch.randn(len(digits), LAT, device=DEV, generator=self._g)
        return self._vae.decode(euler(self._fm, x0, digits, self._steps))


# ---------------------------------------------------------------- scores
@torch.no_grad()
def features(clf, imgs):
    out = [clf(imgs[i:i + 4096]) for i in range(0, len(imgs), 4096)]
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


def frechet(a, b):
    a, b = a.double(), b.double()
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    ev = torch.linalg.eigvals(ca @ cb).real.clamp_min(0).sqrt().sum()
    return float((ma - mb).square().sum() + ca.trace() + cb.trace() - 2 * ev)


@torch.no_grad()
def nn_angle(Z, ref):
    best = torch.cat([(Z[i:i + 2048] @ ref.T).max(1).values for i in range(0, len(Z), 2048)])
    return torch.rad2deg(best.clamp(-1, 1).acos())


def score(name, Zgen, labels, vae, clf, test_feat, Ztrain, test_ratio_ref, gen_ms, calls, extra=None):
    imgs = decode_all(vae, Zgen)
    logits, feat = features(clf, imgs)
    acc = float((logits.argmax(-1) == labels).float().mean())
    fd = frechet(feat, test_feat)
    ang = nn_angle(F.normalize(Zgen, dim=-1), Ztrain)
    r = dict(method=name, calls=calls, accuracy=round(acc, 4), frechet=round(fd, 3),
             nn_angle_median=round(float(ang.median()), 2), novelty=round(float(ang.median()) / test_ratio_ref, 3),
             copies_under_1deg=round(float((ang < 1).float().mean()), 4), gen_ms_10k=round(gen_ms, 2), **(extra or {}))
    log('score', json.dumps(r))
    return r, imgs


def timed(fn, labels):
    fn(labels[:256]); torch.cuda.synchronize()  # warm-up
    t = time.perf_counter(); z = fn(labels); torch.cuda.synchronize()
    return z, 1000 * (time.perf_counter() - t)


def grid(imgs, labels, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 10, figsize=(6, 6.4), dpi=120)
    for dgt in range(10):
        idx = (labels == dgt).nonzero()[:10, 0]
        for j, i in enumerate(idx.tolist()):
            ax = axes[dgt, j]; ax.imshow(imgs[i, 0].cpu().numpy(), cmap='gray'); ax.axis('off')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(); plt.savefig(path); plt.close(fig)



class Decoder(nn.Module):
    """latent point on the sphere -> 28x28 picture (for the ONNX export)"""
    def __init__(self, vae):
        super().__init__()
        self.decoder = vae.decoder

    def forward(self, z):
        return torch.sigmoid(self.decoder(F.normalize(z, dim=-1)))


def main():
    assert torch.cuda.is_available(), 'needs a GPU'
    torch.backends.cuda.matmul.allow_tf32 = False
    (Xtr, Ytr), (Xte, Yte) = load_mnist()
    if SMOKE:
        Xtr, Ytr = Xtr[:4096], Ytr[:4096]
    log(f'MNIST: {len(Xtr)} train, {len(Xte)} test')
    vae = train_vae(Xtr, 1 if SMOKE else 15)
    clf = train_classifier(Xtr, Ytr, 1 if SMOKE else 3)
    with torch.no_grad():
        acc = float((features(clf, Xte)[0].argmax(-1) == Yte).float().mean())
        Ztr, Zte = encode_all(vae, Xtr), encode_all(vae, Xte)
    test_feat = features(clf, Xte)[1]
    ref = float(nn_angle(Zte, Ztr).median())
    log(f'classifier accuracy {acc:.4f}; real test digits: median angle to nearest training latent {ref:.2f} deg')

    fm, fm_steps = train_fm(Ztr, Ytr, 'teacher FM')
    teacher = ClosedTeacher(fm, vae)
    digits = torch.arange(NCLS, device=DEV).repeat_interleave(PER_CLASS)
    pics = torch.cat([teacher.generate(digits[i:i + 3000]) for i in range(0, len(digits), 3000)])
    with torch.no_grad():
        codes = encode_all(vae, pics)  # the student side encodes the pictures itself
    log(f'closed teacher: {len(pics)} pictures')

    gen = RNOT(dim=LAT, num_classes=NCLS, k=K, hidden=int(os.environ.get('HIDDEN', 256))).to(DEV)
    gen.fit(codes, digits, seconds=BUDGET, log=log)
    gen.save(os.path.join(OUT, 'rnot_digits.pt'))

    labels = torch.arange(NCLS, device=DEV).repeat_interleave(N_EVAL // NCLS)
    torch.manual_seed(2024)
    results = []
    for i, (name, fn, calls, upd) in enumerate([
            ('FM, 128 шагов (учитель)', lambda l: euler(fm, torch.randn(len(l), LAT, device=DEV), l, 128), 128, fm_steps),
            ('FM, 2 шага', lambda l: euler(fm, torch.randn(len(l), LAT, device=DEV), l, 2), 2, fm_steps),
            ('R-NOT, 2 шага (библиотека rnot)', lambda l: gen.sample(l).float(), 2, gen.updates)]):
        with torch.no_grad():
            z, ms = timed(fn, labels)
        r, imgs = score(name, z, labels, vae, clf, test_feat, Ztr, ref, ms, calls, dict(train_updates=upd))
        results.append(r)
        grid(imgs, labels, os.path.join(OUT, f'digits_{i}.png'), f'{name}\nверных {100 * r["accuracy"]:.1f}%, время 10к: {ms:.0f} мс')

    # web export: 2 R-NOT steps + decoder in one ONNX graph
    gen.cpu().export_onnx(os.path.join(OUT, 'rnot_digits.onnx'), decoder=Decoder(vae))
    meta = dict(k=gen.k, classes=NCLS, image=28, accuracy=results[-1]['accuracy'], frechet=results[-1]['frechet'],
                gen_ms_10k=results[-1]['gen_ms_10k'], teacher_ms_10k=results[0]['gen_ms_10k'])
    json.dump(dict(results=results, classifier_acc=acc, meta=meta), open(os.path.join(OUT, 'digits_results.json'), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)
    print('\n===== ИТОГ: генератор цифр на библиотеке rnot =====')
    for r in results:
        print(f"  {r['method']:34s} шагов {r['calls']:3d} | верная цифра {100 * r['accuracy']:5.1f}% | Фреше {r['frechet']:6.2f} | "
              f"новизна {r['novelty']:.3f} | время 10к {r['gen_ms_10k']:7.1f} мс")
    print('  ONNX:', os.path.getsize(os.path.join(OUT, 'rnot_digits.onnx')) // 1024, 'KB')


if __name__ == '__main__':
    main()
