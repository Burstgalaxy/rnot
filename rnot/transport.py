"""Semi-discrete optimal transport from uniform noise on a k-sphere to a finite set of codes: balanced Laguerre cells.

Every code gets the same share of the noise; a noise point z belongs to the cell of argmax_i <z, key_i> - psi_i.
The potentials psi are found by a damped diagonal Newton method on the dual problem."""
import torch
import torch.nn.functional as F


def cells(z, keys, psi, chunk=4096, with_gap=False):
    """Laguerre cell of every noise point (and the top-1 / top-2 score gap).
    For k <= 8 the dot product loses precision near 1, so 8 candidates are re-ranked by explicit distances."""
    exact = keys.shape[-1] <= 8
    keys_, psi_ = keys.float(), psi.float()
    out, gaps = [], []
    for part in z.split(chunk):
        score = part.float() @ keys_.T - psi_[None]
        if exact:
            cand = score.topk(min(8, score.shape[1]), dim=-1).indices
            score = -.5 * (part.float()[:, None] - keys_[cand]).square().sum(-1) - psi_[cand]
        top = score.topk(min(2, score.shape[1]), dim=-1)
        best = top.indices[:, 0]
        out.append(cand.gather(1, best[:, None])[:, 0] if exact else best)
        if with_gap:
            gaps.append(top.values[:, 0] - top.values[:, -1])
    return (torch.cat(out), torch.cat(gaps)) if with_gap else torch.cat(out)


@torch.no_grad()
def fit_potentials(keys, iterations=800, samples=None, gamma=.5, log=None):
    """keys: [classes, n, k] unit vectors. Returns psi: [classes, n] such that every cell holds 1/n of the noise.
    Each psi moves in units of the mean score gap of its own cell (dimensionless step); Polyak average of the
    second half of the iterations."""
    classes, n, k = keys.shape
    samples = samples or max(32768, 4 * n)
    psi = torch.zeros(classes, n, dtype=torch.float64, device=keys.device)
    avg = torch.zeros_like(psi)
    for it in range(iterations):
        for c in range(classes):
            z = F.normalize(torch.randn(samples, k, device=keys.device), dim=-1)
            cell, gap = cells(z, keys[c], psi[c], with_gap=True)
            hits = torch.bincount(cell, minlength=n).double()
            g = torch.bincount(cell, weights=gap.double(), minlength=n) / hits.clamp_min(1)
            g = torch.where(hits > 0, g, gap.double().mean())  # an empty cell is pulled back in
            psi[c] += gamma * g * (hits / samples * n - 1)
        if it >= iterations // 2:
            avg += psi / (iterations - iterations // 2)
        if log and it % 200 == 0:
            log(f'potentials: iteration {it}/{iterations}')
    return avg
