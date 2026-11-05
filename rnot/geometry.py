"""Geometry of the unit hypersphere S^(d-1): tangent projection and the geodesic (exponential-map) step."""
import torch
import torch.nn.functional as F


def tangent(x, u):
    """Part of u tangent to the sphere at x."""
    return u - (x * u).sum(-1, keepdim=True) * x


def exp_step(x, u, max_step):
    """Move x along the great circle in the direction of the tangent part of u.
    The angle is max_step * tanh(|u| / max_step), so a step never exceeds max_step radians.
    Returns the new point and the length of the tangent vector."""
    u_tan = tangent(x, u)
    n = u_tan.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    angle = torch.tanh(n / max_step) * max_step
    return F.normalize(torch.cos(angle) * x + torch.sin(angle) * (u_tan / n), dim=-1), n.squeeze(-1)


def angle(a, b):
    """Geodesic distance (radians) between unit vectors."""
    return (a * b).sum(-1).clamp(-1, 1).acos()
