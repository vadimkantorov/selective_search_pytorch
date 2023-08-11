"""Poincare utils functions."""
"""LCA construction utils."""
"""Math util functions."""

import torch

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# tanh ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output / (1 - input ** 2)
        return grad


def arctanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


# ################# cosh ########################

class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1 + 1e-7)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


def arcosh(x):
    return Arcosh.apply(x)


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


# ################# sinh ########################

class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def arsinh(x):
    return Arsinh.apply(x)


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()

def egrad2rgrad(p, dp):
    """Converts Euclidean gradient to Hyperbolic gradient."""
    lambda_p = lambda_(p)
    dp /= lambda_p.pow(2)
    return dp


def lambda_(x):
    """Computes the conformal factor."""
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - x_sqnorm).clamp_min(MIN_NORM)


def inner(x, u, v=None):
    """Computes inner product for two tangent vectors."""
    if v is None:
        v = u
    lx = lambda_(x)
    return lx ** 2 * (u * v).sum(dim=-1, keepdim=True)


def gyration(u, v, w):
    """Gyration."""
    u2 = u.pow(2).sum(dim=-1, keepdim=True)
    v2 = v.pow(2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    uw = (u * w).sum(dim=-1, keepdim=True)
    vw = (v * w).sum(dim=-1, keepdim=True)
    a = - uw * v2 + vw + 2 * uv * vw
    b = - vw * u2 - uw
    d = 1 + 2 * uv + u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def ptransp(x, y, u):
    """Parallel transport."""
    lx = lambda_(x)
    ly = lambda_(y)
    return gyration(y, -x, u) * lx / ly


def expmap(u, p):
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = tanh(lambda_(p) * u_norm / 2) * u / u_norm
    gamma_1 = mobius_add(p, second_term)
    return gamma_1


def project(x):
    """Projects points on the manifold."""
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y):
    """Mobius addition."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def mobius_mul(x, t):
    """Mobius scalar multiplication."""
    normx = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return tanh(t * arctanh(normx)) * x / normx


def get_midpoint_o(x):
    """
    Computes hyperbolic midpoint between x and the origin.
    """
    return mobius_mul(x, 0.5)


def hyp_dist_o(x):
    """
    Computes hyperbolic distance between x and the origin.
    """
    x_norm = x.norm(dim=-1, p=2, keepdim=True)
    return 2 * arctanh(x_norm)


def isometric_transform(a, x):
    """Reflection (circle inversion of x through orthogonal circle centered at a)."""
    r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
    u = x - a
    return r2 / torch.sum(u ** 2, dim=-1, keepdim=True) * u + a


def reflection_center(mu):
    """Center of inversion circle."""
    return mu / torch.sum(mu ** 2, dim=-1, keepdim=True)


def euc_reflection(x, a):
    """
    Euclidean reflection (also hyperbolic) of x
    Along the geodesic that goes through a and the origin
    (straight line)
    """
    xTa = torch.sum(x * a, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    proj = xTa * a / norm_a_sq
    return 2 * proj - x


def _halve(x):
    """ computes the point on the geodesic segment from o to x at half the distance """
    return x / (1. + torch.sqrt(1 - torch.sum(x ** 2, dim=-1, keepdim=True)))


def hyp_lca(a, b, return_coord=True):
    """
    Computes projection of the origin on the geodesic between a and b, at scale c

    More optimized than hyp_lca1
    """
    r = reflection_center(a)
    b_inv = isometric_transform(r, b)
    o_inv = a
    o_inv_ref = euc_reflection(o_inv, b_inv)
    o_ref = isometric_transform(r, o_inv_ref)
    proj = _halve(o_ref)
    if not return_coord:
        return hyp_dist_o(proj)
    else:
        return proj
