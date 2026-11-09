"""Official Mamba-3 SISO module with differentiable PyTorch CPU kernels.

Module parameters and preprocessing are executed from the pinned official class.
Dense short-history and recurrent continuation implement its reference equations;
the fused Triton/CuTe kernels and MIMO mode are not executed by this experiment.
"""
import ast
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
ROOT = Path(__file__).resolve().parent

class CPURMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__(); self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
    def forward(self, x):
        return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+self.eps)*self.weight

def rotary(x, angle):
    """SISO uses adjacent real pairs, with unrotated trailing dimensions."""
    pair = x.reshape(*x.shape[:-1], -1, 2)
    count = pair.shape[-2]-angle.shape[-1]
    cos = F.pad(angle.cos(), (0, count), value=1.)
    sin = F.pad(angle.sin(), (0, count), value=0.)
    a, b = pair.unbind(-1)
    return torch.stack((a*cos-b*sin, a*sin+b*cos), -1).reshape_as(x)

def siso_dense(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D=None, Z=None,
               Input_States=None, return_final_states=False, **kwargs):
    """Quadratic short-context dual form of official trapezoidal SISO recurrence."""
    batch, length, heads, width = V.shape
    state_size = Q.shape[-1]; angle_size = Angles.shape[-1]
    Q = Q.repeat_interleave(heads//Q.shape[2], 2)+Q_bias[None, None]
    K = K.repeat_interleave(heads//K.shape[2], 2)+K_bias[None, None]
    angles = (Angles.tanh()*math.pi*DT.transpose(1, 2)[..., None]).cumsum(1)
    if Input_States is not None: angles = angles+Input_States[0][:, None]
    angles = torch.remainder(angles, 2*math.pi)
    q, k = rotary(Q, angles), rotary(K, angles)
    trap = Trap.sigmoid(); gamma = DT*trap
    shifted = F.pad((DT*(1-trap))[:, :, 1:], (0, 1))
    scaled_k = k*(gamma+shifted).transpose(1, 2)[..., None]
    cumsum = ADT.cumsum(-1)
    segments = cumsum[..., :, None]-cumsum[..., None, :]
    mask = torch.ones(length, length, dtype=torch.bool, device=Q.device).tril()
    decay = segments.masked_fill(~mask, float('-inf')).exp()
    qk = torch.einsum('bthn,bshn->bhts', q, scaled_k)
    diagonal = torch.eye(length, dtype=torch.bool, device=Q.device)
    qk = qk.masked_fill(diagonal, 0)+torch.diag_embed((q*k).sum(-1).transpose(1, 2)*gamma)
    out = torch.einsum('bhts,bshp->bthp', qk*decay, V)
    # Compute the diagonal directly, avoiding cancellation of future-token mass.
    last_state = torch.einsum('bhs,bshp,bshn->bhpn', decay[:, :, -1], V, scaled_k)
    if Input_States is not None:
        _, initial, previous_k, previous_v = Input_States
        correction = previous_v[..., None]*previous_k[..., None, :]*(DT[:, :, 0]*(1-trap[:, :, 0]))[..., None, None]
        initial = initial+correction
        out = out+torch.einsum('bhpn,bthn,bht->bthp', initial, q, cumsum.exp())
        last_state = last_state+initial*cumsum[:, :, -1].exp()[..., None, None]
    if D is not None: out = out+D[None, None, :, None]*V
    if Z is not None: out = out*F.silu(Z)
    cache = (angles[:, -1], last_state, k[:, -1], V[:, -1])
    return (out, *cache) if return_final_states else out

def official_class():
    path = ROOT/'official_mamba3.py'
    tree = ast.parse(path.read_text())
    selected = [v for v in tree.body if isinstance(v, (ast.FunctionDef, ast.ClassDef)) and v.name in ['heavy_tail_activation', 'Mamba3']]
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'rearrange': rearrange, 'repeat': repeat,
                 'RMSNormGated': CPURMSNorm, 'mamba3_siso_combined': siso_dense,
                 'mamba3_mimo_combined': None, 'mamba3_step_fn': None}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['Mamba3']

class StreamingMamba3(official_class()):
    def __init__(self, width=16, state=8, expand=2, headdim=4):
        super().__init__(d_model=width, d_state=state, expand=expand, headdim=headdim,
                         ngroups=1, rope_fraction=.5, is_mimo=False, is_outproj_norm=False)
    def projections(self, token):
        sizes = [self.d_inner, self.d_inner, self.d_state, self.d_state,
                 self.nheads, self.nheads, self.nheads, self.num_rope_angles]
        z, x, B, C, dt, A, trap, angles = self.in_proj(token).split(sizes, -1)
        shape = (*token.shape[:-1], self.nheads, self.headdim)
        return z.reshape(shape), x.reshape(shape), self.B_norm(B), self.C_norm(C), dt, A, trap, angles
    def full(self, tokens):
        z, x, B, C, raw_dt, raw_A, trap, angles = self.projections(tokens)
        delta = F.softplus(raw_dt+self.dt_bias)
        A = -(raw_A.clamp_min(0)+(1-raw_A.clamp_max(0)).reciprocal()).clamp_min(self.A_floor)
        out, *cache = siso_dense(C[:, :, None], B[:, :, None], x, (A*delta).transpose(1, 2), delta.transpose(1, 2),
                                 trap.transpose(1, 2), self.C_bias[:, 0], self.B_bias[:, 0],
                                 angles[:, :, None].expand(-1, -1, self.nheads, -1), self.D, z, return_final_states=True)
        return self.out_proj(out.flatten(2)), tuple(cache)
    def step(self, token, cache):
        angle, state, last_k, last_v = cache
        z, v, B, C, raw_dt, raw_A, raw_trap, raw_angles = self.projections(token)
        delta = F.softplus(raw_dt+self.dt_bias)
        A = -(raw_A.clamp_min(0)+(1-raw_A.clamp_max(0)).reciprocal()).clamp_min(self.A_floor)
        angle = torch.remainder(angle+raw_angles[:, None].tanh()*math.pi*delta[..., None], 2*math.pi)
        q = rotary(C[:, None]+self.C_bias[None, :, 0], angle)
        k = rotary(B[:, None]+self.B_bias[None, :, 0], angle)
        alpha = (A*delta).exp(); trap = raw_trap.sigmoid()
        beta = (1-trap)*delta*alpha; gamma = trap*delta
        state = (alpha[..., None, None]*state+beta[..., None, None]*last_v[..., None]*last_k[..., None, :]
                 +gamma[..., None, None]*v[..., None]*k[..., None, :])
        out = torch.bmm(state.flatten(0, 1), q.flatten(0, 1)[..., None]).reshape_as(v)+self.D[None, :, None]*v
        return self.out_proj((out*F.silu(z)).flatten(1)), (angle, state, k, v)

