"""Final progressive64 ablations. Numerical use is restricted to HSY-A6000.

Every variant first constructs the unchanged parent full model. Ablations then
remove or select existing parameters, preserving paired common initialization.
The eight-coordinate GRU control is newly initialized in an isolated RNG scope.
No historical experiment source or checkpoint is modified.
"""
from pathlib import Path
import math
import sys


import torch
from torch import nn
from torch.nn import functional as F

R = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = R.parent / 'enso_progressive64_training_20261102'
sys.path.insert(0, str(BASE_MODEL_DIR / 'code'))
from attention_model import AttentionWholeModel, role as parent_role, P, PREV
from data import backbone_parts

KINDS = (
    'full', 'no_ema', 'fixed_ema', 'gru', 'raw_ema', 'static_fusion',
    'no_router', 'without_tcn', 'without_mamba', 'without_dpot',
    'without_itransformer', 'no_ro',
)
BRANCHES = ('TCN', 'Mamba3', 'DPOT', 'iTransformer')
DROP_INDEX = {
    'without_tcn': 0, 'without_mamba': 1,
    'without_dpot': 2, 'without_itransformer': 3,
}
INITIAL_TAU = (6., 6., 6., 6., 6., 12., 12., 6.)


def _select_linear_rows(module, rows):
    """Copy selected outputs without consuming the caller's random stream."""
    with torch.random.fork_rng(devices=[]):
        selected = nn.Linear(
            module.in_features, len(rows), bias=module.bias is not None,
            device=module.weight.device, dtype=module.weight.dtype,
        )
    with torch.no_grad():
        selected.weight.copy_(module.weight[list(rows)])
        if module.bias is not None:
            selected.bias.copy_(module.bias[list(rows)])
    return selected


class AblationModel(AttentionWholeModel):
    """Same input/return API as the parent, with configurable output count.

    ``forecast_steps=12`` is the primary training rollout. Setting it to 25
    extends the identical recursion for index center leads 0--24. History
    encoders are evaluated once, including for no_ema and GRU controls.
    """
    forecast_steps = 12

    def __init__(self, d, raw, seed, kind='full'):
        if kind not in KINDS:
            raise ValueError(f'Unknown ablation {kind!r}; expected {KINDS}')
        # Always request full training scope from the historical constructor.
        super().__init__(d, raw, seed, 'full')
        self.kind = kind
        self.original_full_parameters = sum(p.numel() for p in self.parameters())
        self.active_indices = tuple(i for i in range(4) if i != DROP_INDEX.get(kind))
        self.active_branches = tuple(BRANCHES[i] for i in self.active_indices)

        if kind == 'fixed_ema':
            # Copy the transformed initial values exactly, not a new trainable tau.
            self.register_buffer('fixed_tau', self.forecast.tau.detach().clone())
            del self.forecast.raw_tau
        elif kind in ('no_ema', 'gru'):
            del self.forecast.raw_tau
            if kind == 'gru':
                with torch.random.fork_rng(devices=[]):
                    # GRU is created on CPU; do not reseed CUDA generators.
                    torch.random.default_generator.manual_seed(int(seed) + 100003)
                    self.memory_gru = nn.GRUCell(8, 8)

        if kind == 'raw_ema':
            del self.memory_attention
        if kind == 'no_router':
            del self.forecast.router
        if kind == 'no_ro':
            # Keep calendar conditioning and the inherited innovation bound.
            # Only the explicit backbone tendency and its parameters disappear.
            del self.forecast.base

        if kind in DROP_INDEX:
            # Remove the branch identity and router output associated with it.
            self.cross_fusion.branch_id = nn.Parameter(
                self.cross_fusion.branch_id.detach()[list(self.active_indices)].clone()
            )
            self.forecast.router = _select_linear_rows(
                self.forecast.router, self.active_indices
            )
            name = {
                0: 'tcn', 1: 'mamba', 2: 'spatial_projection', 3: 'itransformer'
            }[DROP_INDEX[kind]]
            delattr(self, name)
            if kind == 'without_dpot':
                # This intentionally removes the sole SST-map information path.
                # It is a branch/information deletion, not a same-information or
                # parameter-matched test of public pretraining.
                del self.adapter
                del self.forecast.prototype
                for buffer in ('spatial_mean', 'spatial_sd', 'sim_mean', 'sim_sd'):
                    delattr(self.forecast, buffer)
                del self.source_images
                del self.source_positions
                del self.source_origins

    @property
    def memory_tau(self):
        """Actual EMA scales, or None for current-read and GRU controls."""
        if self.kind in ('no_ema', 'gru'):
            return None
        if self.kind == 'fixed_ema':
            return self.fixed_tau
        return self.forecast.tau

    @property
    def memory_mode(self):
        return {
            'no_ema': 'current_state_read_without_explicit_memory',
            'gru': 'eight_latent_GRU_coordinates_not_physical_EMA',
            'fixed_ema': 'fixed_initial_variable_EMA',
            'raw_ema': 'learned_variable_EMA_without_read_enhancement',
        }.get(self.kind, 'learned_variable_EMA_with_read_only_enhancement')

    def metadata(self):
        tau = self.memory_tau
        return {
            'kind': self.kind,
            'active_branches': list(self.active_branches),
            'active_indices': list(self.active_indices),
            'memory_mode': self.memory_mode,
            'memory_tau_months': None if tau is None else tau.detach().cpu().tolist(),
            'forecast_steps': int(self.forecast_steps),
            'original_full_parameters': int(self.original_full_parameters),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'map_information_removed': self.kind == 'without_dpot',
            'backbone_tendency_removed': self.kind == 'no_ro',
            'spatial_information_scope': (
                'No image path remains; branch/information deletion, not a '
                'same-information or isolated-pretraining comparison'
                if self.kind == 'without_dpot' else
                'Same original SST image packet and public initialization retained'
            ),
            'static_fusion_scope': (
                'Initial context fixes both queries and router prior; current '
                'context still directly conditions the residual head'
                if self.kind == 'static_fusion' else None
            ),
        }

    def _spatial_feature(self, images):
        # Kept in parent operation order to preserve numerical parity.
        f = self.forecast
        n = len(images)
        tokens = self.adapter(torch.cat([images, self.source_images], 0))
        z = tokens[:n]
        src = F.normalize(tokens[n:], dim=-1)
        anchors = torch.stack([src[a, b, c, e] for a, b, c, e in self.source_positions])
        zn = F.normalize(z, dim=-1)
        flat = zn.flatten(1, 3)
        loc = f.prototype.location_logits.masked_fill(
            ~f.prototype.valid_locations[None], -torch.inf
        ).softmax(-1)
        sim = (torch.einsum('bld,kd->bkl', flat, anchors) * loc[None]).sum(-1)
        pooled = zn[:, f.prototype.valid_locations.reshape(f.prototype.geometry)].mean(1)
        pooled = ((pooled - f.spatial_mean) / f.spatial_sd).clamp(-5, 5)
        sim = ((sim - f.sim_mean) / f.sim_sd).clamp(-5, 5)
        return self.spatial_projection(pooled, sim)

    def encode(self, h, images):
        if len(self.active_indices) == 4:
            return super().encode(h, images)
        spatial = self._spatial_feature(images) if 2 in self.active_indices else None
        result = []
        for i in self.active_indices:
            if i == 0:
                result.append(self.tcn(h))
            elif i == 1:
                result.append(self.mamba(h))
            elif i == 2:
                result.append(spatial)
            else:
                result.append(self.itransformer(h))
        return result

    def _initial_memory(self, h):
        if self.kind == 'no_ema':
            return h[:, -1], None
        memory = h.new_zeros(len(h), 8)
        if self.kind == 'gru':
            for past in h.unbind(1):
                memory = self.memory_gru(past, memory)
            return memory, None
        a = torch.exp(-1 / self.memory_tau)
        for past in h.unbind(1):
            memory = a * memory + (1 - a) * past
        return memory, a

    def forward(self, h, month, images, diagnostics=False):
        f = self.forecast
        features = self.encode(h, images)
        x = h[:, -1]
        memory, a = self._initial_memory(h)
        paths, deltas, weights, memory_weights, priors = [], [], [], [], []
        raw_trace, read_trace, contexts, fusion_contexts, latents = [memory], [], [], [], []
        first_context = None
        for step in range(int(self.forecast_steps)):
            if self.kind == 'no_ro':
                angle = 2 * math.pi * (month + step) / 12
                season = torch.stack([angle.sin(), angle.cos()], -1)
                tendency = torch.zeros_like(x)
            else:
                tendency, season = backbone_parts(f.base, x, month + step)
            if self.kind == 'raw_ema':
                enhanced, mw = memory, None
            else:
                enhanced, mw = self.memory_attention(memory)
            context = self.context_encoder(torch.cat([x, enhanced, season], -1))
            if first_context is None:
                first_context = context
            fusion_context = first_context if self.kind == 'static_fusion' else context
            if self.kind == 'no_router':
                log_prior = context.new_zeros(len(context), len(features))
                prior = log_prior.softmax(-1)
            else:
                logits = f.router(fusion_context)
                log_prior = F.log_softmax(logits, -1)
                prior = logits.softmax(-1)
            latent, w = self.cross_fusion(fusion_context, features, log_prior)
            delta = 2 * f.innovation * torch.tanh(
                f.residual(torch.cat([context, latent], -1))
            )
            x = x + tendency + delta
            if self.kind == 'no_ema':
                memory = x
            elif self.kind == 'gru':
                memory = self.memory_gru(x, memory)
            else:
                memory = a * memory + (1 - a) * x
            paths.append(x)
            deltas.append(delta)
            if diagnostics:
                weights.append(w)
                if mw is not None:
                    memory_weights.append(mw)
                priors.append(prior)
                raw_trace.append(memory)
                read_trace.append(enhanced)
                contexts.append(context)
                fusion_contexts.append(fusion_context)
                latents.append(latent)
        path = torch.stack(paths, 1)
        if not diagnostics:
            return path
        return path, torch.stack(deltas, 1), {
            'router_prior': torch.stack(priors, 1),
            'fusion': torch.stack(weights, 1),
            'memory': torch.stack(memory_weights, 1) if memory_weights else None,
            'branch_tokens': torch.stack(features, 1),
            'final_raw_memory': memory,
            'raw_memory_trajectory': torch.stack(raw_trace, 1),
            'enhanced_memory': torch.stack(read_trace, 1),
            'contexts': torch.stack(contexts, 1),
            'fusion_contexts': torch.stack(fusion_contexts, 1),
            'fusion_latent': torch.stack(latents, 1),
            'memory_mode': self.memory_mode,
            'active_branches': self.active_branches,
        }


def role(name):
    if name.startswith('memory_gru.'):
        return 'GRU_memory'
    return parent_role(name)
