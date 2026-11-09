"""Honest SST/physical-field adapter around a strictly frozen official DPOT Tiny.

The four source interface channels are learned latent coordinates, not density,
velocity, pressure, or measured extra variables. Twelve observed history frames
are causally resampled into ten latent history slots. The original temporal
aggregator produces ONE spatial feature map; no temporal tokens are copied.

Native 24x48 inputs are processed by the frozen 8x8 patch kernels. We resize the
source positional embedding as a functional value, without changing its stored
weights. This is a task/input adaptation, not an official paper reproduction.
"""
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from fetch import ROOT,EXPECTED_SHA256,digest

CONFIG=dict(img_size=128,patch_size=8,mixing_type='afno',in_channels=4,out_channels=4,
            in_timesteps=10,out_timesteps=1,n_blocks=4,embed_dim=512,out_layer_dim=32,
            depth=4,modes=32,mlp_ratio=1,n_cls=12,normalize=False)

def source_module():
    path=ROOT/'provenance/source/models/dpot.py'
    if not path.exists(): raise FileNotFoundError('Run operator/fetch.py first; official DPOT source is missing')
    manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    if digest(path)!=manifest['source_files_sha256']['models/dpot.py']:
        raise ValueError('Official DPOT source bytes changed')
    spec=importlib.util.spec_from_file_location('route_b_official_dpot',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class OrderedHistoryResampler(nn.Module):
    """Ten ordered convex mixtures of adjacent observed monthly frames.

    The slot order remains chronological and no observation after the issue's
    last available month can enter. Slot coordinates are representation indices,
    not claims of new submonthly observations.
    """
    def __init__(self,history=12,slots=10):
        super().__init__(); self.history=history; self.slots=slots
        position=torch.linspace(0,history-1,slots)
        lower=position.floor().long(); upper=position.ceil().long()
        matrix=torch.zeros(slots,history)
        for j in range(slots):
            if lower[j]==upper[j]: matrix[j,lower[j]]=1
            else:
                matrix[j,lower[j]]=upper[j]-position[j]
                matrix[j,upper[j]]=position[j]-lower[j]
        self.register_buffer('allowed',matrix>0)
        self.logits=nn.Parameter(matrix.clamp_min(1e-8).log())
    def weights(self): return self.logits.masked_fill(~self.allowed,-torch.inf).softmax(-1)
    def forward(self,x):
        if x.ndim!=5 or x.shape[1]!=self.history: raise ValueError(f'Expected B,{self.history},C,H,W history')
        return torch.einsum('st,btchw->bschw',self.weights(),x)

class DPOTTinyAdapter(nn.Module):
    def __init__(self,in_channels=1,history=12,checkpoint=None,train_adapters=True,
                 random_structure_test_only=False,seed=9929):
        super().__init__()
        torch.manual_seed(seed)
        self.net=source_module().DPOTNet(**CONFIG)
        self.in_channels=in_channels; self.history=history; self.patch_size=8
        self.audit={'source_commit':json.loads((ROOT/'provenance/source_pin.json').read_text())['commit'],
                    'pretrained_weights_loaded':False,'official_config':CONFIG,
                    'adapter_semantics':'learned latent source-interface channels, no physical variable substitution',
                    'history_adapter':'ordered adjacent convex monthly mixtures; all 12 frames within the available past',
                    'native_spatial_input':'frozen patch projection on original grid, functional bicubic position-embedding resize',
                    'token_temporal_extent':1,'seed':seed}
        if random_structure_test_only:
            self.audit.update(status='RANDOM_ARCHITECTURE_TEST_ONLY',eligible_for_skill_claim=False)
        else:
            checkpoint=Path(checkpoint) if checkpoint is not None else ROOT/'weights/model_Ti.pth'
            if not checkpoint.exists(): raise FileNotFoundError('Official DPOT Tiny checkpoint missing; random fallback is forbidden')
            actual=digest(checkpoint)
            if actual!=EXPECTED_SHA256: raise ValueError('Official published DPOT Tiny SHA256 mismatch')
            # The official, hash-verified checkpoint also contains argparse.Namespace
            # and optimizer state. Its model tensors are loaded strictly below.
            raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
            state=raw['model']
            self.net.load_state_dict(state,strict=True)
            copied=self.net.state_dict()
            if set(copied)!=set(state) or any(not torch.equal(copied[k],v.cpu()) for k,v in state.items()):
                raise ValueError('Pretrained state tensor readback mismatch')
            self.audit.update(status='PRETRAINED_BODY_VERIFIED_ADAPTER_UNFITTED',eligible_for_skill_claim=False,
                              pretrained_weights_loaded=True,checkpoint_sha256=actual,
                              strict_state_tensors=len(state),all_state_tensors_equal=True)
        for parameter in self.net.parameters(): parameter.requires_grad_(False)
        self.net.eval()
        self.history_adapter=OrderedHistoryResampler(history,10)
        self.channel_adapter=nn.Conv2d(in_channels,4,3,padding=1)
        self.recovery_adapter=nn.Conv2d(4,in_channels,1)
        nn.init.normal_(self.channel_adapter.weight,mean=0,std=.1/math.sqrt(in_channels*9))
        nn.init.zeros_(self.channel_adapter.bias)
        nn.init.normal_(self.recovery_adapter.weight,mean=0,std=.1)
        nn.init.zeros_(self.recovery_adapter.bias)
        if not train_adapters:
            for module in (self.history_adapter,self.channel_adapter,self.recovery_adapter):
                for parameter in module.parameters(): parameter.requires_grad_(False)
        self.audit.update(frozen_source_parameters=sum(p.numel() for p in self.net.parameters()),
                          new_adapter_parameters=sum(p.numel() for n,p in self.named_parameters() if not n.startswith('net.')),
                          trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def train(self,mode=True):
        super().train(mode); self.net.eval(); return self

    def source_state_hash(self):
        import hashlib
        h=hashlib.sha256()
        for key,value in self.net.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def encode_map(self,x):
        if x.ndim==4: x=x.unsqueeze(2)
        if x.ndim!=5 or x.shape[1]!=self.history or x.shape[2]!=self.in_channels:
            raise ValueError('Expected B,T,C,H,W or single-channel B,T,H,W')
        if not torch.isfinite(x).all(): raise ValueError('NaN/Inf must be handled by the audited neural-input mask policy')
        b,_,_,h,w=x.shape
        if h%8 or w%8 or h<8 or w<8: raise ValueError('Native grid height/width must be positive multiples of frozen patch size 8')
        slot=self.history_adapter(x); t=slot.shape[1]
        latent=self.channel_adapter(slot.reshape(b*t,self.in_channels,h,w))
        latent=latent.reshape(b,t,4,h,w).permute(0,3,4,1,2).contiguous()
        return self.encode_source_latents(latent)

    def encode_source_latents(self,latent):
        """Exact source encoder algebra, with functional positional resizing.

        Public for canonical-grid equivalence checks; these four channels are
        explicitly source-interface latents, not fabricated measured variables.
        """
        if latent.ndim!=5 or latent.shape[-2:]!=(10,4): raise ValueError('Expected B,H,W,10,4 source-interface latents')
        b,h,w,t,_=latent.shape
        grid=self.net.get_grid_3d(latent)
        projected=torch.cat([latent,grid],-1).permute(0,3,4,1,2).reshape(b*t,7,h,w)
        # Applying the exact frozen projection avoids only PatchEmbed's fixed-size
        # assertion. Kernel/stride/weights remain untouched and are audited.
        embedded=self.net.patch_embed.proj(projected)
        gh,gw=embedded.shape[-2:]
        positional=F.interpolate(self.net.pos_embed,size=(gh,gw),mode='bicubic',align_corners=False)
        embedded=embedded+positional
        embedded=embedded.reshape(b,t,512,gh,gw).permute(0,3,4,1,2)
        aggregated=self.net.time_agg_layer(embedded).permute(0,3,1,2).contiguous()
        for block in self.net.blocks: aggregated=block(aggregated)
        if not torch.isfinite(aggregated).all(): raise FloatingPointError('Nonfinite adapted DPOT features')
        return aggregated

    def forward(self,x):
        feature=self.encode_map(x)
        return feature.permute(0,2,3,1)[:,None].contiguous()

    def predict_next(self,x):
        feature=self.encode_map(x)
        latent_prediction=self.net.out_layer(feature)
        return self.recovery_adapter(latent_prediction)

    def mark_adapter_fitted(self,metadata):
        if not self.audit['pretrained_weights_loaded']: raise ValueError('Random architecture cannot become a pretrained claim')
        self.audit.update(status='PRETRAINED_WITH_TRAIN_DOMAIN_ADAPTER',adapter_fit=metadata)

    def adapter_state(self):
        return {name:{k:v.detach().cpu() for k,v in module.state_dict().items()} for name,module in
                [('history_adapter',self.history_adapter),('channel_adapter',self.channel_adapter),('recovery_adapter',self.recovery_adapter)]}

    def load_adapter_checkpoint(self,path):
        if not self.audit['pretrained_weights_loaded']: raise ValueError('Verified source weights required before fitted adapters')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['published_source_sha256']!=EXPECTED_SHA256 or saved['frozen_body_state_hash']!=self.source_state_hash():
            raise ValueError('Adapter source provenance mismatch')
        if saved['in_channels']!=self.in_channels or saved['history']!=self.history:
            raise ValueError('Adapter input/history mismatch')
        for name in ('history_adapter','channel_adapter','recovery_adapter'):
            getattr(self,name).load_state_dict(saved['adapters'][name],strict=True)
        self.mark_adapter_fitted(saved['fit_metadata'])
        return saved

