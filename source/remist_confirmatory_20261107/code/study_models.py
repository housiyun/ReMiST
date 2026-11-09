"""Frozen adapters for a unified 12-state-output comparison."""
import math, sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(__file__).resolve().parents[2]

def phase(month):
    a = month*(2*math.pi/12)
    return torch.stack([a.sin(), a.cos()], -1)

def make(kind, d, raw, seed, device):
    sys.path.insert(0,str(DATA/'enso_progressive64_ablations_20261103/code'))
    from ablation_model import AblationModel
    from data import backbone_parts
    # A common inherited CMIP backbone is available for every new training seed.
    d = dict(d); d['bases'] = dict(d['bases']); d['innovation'] = dict(d['innovation'])
    d['bases'][seed] = d['bases'][17]; d['innovation'][seed] = d['innovation'][17]
    memory_kind = 'gru' if kind.startswith('gru_') else 'no_ema' if kind.startswith('current_') else 'full'
    m = AblationModel(d, raw, seed, memory_kind)
    static = kind in ('ema_static','current_static','gru_static')
    if kind in ('direct','ordinary'):
        del m.memory_attention, m.context_encoder, m.cross_fusion
        del m.forecast.base, m.forecast.router, m.forecast.residual, m.forecast.raw_tau
        width = 12*8 if kind == 'direct' else 8
        m.simple_head = nn.Sequential(nn.Linear(266,128),nn.Tanh(),nn.Linear(128,width))
        nn.init.zeros_(m.simple_head[-1].weight); nn.init.zeros_(m.simple_head[-1].bias)
        def forward(h, month, images):
            features = torch.cat(m.encode(h,images),-1); x=h[:,-1]
            if kind == 'direct':
                z=torch.cat([features,x,phase(month)],-1)
                return x[:,None]+m.simple_head(z).reshape(-1,12,8)
            outputs=[]
            for k in range(12):
                x=x+m.simple_head(torch.cat([features,x,phase(month+k)],-1))
                outputs.append(x)
            return torch.stack(outputs,1)
    elif kind == 'bone':
        # Remove all unused neural forecast resources; this is an information deletion control.
        base=m.forecast.base
        class Bone(nn.Module):
            def __init__(self,base): super().__init__();self.base=base
            def forward(self,h,month,images):
                x=h[:,-1]; out=[]
                for k in range(12):
                    tendency,_=backbone_parts(self.base,x,month+k);x=x+tendency;out.append(x)
                return torch.stack(out,1)
        return Bone(base).to(device)
    else:
        calibrated = kind in ('neural_calibrated','hybrid_calibrated')
        if calibrated:
            with torch.no_grad(): m.forecast.innovation.copy_(d['increment_scale'])
        if kind == 'neural_calibrated': del m.forecast.base
        def forward(h,month,images):
            f=m.forecast;features=m.encode(h,images);x=h[:,-1]
            memory,a=m._initial_memory(h); out=[]; first=None
            for k in range(12):
                if kind == 'neural_calibrated':
                    tendency=torch.zeros_like(x);season=phase(month+k)
                else: tendency,season=backbone_parts(f.base,x,month+k)
                enhanced,_=m.memory_attention(memory)
                context=m.context_encoder(torch.cat([x,enhanced,season],-1))
                if first is None:first=context
                read_context=first if static else context
                latent,_=m.cross_fusion(read_context,features,F.log_softmax(f.router(read_context),-1))
                delta=2*f.innovation*torch.tanh(f.residual(torch.cat([context,latent],-1)))
                x=x+tendency+delta;out.append(x)
                if memory_kind=='no_ema':memory=x
                elif memory_kind=='gru':memory=m.memory_gru(x,memory)
                else:memory=a*memory+(1-a)*x
            return torch.stack(out,1)
    m.forward=forward
    return m.to(device)
