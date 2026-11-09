"""Location-aware real-case prototypes and exactly additive, bounded RO residuals.

This is a task adaptation inspired by ProtoLNet, not the official full network.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def pool_tokens(x,shape=(3,6,12)):
    # B,T,H,W,D -> fixed spatial/temporal token geometry, same for each encoder.
    return F.adaptive_avg_pool3d(x.permute(0,4,1,2,3),shape).permute(0,2,3,4,1).contiguous()

def location_support(wet,p):
    wet=torch.as_tensor(wet,dtype=torch.float32)
    fraction=F.adaptive_avg_pool2d(wet[None,None],p['token_grid'])[0,0]
    valid=(fraction>=p['minimum_ocean_fraction'])[None].expand(p['temporal_bins'],-1,-1).clone()
    if not valid.any(): raise ValueError('No supported ocean prototype locations')
    return valid

class ConvLSTMEncoder(nn.Module):
    def __init__(self,hidden=8):
        super().__init__(); self.hidden=hidden
        self.gates=nn.Conv2d(1+hidden,4*hidden,3,padding=1)
        self.readout=nn.Conv2d(hidden,hidden,1)
    def forward(self,x):
        # Official Earthformer-compatible SST anomalies/min-max input, N,T,H,W.
        x=F.avg_pool2d(x,2); n,t,h,w=x.shape
        hidden=x.new_zeros(n,self.hidden,h,w); cell=torch.zeros_like(hidden); result=[]
        for j in range(t):
            i,f,o,g=self.gates(torch.cat([x[:,j:j+1],hidden],1)).chunk(4,1)
            cell=torch.sigmoid(f)*cell+torch.sigmoid(i)*torch.tanh(g)
            hidden=torch.sigmoid(o)*torch.tanh(cell)
            result.append(self.readout(hidden).permute(0,2,3,1))
        return pool_tokens(torch.stack(result,1))

class PositionPrototypes(nn.Module):
    def __init__(self,tokens,count=6,source_issues=None,support=None):
        super().__init__()
        if tokens.ndim!=5 or len(tokens)==0: raise ValueError('Training tokens required')
        z=F.normalize(tokens.detach(),dim=-1); flat=z.reshape(-1,z.shape[-1])
        # Deterministic training-only farthest-point coverage; actual source tokens.
        valid=torch.ones(z.shape[1:4],dtype=torch.bool,device=z.device) if support is None else support.to(z.device)
        if tuple(valid.shape)!=tuple(z.shape[1:4]) or not valid.any(): raise ValueError('Invalid spatial support')
        candidates=valid.flatten().repeat(len(z))
        first=int(torch.argmax(flat.square().sum(-1).masked_fill(~candidates,-1)).item()); chosen=[first]
        nearest=(flat-flat[first]).square().sum(-1).masked_fill(~candidates,-1)
        for _ in range(count-1):
            pos=int(nearest.argmax().item()); chosen.append(pos)
            nearest=torch.minimum(nearest,(flat-flat[pos]).square().sum(-1)).masked_fill(~candidates,-1)
        self.register_buffer('prototypes',flat[chosen].clone())
        self.geometry=tuple(z.shape[1:4]); sites=int(np.prod(self.geometry))
        self.register_buffer('valid_locations',valid.flatten().clone())
        self.source_indices=[np.unravel_index(i,z.shape[:-1]) for i in chosen]
        coords=torch.cartesian_prod(*[torch.arange(s,device=z.device) for s in self.geometry]).float()
        locations=[]
        for row in self.source_indices:
            center=coords.new_tensor(row[1:]); sizes=coords.new_tensor([1.,1.5,2.5])
            locations.append(-.5*((coords-center)/sizes).square().sum(-1))
        self.location_logits=nn.Parameter(torch.stack(locations))
        self.sources=[{'training_origin':int(r[0]),'time_bin':int(r[1]),'y':int(r[2]),'x':int(r[3]),
                       'issue':int(source_issues[r[0]]) if source_issues is not None else None} for r in self.source_indices]
    def forward(self,tokens):
        if tuple(tokens.shape[1:4])!=self.geometry: raise ValueError('Prototype geometry mismatch')
        z=F.normalize(tokens,dim=-1).flatten(1,3)
        similarity=torch.einsum('bld,kd->bkl',z,self.prototypes)
        location_weights=self.location_logits.masked_fill(~self.valid_locations[None],-torch.inf).softmax(-1)
        return (similarity*location_weights[None]).sum(-1)

class AdditiveFusion(nn.Module):
    def __init__(self,features,horizon=12,hidden=4,bound=.5):
        super().__init__(); self.features=features; self.bound=bound
        self.terms=nn.ModuleList([nn.Sequential(nn.Linear(1,hidden),nn.Tanh(),nn.Linear(hidden,3)) for _ in range(features)])
        for term in self.terms: nn.init.zeros_(term[-1].weight); nn.init.zeros_(term[-1].bias)
        lead=torch.arange(1,horizon+1,dtype=torch.float32)/horizon
        basis=torch.stack([lead,lead**2,torch.sin(2*math.pi*lead)],-1)
        self.register_buffer('basis',basis/basis.abs().sum(-1,keepdim=True).clamp_min(1.))
    def contributions(self,concepts,ro_features):
        horizon=ro_features.shape[1]
        x=torch.cat([concepts[:,None].expand(-1,horizon,-1),ro_features],-1)
        if x.shape[-1]!=self.features: raise ValueError('Additive feature count mismatch')
        terms=[]
        for k,term in enumerate(self.terms):
            value=x[:,:,k:k+1]
            coefficient=torch.tanh(term(value)-term(torch.zeros_like(value)))
            terms.append(self.bound/self.features*(coefficient*self.basis).sum(-1))
        return torch.stack(terms,-1)
    def forward(self,concepts,ro_features): return self.contributions(concepts,ro_features).sum(-1)

class SpatialFusion(nn.Module):
    def __init__(self,training_tokens,p,issues=None,encoder=None,support=None):
        super().__init__(); self.encoder=encoder
        self.prototype=PositionPrototypes(training_tokens,p['prototypes'],issues,support)
        self.fusion=AdditiveFusion(p['prototypes']+4+3,p['horizon'],p['nam_hidden'],p['residual_bound_C'])
    def concepts(self,tokens,physical): return torch.cat([self.prototype(tokens),physical],-1)
    def forward(self,inputs,physical,ro_features):
        tokens=self.encoder(inputs) if self.encoder is not None else inputs
        concepts=self.concepts(tokens,physical)
        return self.fusion(concepts,ro_features)

def training_loss(paths,residual,target,mask,p):
    pred=paths+residual[:,None]; error=pred.transpose(1,2)[mask]-target[mask][:,None]
    ordered=error.sort(1)[0]; draws=error.shape[1]
    weights=2*torch.arange(1,draws+1,device=paths.device)-draws-1
    crps=(error.abs().mean(1)-(weights*ordered).sum(1)/draws**2).mean()
    complete=mask[:,-1]
    soft=torch.sigmoid((-.5-pred[:,:,-1])/p['soft_cold_temperature_C']).mean(1)
    brier=(soft[complete]-(target[complete,-1]<=-.5).float()).square().mean() if complete.any() else crps*0
    loss=crps+.5*brier+.03*(residual/p['residual_bound_C']).square().mean()
    return loss

def local_residual(train_x,train_y,train_mask,query_x,neighbors=8,ridge=10.,bound=.5):
    """Frozen features with genuine historical support labels, no gradient training."""
    train_x=np.asarray(train_x,dtype=np.float64); query_x=np.asarray(query_x,dtype=np.float64)
    mean=train_x.mean(0); std=train_x.std(0).clip(.1)
    x=(train_x-mean)/std; q=(query_x-mean)/std; out=np.zeros((len(q),train_y.shape[1])); support=[]
    for row,point in enumerate(q):
        distances=((x-point)**2).mean(-1); support_row=[]
        for h in range(train_y.shape[1]):
            valid=np.flatnonzero(train_mask[:,h]); ids=valid[np.argsort(distances[valid])[:neighbors]]
            if not len(ids): raise ValueError('No released support target')
            features=np.c_[np.ones(len(ids)),x[ids]-point]
            reg=np.eye(features.shape[1])*ridge; reg[0,0]=1e-6
            beta=np.linalg.solve(features.T@features+reg,features.T@train_y[ids,h])
            out[row,h]=bound*np.tanh(beta[0]/bound); support_row.append(ids.tolist())
        support.append(support_row)
    return out.astype('float32'),support
