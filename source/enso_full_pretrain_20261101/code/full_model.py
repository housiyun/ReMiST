from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
P=Path(__file__).resolve().parents[2]/'enso_ema_ablation_20261031'
sys.path.insert(0,str(P/'code'))
from model import Integrated as Forecast
from engine import base_object
from data import backbone_parts
sys.path.insert(0,str(P.parent/'enso_route_b_mechanism_20261029/operator'))
from dpot_adapter import DPOTTinyAdapter

class WholeModel(nn.Module):
 def __init__(self,d,raw,seed,kind):
  super().__init__();self.kind=kind
  self.adapter=DPOTTinyAdapter(seed=9929)
  state=torch.load(P/'rebuild_inputs/dpot_pretrained_interface.pt',map_location='cpu',weights_only=False)
  for k,v in state.items():getattr(self.adapter,k).load_state_dict(v,strict=True)
  # Unused public reconstruction/classification modules are not forecast components.
  del self.adapter.net.cls_head;del self.adapter.net.out_layer;del self.adapter.recovery_adapter
  base=base_object(d['bases'][seed]);del base.tau_raw
  torch.manual_seed(seed);self.forecast=Forecast(base,d['tt'],d['tr']['issue'].tolist(),d['support'],d['innovation'][seed],'ema_all')
  origins=list(dict.fromkeys(int(x[0]) for x in self.forecast.prototype.source_indices));self.source_origins=origins
  self.register_buffer('source_images',raw[origins].clone());self.source_positions=[(origins.index(int(x[0])),int(x[1]),int(x[2]),int(x[3])) for x in self.forecast.prototype.source_indices]
  for q in self.adapter.parameters():q.requires_grad_(kind=='full')
  for q in self.forecast.base.parameters():q.requires_grad_(kind=='full')
 def forward(self,h,month,images,diagnostics=False):
  f=self.forecast;n=len(h);tokens=self.adapter(torch.cat([images,self.source_images],0));z=tokens[:n];src=F.normalize(tokens[n:],dim=-1)
  anchors=torch.stack([src[a,b,c,e] for a,b,c,e in self.source_positions]);zn=F.normalize(z,dim=-1);flat=zn.flatten(1,3)
  loc=f.prototype.location_logits.masked_fill(~f.prototype.valid_locations[None],-torch.inf).softmax(-1)
  sim=(torch.einsum('bld,kd->bkl',flat,anchors)*loc[None]).sum(-1)
  pooled=zn[:,f.prototype.valid_locations.reshape(f.prototype.geometry)].mean(1);pooled=((pooled-f.spatial_mean)/f.spatial_sd).clamp(-5,5);sim=((sim-f.sim_mean)/f.sim_sd).clamp(-5,5)
  s=f.spatial(torch.cat([pooled,sim],-1));t=f.tcn_norm(f.tcn_sequence(h)[:,-1]).tanh();m,_=f.mamba.full(f.mamba_norm(f.mamba_input(h)));m=f.mamba_outnorm(m[:,-1]).tanh();features=[t,m,s]
  x=h[:,-1];a=torch.exp(-1/f.tau);memory=x.new_zeros(len(x),8)
  for past in h.unbind(1):memory=a*memory+(1-a)*past
  paths=[];deltas=[];weights=[]
  for step in range(12):
   tendency,season=backbone_parts(f.base,x,month+step);context=torch.cat([x,memory,season],-1);w=f.router(context).softmax(-1);latent=torch.cat([v*w[:,j:j+1] for j,v in enumerate(features)],-1)
   delta=2*f.innovation*torch.tanh(f.residual(torch.cat([context,latent],-1)));x=x+tendency+delta;memory=a*memory+(1-a)*x;paths.append(x);deltas.append(delta);weights.append(w)
  path=torch.stack(paths,1)
  return (path,torch.stack(deltas,1),torch.stack(weights,1)) if diagnostics else path

def role(name):
 if name.startswith('adapter.'):return 'DPOT_encoder_adapter'
 if name.startswith('forecast.base.'):return 'RO'
 if name.startswith('forecast.tcn'):return 'TCN'
 if name.startswith('forecast.mamba'):return 'Mamba3'
 if name.startswith('forecast.raw_tau'):return 'EMA'
 if name.startswith('forecast.router'):return 'router'
 if name.startswith('forecast.prototype'):return 'prototype_locations'
 if name.startswith('forecast.spatial'):return 'spatial_projection'
 if name.startswith('forecast.residual'):return 'residual'
 raise ValueError(name)
