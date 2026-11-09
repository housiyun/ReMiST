"""Joint seasonal-RO + causal TCN + Mamba3 + frozen-DPOT-feature residual."""
import torch,numpy as np
from torch import nn
from torch.nn import functional as F
from data import backbone_parts
import prototypes as parts
from mamba3_model import StreamingMamba3
KINDS=['integrated','without_tcn','without_mamba3','without_dpot','thermal_mlp']
class Integrated(nn.Module):
 def __init__(self,base,training_tokens,issues,support,innovation,kind='integrated'):
  super().__init__();self.base=base;self.kind=kind;self.use_tcn=kind not in ('without_tcn','thermal_mlp');self.use_mamba=kind not in ('without_mamba3','thermal_mlp');self.use_dpot=kind not in ('without_dpot','thermal_mlp')
  self.register_buffer('innovation',torch.as_tensor(innovation).float());self.raw_tau=nn.Parameter(torch.logit((torch.tensor([6.,12.])-.5)/35.5))
  self.tcn=nn.ModuleList([nn.Conv1d(8,16,3,padding=2),nn.Conv1d(16,16,3,dilation=2,padding=4)]);self.tcn_norm=nn.LayerNorm(16)
  self.mamba_input=nn.Linear(8,16);self.mamba_norm=nn.LayerNorm(16);self.mamba=StreamingMamba3(width=16,state=8);self.mamba_outnorm=nn.LayerNorm(16)
  self.prototype=parts.PositionPrototypes(training_tokens,6,issues,support=support)
  with torch.no_grad():
   pooled=F.normalize(training_tokens,dim=-1)[:,support].mean(1);sim=self.prototype(training_tokens);self.register_buffer('spatial_mean',pooled.mean(0));self.register_buffer('spatial_sd',pooled.std(0,unbiased=False).clamp_min(.002));self.register_buffer('sim_mean',sim.mean(0));self.register_buffer('sim_sd',sim.std(0,unbiased=False).clamp_min(.01))
  self.spatial=nn.Sequential(nn.Linear(518,16),nn.Tanh());self.router=nn.Sequential(nn.Linear(12,12),nn.Tanh(),nn.Linear(12,3));self.residual=nn.Sequential(nn.Linear(60,32),nn.Tanh(),nn.Linear(32,8))
  nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
  for modules,active in [(list(self.tcn)+[self.tcn_norm],self.use_tcn),([self.mamba_input,self.mamba_norm,self.mamba,self.mamba_outnorm],self.use_mamba),([self.prototype,self.spatial],self.use_dpot)]:
   if not active:
    for module in modules:
     for p in module.parameters():p.requires_grad_(False)
 @property
 def tau(self):return .5+35.5*self.raw_tau.sigmoid()
 def tcn_sequence(self,h):
  n=h.shape[1];v=h.transpose(1,2)
  for conv in self.tcn:v=F.gelu(conv(v)[...,:n])
  return v.transpose(1,2)
 def encode(self,h,tokens,drop=None):
  zero=h.new_zeros(len(h),16)
  t=self.tcn_norm(self.tcn_sequence(h)[:,-1]).tanh() if self.use_tcn else zero
  if self.use_mamba:
   z,_=self.mamba.full(self.mamba_norm(self.mamba_input(h)));m=self.mamba_outnorm(z[:,-1]).tanh()
  else:m=zero
  if self.use_dpot:
   pooled=F.normalize(tokens,dim=-1)[:,self.prototype.valid_locations.reshape(self.prototype.geometry)].mean(1);pooled=((pooled-self.spatial_mean)/self.spatial_sd).clamp(-5,5);sim=((self.prototype(tokens)-self.sim_mean)/self.sim_sd).clamp(-5,5);s=self.spatial(torch.cat([pooled,sim],-1))
  else:s=zero
  result=[t,m,s]
  if drop is not None:result[drop]=zero
  return result
 def forward(self,h,month,tokens,override_zero=False,drop=None,diagnostics=False):
  features=self.encode(h,tokens,drop);x=h[:,-1];a=torch.exp(-1/self.tau);memory=x.new_zeros(len(x),2)
  for past in h.unbind(1):memory=a*memory+(1-a)*past[:,[4,5]]
  paths=[];responses=[];weights=[]
  for step in range(12):
   tendency,season=backbone_parts(self.base,x,month+step);context=torch.cat([x,memory,season],-1);logits=self.router(context);active=torch.tensor([self.use_tcn,self.use_mamba,self.use_dpot],device=x.device)
   if active.any():w=logits.masked_fill(~active,-torch.inf).softmax(-1)
   else:w=torch.zeros_like(logits)
   latent=torch.cat([v*w[:,j:j+1] for j,v in enumerate(features)],-1);delta=2*self.innovation*torch.tanh(self.residual(torch.cat([context,latent],-1)))
   if override_zero:delta=torch.zeros_like(delta)
   x=x+tendency+delta;paths.append(x);responses.append(delta);weights.append(w);memory=a*memory+(1-a)*x[:,[4,5]]
  paths=torch.stack(paths,1)
  return (paths,torch.stack(responses,1),torch.stack(weights,1)) if diagnostics else paths

@torch.no_grad()
def predict(m,b,features,diagnostics=False):
 m.eval();out=[];delta=[];weights=[]
 for i in range(0,len(b['issue']),128):
  q=m(torch.tensor(b['history'][i:i+128]),torch.tensor(b['month'][i:i+128],dtype=torch.float32),features[i:i+128],diagnostics=diagnostics)
  if diagnostics:v,d,w=q;out.append(v.numpy());delta.append(d.numpy());weights.append(w.numpy())
  else:out.append(q.numpy())
 return (np.concatenate(out),np.concatenate(delta),np.concatenate(weights)) if diagnostics else np.concatenate(out)
