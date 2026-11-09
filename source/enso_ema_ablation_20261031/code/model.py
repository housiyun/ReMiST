"""EMA-channel ablations; preserve original encoder and common head initialization."""
import torch
from torch import nn
from original_model import Integrated as Original, predict
from data import backbone_parts
class Integrated(Original):
 def __init__(self,base,tokens,issues,support,innovation,kind):
  super().__init__(base,tokens,issues,support,innovation,'integrated')
  self.kind=kind
  channels=[] if kind.startswith('ema_none') else ([4,5] if kind.startswith('ema_h300') else list(range(8)))
  self.channels=channels;self.padded=kind.endswith('_padded');self.slots=8 if self.padded else len(channels)
  if kind=='ema_h300':return
  oldrouter=self.router[0];oldhead=self.residual[0]
  # Use a separate generator for additional columns, leaving common encoder tensors paired.
  g=torch.Generator().manual_seed(torch.initial_seed()+100000)
  rw=torch.empty(12,18).uniform_(-12**-.5,12**-.5,generator=g)
  hw=torch.empty(32,66).uniform_(-60**-.5,60**-.5,generator=g)
  with torch.no_grad():
   rw[:,:8]=oldrouter.weight[:,:8];rw[:,12:14]=oldrouter.weight[:,8:10];rw[:,16:]=oldrouter.weight[:,10:]
   hw[:,:8]=oldhead.weight[:,:8];hw[:,12:14]=oldhead.weight[:,8:10];hw[:,16:18]=oldhead.weight[:,10:12];hw[:,18:]=oldhead.weight[:,12:]
   selected=list(range(8))+(list(range(8,16)) if self.padded else [8+j for j in channels])+[16,17]
   r=nn.Linear(10+self.slots,12);r.weight.copy_(rw[:,selected]);r.bias.copy_(oldrouter.bias);self.router[0]=r
   q=nn.Linear(58+self.slots,32);q.weight.copy_(hw[:,selected+list(range(18,66))]);q.bias.copy_(oldhead.bias);self.residual[0]=q
  tau=torch.tensor([6.,6.,6.,6.,6.,12.,12.,6.]);tau=tau[channels]
  self.raw_tau=nn.Parameter(torch.logit((tau-.5)/35.5))
 def context(self,x,memory,season):
  if self.padded:
   full=x.new_zeros(len(x),8)
   if self.channels:full[:,self.channels]=memory
   memory=full
  return torch.cat([x,memory,season],-1)
 def forward(self,h,month,tokens,override_zero=False,drop=None,diagnostics=False):
  features=self.encode(h,tokens,drop);x=h[:,-1];a=torch.exp(-1/self.tau);memory=x.new_zeros(len(x),len(self.channels))
  for past in h.unbind(1):memory=a*memory+(1-a)*past[:,self.channels]
  paths=[];responses=[];weights=[]
  for step in range(12):
   tendency,season=backbone_parts(self.base,x,month+step);context=self.context(x,memory,season);w=self.router(context).softmax(-1)
   latent=torch.cat([v*w[:,j:j+1] for j,v in enumerate(features)],-1);delta=2*self.innovation*torch.tanh(self.residual(torch.cat([context,latent],-1)))
   if override_zero:delta=torch.zeros_like(delta)
   x=x+tendency+delta;paths.append(x);responses.append(delta);weights.append(w);memory=a*memory+(1-a)*x[:,self.channels]
  paths=torch.stack(paths,1)
  return (paths,torch.stack(responses,1),torch.stack(weights,1)) if diagnostics else paths
