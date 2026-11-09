"""Whole STIR/VMIR/RCER deletions with explicitly minimal output interfaces."""
from common import PARENT
import sys,copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
sys.path.append(str(PARENT/'code'))
from study_models import make as parent_make

def phase(month):
 angle=2*np.pi*month/12
 return torch.stack([angle.sin(),angle.cos()],-1)

def full(packet,seed,device):
 return parent_make('remist',packet['parent'],packet['train']['maps'],seed,device)

def make(kind,packet,seed,device):
 assert kind in ('full','without_stir','without_vmir','without_rcer')
 m=full(packet,seed,device);m.module_ablation=kind
 if kind=='full':return m
 f=m.forecast
 if kind=='without_stir':
  m.absolute_head=copy.deepcopy(f.residual)
  del f.base,f.innovation,f.residual
 elif kind=='without_vmir':
  del f.raw_tau,m.memory_attention
  old=m.context_encoder[0];columns=[*range(8),16,17]
  with torch.random.fork_rng(devices=[]):
   layer=nn.Linear(10,old.out_features,bias=old.bias is not None,device='cpu',dtype=old.weight.dtype)
  layer=layer.to(old.weight.device)
  with torch.no_grad():
   layer.weight.copy_(old.weight[:,columns]);layer.bias.copy_(old.bias)
  m.context_encoder[0]=layer
 else:
  del m.context_encoder,f.router,m.cross_fusion
  old=f.residual[0]
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(int(seed)+300003)
   layer=nn.Linear(82,old.out_features,bias=old.bias is not None,device='cpu',dtype=old.weight.dtype)
  layer=layer.to(old.weight.device)
  with torch.no_grad():layer.bias.copy_(old.bias)
  f.residual[0]=layer
 def forward(h,month,images):
  from data import backbone_parts
  features=m.encode(h,images);x=h[:,-1];out=[]
  if kind!='without_vmir':memory,a=m._initial_memory(h)
  if kind=='without_rcer':pooled=torch.stack(features,1).mean(1)
  for step in range(12):
   if kind=='without_stir':season=phase(month+step)
   else:tendency,season=backbone_parts(f.base,x,month+step)
   if kind=='without_vmir':
    context=m.context_encoder(torch.cat([x,season],-1))
   else:
    enhanced,_=m.memory_attention(memory)
    if kind!='without_rcer':context=m.context_encoder(torch.cat([x,enhanced,season],-1))
   if kind=='without_rcer':
    head_input=torch.cat([x,enhanced,season,pooled],-1)
   else:
    latent,_=m.cross_fusion(context,features,F.log_softmax(f.router(context),-1))
    head_input=torch.cat([context,latent],-1)
   if kind=='without_stir':x=m.absolute_head(head_input)
   else:x=x+tendency+2*f.innovation*torch.tanh(f.residual(head_input))
   if kind!='without_vmir':memory=a*memory+(1-a)*x
   out.append(x)
  return torch.stack(out,1)
 m.forward=forward
 return m

def removed_prefixes(kind):
 return {'full':(), 'without_stir':('forecast.base.','forecast.residual.'), 'without_vmir':('forecast.raw_tau','memory_attention.'), 'without_rcer':('context_encoder.','forecast.router.','cross_fusion.')}[kind]
