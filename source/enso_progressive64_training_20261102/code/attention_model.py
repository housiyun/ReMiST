"""Progressive-width four-branch candidate; all numerical validation runs remotely.
Uses the existing pinned Mamba3 SISO reference implementation, not fused/MIMO kernels.
"""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
PREV=Path(__file__).resolve().parents[2]/'enso_full_pretrain_20261101'
sys.path.insert(0,str(PREV/'code'))
from full_model import WholeModel,role as base_role,P
from data import backbone_parts
from mamba3_model import StreamingMamba3


def progressive_mlp(dimensions):
 layers=[]
 for i,(din,dout) in enumerate(zip(dimensions[:-1],dimensions[1:])):
  layers.extend([nn.Linear(din,dout),nn.LayerNorm(dout),nn.GELU()])
 return nn.Sequential(*layers)


class CausalTCNStage(nn.Module):
 def __init__(self,din,dout,dilation):
  super().__init__();self.left=2*dilation;self.conv=nn.Conv1d(din,dout,3,dilation=dilation);self.norm=nn.LayerNorm(dout)
 def forward(self,x):
  z=self.conv(F.pad(x.transpose(1,2),(self.left,0))).transpose(1,2)
  return F.gelu(self.norm(z))


class ProgressiveTCN(nn.Module):
 def __init__(self):
  super().__init__();self.stages=nn.ModuleList([CausalTCNStage(8,16,1),CausalTCNStage(16,32,2),CausalTCNStage(32,64,4)])
 def sequence(self,h):
  for stage in self.stages:h=stage(h)
  return h
 def forward(self,h):return self.sequence(h)[:,-1].tanh()


class MambaStage(nn.Module):
 def __init__(self,din,dout):
  super().__init__();self.projection=nn.Linear(din,dout);self.input_norm=nn.LayerNorm(dout);self.ssm=StreamingMamba3(width=dout,state=8,expand=2,headdim=4);self.output_norm=nn.LayerNorm(dout)
 def forward(self,x):
  z=self.projection(x);u,_=self.ssm.full(self.input_norm(z));return F.gelu(self.output_norm(z+u))


class ProgressiveMamba(nn.Module):
 def __init__(self):
  super().__init__();self.stages=nn.ModuleList([MambaStage(8,16),MambaStage(16,32),MambaStage(32,64)])
 def sequence(self,h):
  for stage in self.stages:h=stage(h)
  return h
 def forward(self,h):return self.sequence(h)[:,-1].tanh()


class ProgressiveSpatial(nn.Module):
 def __init__(self):
  super().__init__();self.pooled=progressive_mlp([512,256,128,64]);self.prototype=nn.Linear(6,64);self.output_norm=nn.LayerNorm(64)
 def forward(self,pooled,similarities):
  # Preserve the six existing prototype features without changing the requested 512 input.
  return self.output_norm(self.pooled(pooled)+self.prototype(similarities)).tanh()


class ProgressiveMemoryAttention(nn.Module):
 def __init__(self):
  super().__init__();self.value=progressive_mlp([1,16,32,64]);self.variable_id=nn.Parameter(torch.randn(8,64)*.02);self.norm=nn.LayerNorm(64);self.attn=nn.MultiheadAttention(64,2,dropout=0,batch_first=True);self.ffnorm=nn.LayerNorm(64);self.ffn=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64));self.readout=nn.Linear(64,1);self.raw_gate=nn.Parameter(torch.full((8,),-2.197224577))
 def forward(self,memory):
  tok=self.value(memory.unsqueeze(-1))+self.variable_id[None];z=self.norm(tok);a,w=self.attn(z,z,z,need_weights=True,average_attn_weights=False);z=tok+a;z=z+self.ffn(self.ffnorm(z));return memory+self.raw_gate.sigmoid()*self.readout(z).squeeze(-1).tanh(),w


class ProgressiveInvertedHistory(nn.Module):
 def __init__(self):
  super().__init__();self.embedding=progressive_mlp([24,16,32,64]);self.variable_id=nn.Parameter(torch.randn(8,64)*.02);self.attn=nn.MultiheadAttention(64,2,dropout=0,batch_first=True);self.norm1=nn.LayerNorm(64);self.ffn=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64));self.norm2=nn.LayerNorm(64);self.outnorm=nn.LayerNorm(64)
 def forward(self,h):
  assert h.shape[1:]==(24,8)
  z=self.embedding(h.transpose(1,2))+self.variable_id[None];a,_=self.attn(z,z,z,need_weights=False);z=self.norm1(z+a);z=self.norm2(z+self.ffn(z));return self.outnorm(z.mean(1)).tanh()


class ContextCrossFusion(nn.Module):
 def __init__(self):
  super().__init__();self.context_query=nn.Linear(64,3*64);self.query_id=nn.Parameter(torch.randn(3,64)*.02);self.branch_id=nn.Parameter(torch.randn(4,64)*.02);self.qnorm=nn.LayerNorm(64);self.kvnorm=nn.LayerNorm(64);self.attn=nn.MultiheadAttention(64,2,dropout=0,batch_first=True);self.ffnorm=nn.LayerNorm(64);self.ffn=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
 def forward(self,context,features,log_prior):
  q=self.context_query(context).reshape(len(context),3,64)+self.query_id[None];z=torch.stack(features,1)+self.branch_id[None];mask=log_prior[:,None,:].expand(-1,3,-1).repeat_interleave(2,0);a,w=self.attn(self.qnorm(q),self.kvnorm(z),self.kvnorm(z),attn_mask=mask,need_weights=True,average_attn_weights=False);u=q+a;return (u+self.ffn(self.ffnorm(u))).flatten(1),w


class AttentionWholeModel(WholeModel):
 def __init__(self,d,raw,seed,kind='full'):
  super().__init__(d,raw,seed,kind)
  # Remove replaced modules so unused parameters are not counted or optimized.
  for name in ['tcn','tcn_norm','mamba_input','mamba_norm','mamba','mamba_outnorm','spatial']:
   delattr(self.forecast,name)
  self.tcn=ProgressiveTCN();self.mamba=ProgressiveMamba();self.spatial_projection=ProgressiveSpatial();self.itransformer=ProgressiveInvertedHistory();self.memory_attention=ProgressiveMemoryAttention();self.context_encoder=progressive_mlp([18,16,32,64]);self.forecast.router=nn.Linear(64,4);self.cross_fusion=ContextCrossFusion();self.forecast.residual=nn.Sequential(nn.Linear(256,64),nn.Tanh(),nn.Linear(64,8))
  # Preserve the established zero-output-head initialization, not its incompatible widths.
  nn.init.zeros_(self.forecast.residual[-1].weight);nn.init.zeros_(self.forecast.residual[-1].bias)
 def encode(self,h,images):
  f=self.forecast;n=len(h);tokens=self.adapter(torch.cat([images,self.source_images],0));z=tokens[:n];src=F.normalize(tokens[n:],dim=-1);anchors=torch.stack([src[a,b,c,e] for a,b,c,e in self.source_positions]);zn=F.normalize(z,dim=-1);flat=zn.flatten(1,3);loc=f.prototype.location_logits.masked_fill(~f.prototype.valid_locations[None],-torch.inf).softmax(-1);sim=(torch.einsum('bld,kd->bkl',flat,anchors)*loc[None]).sum(-1);pooled=zn[:,f.prototype.valid_locations.reshape(f.prototype.geometry)].mean(1);pooled=((pooled-f.spatial_mean)/f.spatial_sd).clamp(-5,5);sim=((sim-f.sim_mean)/f.sim_sd).clamp(-5,5)
  return [self.tcn(h),self.mamba(h),self.spatial_projection(pooled,sim),self.itransformer(h)]
 def forward(self,h,month,images,diagnostics=False):
  f=self.forecast;features=self.encode(h,images);x=h[:,-1];a=torch.exp(-1/f.tau);memory=x.new_zeros(len(x),8)
  for past in h.unbind(1):memory=a*memory+(1-a)*past
  paths=[];deltas=[];weights=[];memory_weights=[];priors=[]
  for step in range(12):
   tendency,season=backbone_parts(f.base,x,month+step);enhanced,mw=self.memory_attention(memory);context=self.context_encoder(torch.cat([x,enhanced,season],-1));logits=f.router(context);latent,w=self.cross_fusion(context,features,F.log_softmax(logits,-1));delta=2*f.innovation*torch.tanh(f.residual(torch.cat([context,latent],-1)));x=x+tendency+delta;memory=a*memory+(1-a)*x;paths.append(x);deltas.append(delta);weights.append(w);memory_weights.append(mw);priors.append(logits.softmax(-1))
  path=torch.stack(paths,1)
  if diagnostics:return path,torch.stack(deltas,1),{'router_prior':torch.stack(priors,1),'fusion':torch.stack(weights,1),'memory':torch.stack(memory_weights,1),'branch_tokens':torch.stack(features,1),'final_raw_memory':memory}
  return path


def role(name):
 for prefix,label in [('tcn.','TCN'),('mamba.','Mamba3'),('spatial_projection.','spatial_projection'),('itransformer.','iTransformer_history'),('memory_attention.','EMA_variable_attention'),('context_encoder.','context_encoder'),('cross_fusion.','context_cross_attention')]:
  if name.startswith(prefix):return label
 return base_role(name)
