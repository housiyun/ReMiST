"""Fresh CMIP-only fits; model-held-out validation; frozen observation tests."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(v):
 if isinstance(v,dict):return {k:serial(a) for k,a in v.items()}
 if isinstance(v,(tuple,list)):return [serial(a) for a in v]
 if isinstance(v,np.ndarray):return serial(v.tolist())
 if isinstance(v,np.generic):return serial(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def dump(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(serial(v),ensure_ascii=False,indent=2,allow_nan=False))
def features(row,sst_observed=None):
 dates=np.array([int(s[:4])*12+int(s[5:])-1 for s in row['dates']]); x=np.asarray(row['values'],float)
 if sst_observed is None:sst=np.asarray(row['sst_n34'],float)
 else:
  dd,v=sst_observed;ix=np.searchsorted(dd,dates);np.testing.assert_array_equal(dd[ix],dates);sst=v[ix]
 raw=np.c_[x,sst]; anomaly=np.full_like(raw,np.nan)
 for i in range(180,len(dates)):
  past=np.arange(i-180,i,12);assert dates[past].max()<dates[i]
  anomaly[i]=raw[i]-raw[past].mean(0)
 index=np.full(len(dates),np.nan)
 for i in range(2,len(dates)):index[i]=anomaly[i-2:i+1,-1].mean()
 state=np.c_[anomaly[:,:7],index];return dates,state

def windows(dates,state,years,april=False):
 hs=[];ys=[];ms=[];issues=[];centers=[]
 for i in range(23,len(dates)):
  issue=int(dates[i])+2;year=issue//12
  if not years[0]<=year<=years[1] or (april and issue%12!=3):continue
  h=state[i-23:i+1]
  if not np.isfinite(h).all():continue
  target=np.full((12,8),np.nan);n=min(12,len(state)-i-1)
  if n:target[:n]=state[i+1:i+1+n]
  hs.append(h);ys.append(target);ms.append(dates[i]%12);issues.append(issue);centers.append(np.arange(dates[i],dates[i]+12))
 if not hs:raise ValueError('No complete input histories')
 return {'history':np.array(hs,np.float32),'target':np.array(ys,np.float32),'month':np.array(ms),'issue':np.array(issues),'target_centers':np.array(centers)}
def scaled(b,scale):
 return {**b,'history':b['history']/scale,'target':b['target']/scale}
class Model(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  self.damp=nn.Parameter(torch.full((7,),-2.));self.charge=nn.Parameter(torch.tensor([-1.5,0.,0.]));self.discharge=nn.Parameter(torch.tensor([-2.,0.,0.]));self.force=nn.Parameter(torch.zeros(8,3));self.relax=nn.Parameter(torch.tensor(-1.5));self.gain=nn.Parameter(torch.tensor(0.));self.links=nn.Parameter(torch.zeros(3,3))
  self.tau_raw=nn.Parameter(torch.logit((torch.tensor([6.,12.])-.5)/35.5),requires_grad=kind=='memory')
  if kind=='gru':self.gru=nn.GRUCell(8,8)
  if kind!='ro':self.net=nn.Sequential(nn.Linear(8+(8 if kind=='gru' else 2)+2,16),nn.Tanh(),nn.Linear(16,8))
 @property
 def tau(self):return .5+35.5*self.tau_raw.sigmoid()
 def forward(self,h,month):
  x=h[:,-1];a=torch.exp(-1/self.tau);mem=torch.zeros_like(x[:,:2]);context=torch.zeros(len(x),8)
  if self.kind=='memory':
   for v in h.unbind(1):mem=a*mem+(1-a)*v[:,[4,5]]
  if self.kind=='gru':
   for v in h.unbind(1):context=self.gru(v,context)
  results=[]
  for k in range(12):
   angle=2*np.pi*(month+k)/12;phase=torch.stack([torch.ones_like(angle),angle.sin(),angle.cos()],-1);f=.05*(phase@self.force.T);f=f-torch.cat([.3*self.damp.sigmoid()*x[:,:7],torch.zeros_like(x[:,:1])],-1)
   b=.3*(phase@self.charge).sigmoid();c=.3*(phase@self.discharge).sigmoid();extra=torch.zeros_like(x);extra[:,3]=b*x[:,4:7].mean(-1);extra[:,4:7]=-c[:,None]*x[:,3:4];extra[:,:3]=.1*torch.tanh(x[:,4:7]@self.links.T);extra[:,7]=.5*self.relax.sigmoid()*(2*self.gain.sigmoid()*x[:,3]-x[:,7]);f=f+extra
   if self.kind!='ro':f=f+.1*torch.tanh(self.net(torch.cat([x,context if self.kind=='gru' else mem,phase[:,1:]],-1)))
   x=x+f;results.append(x)
   if self.kind=='memory':mem=a*mem+(1-a)*x[:,[4,5]]
   if self.kind=='gru':context=self.gru(x,context)
  return torch.stack(results,1)
@torch.no_grad()
def predict(m,b):return m(torch.tensor(b['history']),torch.tensor(b['month'],dtype=torch.float32)).numpy()
def score(pred,y):
 out=[]
 for h in range(12):
  a=pred[:,h];t=y[:,h];mask=np.isfinite(t)&np.isfinite(a);a=a[mask];t=t[mask];mse=float(np.mean((a-t)**2)) if len(t) else None
  out.append({'model_step':h+1,'n':len(t),'MSE_C2':mse,'RMSE_C':None if mse is None else float(np.sqrt(mse)),'ACC':float(np.corrcoef(a,t)[0,1]) if len(t)>2 and a.std()>1e-12 and t.std()>1e-12 else None})
 valid=np.isfinite(y);mse=float(np.mean((pred[valid]-y[valid])**2));return {'MSE_C2':mse,'RMSE_C':float(np.sqrt(mse)),'by_lead':out}
