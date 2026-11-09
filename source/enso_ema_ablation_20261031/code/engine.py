"""Independent jobs, sealed selection before tests; no observation-guided choice."""
from pathlib import Path
import sys,os,time,math,json,hashlib,resource
import numpy as np,torch
import importlib.util
import model as fm
Integrated=fm.Integrated
from data import core,read,dump,sha
RUN=Path(__file__).resolve().parents[1]
def fit_base(seed,tr,va):
 torch.manual_seed(seed);m=core.Model('ro');opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=.003,weight_decay=.01);rng=np.random.default_rng(seed);best=float('inf');beststate=None
 for step in range(1,161):
  ids=rng.choice(len(tr['history']),min(96,len(tr['history'])),replace=False);y=torch.tensor(tr['target'][ids]);mask=torch.isfinite(y);opt.zero_grad(set_to_none=True);v=m(torch.tensor(tr['history'][ids]),torch.tensor(tr['month'][ids],dtype=torch.float32));loss=(v[mask]-y[mask]).square().mean()+2*(v[:,:,7][mask[:,:,7]]-y[:,:,7][mask[:,:,7]]).square().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step%20==0:
   v=core.predict(m,va)[:,:,7];target=va['target'][:,:,7];mask=np.isfinite(target);value=float(np.mean((v[mask]-target[mask])**2))
   if value<best:best=value;beststate={k:v.detach().clone() for k,v in m.state_dict().items()};chosen=step
 m.load_state_dict(beststate);m.eval()
 for q in m.parameters():q.requires_grad_(False)
 return m,{'state':beststate,'selected':chosen,'validation_MSE':best,'seed':seed,'updates':160,'selection':'CMIP inner validation only; original RO objective'}
def base_object(s):
 m=core.Model('ro');m.load_state_dict(s);m.eval()
 for q in m.parameters():q.requires_grad_(False)
 return m

def run_job(task):
 torch.set_num_threads(1);os.environ['OMP_NUM_THREADS']='1';ds,kind,seed,budget=task;out=RUN/'jobs'/f'{ds}__{kind}__{seed}__{budget}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():return read(out/'complete.json')
 start=time.monotonic();d=torch.load(RUN/'data'/f'{ds}.pt',map_location='cpu',weights_only=False);tr,va=d['tr'],d['va'];base=base_object(d['bases'][seed]);torch.manual_seed(seed);m=Integrated(base,d['tt'],tr['issue'].tolist(),d['support'],d['innovation'][seed],kind)
 initial={k:v.detach().clone() for k,v in m.state_dict().items()};params=[q for q in m.parameters() if q.requires_grad];decay=[];nd=[]
 for name,q in m.named_parameters():
  if q.requires_grad:(nd if q.ndim<2 or getattr(q,'_no_weight_decay',False) or name.split('.')[-1] in ['A_log','D','dt_bias'] else decay).append(q)
 opt=torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':nd,'weight_decay':0}],lr=.0015);rng=np.random.default_rng(seed);best=float('inf');logs=[]
 # Frozen tensor inputs are prepared once rather than copied on every update.
 h=torch.tensor(tr['history']);y=torch.tensor(tr['target']);mo=torch.tensor(tr['month'],dtype=torch.float32)
 for step in range(1,budget+1):
  ids=rng.choice(len(h),64,replace=False);target=y[ids];mask=torch.isfinite(target[:,3:]);imask=mask[:,:,7];m.train();opt.zero_grad(set_to_none=True);path,delta,_=m(h[ids],mo[ids],d['tt'][ids],diagnostics=True);loss=(path[:,3:,7][imask]-target[:,3:,7][imask]).square().mean()+.1*(path[:,3:][mask]-target[:,3:][mask]).square().mean()+.002*(delta/m.innovation).square().mean()
  if not torch.isfinite(loss):raise FloatingPointError((task,step))
  loss.backward();torch.nn.utils.clip_grad_norm_(params,1);fraction=max(0,(step-30)/(budget-30));lr=.0015*(step/30 if step<30 else .1+.45*(1+math.cos(math.pi*fraction)))
  for g in opt.param_groups:g['lr']=lr
  opt.step()
  if step%50==0:
   pred=fm.predict(m,va,d['vt'])[:,:,7];target=va['target'][:,:,7];mask=np.isfinite(target[:,3:]);value=float(np.mean((pred[:,3:][mask]-target[:,3:][mask])**2));logs.append({'step':step,'validation_MSE':value,'last_training_loss':float(loss.detach()),'seconds':time.monotonic()-start})
   if value<best:best=value;chosen=step;state={k:v.detach().clone() for k,v in m.state_dict().items()}
   dump(out/'progress.json',{'step':step,'budget':budget,'best_validation_MSE':best,'seconds':time.monotonic()-start})
 assert all(torch.equal(v,initial['base.'+k]) for k,v in m.base.state_dict().items());m.load_state_dict(state);m.eval();torch.save({'state':state,'seed':seed,'kind':kind,'budget':budget,'selected':chosen,'innovation':d['innovation'][seed]},out/'checkpoint.pt');dump(out/'selection.json',logs)
 # This per-job seal precedes any test prediction/scoring; previous jobs cannot alter its recipe.
 seal={'checkpoint_sha256':sha(out/'checkpoint.pt'),'protocol_sha256':sha(RUN/'protocol.json'),'data_sha256':read(RUN/'data'/f'{ds}_receipt.json')['sha256'],'source_sha256':{str(f):sha(f) for f in (RUN/'code').glob('*.py')},'test_fitted_updates':0};dump(out/'seal.json',seal)
 tests={}
 for name,b in d['tests'].items():
  v,delta,w=fm.predict(m,b,d['testtokens'][name],True);np.savez_compressed(out/f'{name}.npz',prediction=v,truth=b['target'],issue=b['issue'],weights=w,residual=delta);assert np.array_equal(fm.predict(m,b,d['testtokens'][name]),v);tests[name]={'sha256':sha(out/f'{name}.npz')}
 result={'dataset':ds,'kind':kind,'seed':seed,'budget':budget,'selected_update':chosen,'validation_MSE':best,'tau_months':m.tau.detach().tolist(),'parameter_components':{name:sum(q.numel() for q in module.parameters() if q.requires_grad) for name,module in [('router',m.router),('residual',m.residual)]},'trainable_parameters':sum(q.numel() for q in params),'total_registered_parameters':sum(q.numel() for q in m.parameters()),'seconds':time.monotonic()-start,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'tests':tests,'status':'COMPLETE'};dump(out/'complete.json',result);return result
