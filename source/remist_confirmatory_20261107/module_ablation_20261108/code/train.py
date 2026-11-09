"""Resumable fitting; opens only train/validation packet, never observations."""
from common import *
guard()
import argparse, copy, math, random, signal, time
import numpy as np
import torch
from models import make

STOP=False
def stop(signum,frame):
    global STOP
    STOP=True

@torch.no_grad()
def predict(model,batch,device):
    model.eval();out=[]
    for i in range(0,len(batch['h']),32):
        out.append(model(*[batch[k][i:i+32].to(device) for k in ('h','month','maps')]).cpu())
    return torch.cat(out)

def mse(pred,truth):
    valid=torch.isfinite(truth[:,1:,7]); difference=pred[:,1:,7].double()-truth[:,1:,7].double()
    assert torch.isfinite(difference[valid]).all()
    return float(difference[valid].square().mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True);ap.add_argument('--seed',required=True,type=int)
    ap.add_argument('--budget',type=int,default=600);ap.add_argument('--multiplier',type=float,default=1.)
    ap.add_argument('--stage',choices=['smoke','tune','final'],required=True);a=ap.parse_args()
    p=read(ROOT/'protocol.json');assert a.kind in p['neural_kinds']
    if a.stage=='final':assert a.seed in p['seeds'] and a.budget in p['budgets']
    elif a.stage=='tune':assert a.seed==13 and a.budget==1800 and a.multiplier in p['lr_multipliers']
    else:assert a.budget==2
    job=f'{a.kind}_{a.seed}_{a.budget}_lr{a.multiplier:g}'
    out=ROOT/'jobs'/a.stage/job;out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);device='cuda:0'
    data=torch.load(ROOT/'data/train.pt',map_location='cpu',weights_only=False)
    model=make(a.kind,data,a.seed,device)
    bound=bindings();bound[str(ROOT/'data/train.pt')]=read(ROOT/'data/receipt.json')['train_sha256']
    if (out/'binding.json').exists():assert read(out/'binding.json')==bound
    else:dump(out/'binding.json',bound)
    if (out/'complete.json').exists():
        verify(bound); assert sha(out/'checkpoint.pt')==read(out/'complete.json')['checkpoint_sha256'];return
    groups={};active=[]
    for name,q in model.named_parameters():
        if not q.requires_grad:continue
        lr=1e-5 if name.startswith('adapter.') else 1e-4 if name.startswith(('forecast.base.','base.')) else .0015
        lr*=a.multiplier;wd=0. if q.ndim<2 or getattr(q,'_no_weight_decay',False) or name.split('.')[-1] in ('A_log','D','dt_bias') else .01
        groups.setdefault((lr,wd),[]).append(q);active.append(q)
    opt=torch.optim.AdamW([{'params':qs,'lr':lr,'initial_lr':lr,'weight_decay':wd} for (lr,wd),qs in groups.items()])
    tr={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in data['train'].items()}
    rng=np.random.default_rng(a.seed);best=float('inf');selected=0;logs=[];start_step=0;elapsed=0.;best_state=None
    seen=set();initial={k:q.detach().cpu().clone() for k,q in model.named_parameters()}
    initial_digest=__import__('hashlib').sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    if (out/'last.pt').exists():
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        assert last['binding_sha256']==sha(out/'binding.json') and last['initial_sha256']==initial_digest
        model.load_state_dict(last['state']);opt.load_state_dict(last['optimizer']);rng.bit_generator.state=last['numpy_rng']
        torch.set_rng_state(last['torch_rng']);torch.cuda.set_rng_state_all(last['cuda_rng']);random.setstate(last['python_rng'])
        best=last['best'];selected=last['selected'];logs=last['logs'];start_step=last['step'];elapsed=last['seconds'];seen=set(last['seen']);best_state=last['best_state']
        if best_state is not None:save(out/'checkpoint.pt',{'state':best_state,'kind':a.kind,'seed':a.seed,'selected_update':selected})
    start=time.monotonic();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def snapshot(step):
        save(out/'last.pt',{'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer':opt.state_dict(),'numpy_rng':copy.deepcopy(rng.bit_generator.state),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'best':best,'selected':selected,'best_state':best_state,
            'step':step,'logs':logs,'seconds':elapsed+time.monotonic()-start,'seen':sorted(seen),
            'binding_sha256':sha(out/'binding.json'),'initial_sha256':initial_digest})
    for step in range(start_step+1,a.budget+1):
        ids=rng.choice(len(tr['h']),4 if a.stage=='smoke' else 64,replace=False)
        model.train();opt.zero_grad(set_to_none=True)
        v=model(tr['h'][ids],tr['month'][ids],tr['maps'][ids]);target=tr['y'][ids]
        assert v.shape==target.shape and v.shape[1:]==(12,8)
        mask=torch.isfinite(target[:,1:]);im=mask[:,:,7]
        loss=(v[:,1:,7][im]-target[:,1:,7][im]).square().mean()+.1*(v[:,1:][mask]-target[:,1:][mask]).square().mean()
        assert torch.isfinite(loss),(a.kind,step);loss.backward()
        for name,q in model.named_parameters():
            if q.requires_grad and q.grad is not None:
                assert torch.isfinite(q.grad).all(),(name,step)
                if step%50==0 or step<=2:
                    if float(q.grad.norm())>0:seen.add(name)
        norm=torch.nn.utils.clip_grad_norm_(active,1,error_if_nonfinite=True)
        frac=max(0,(step-30)/max(a.budget-30,1));factor=step/30 if step<30 else .1+.45*(1+math.cos(math.pi*frac))
        for group in opt.param_groups:group['lr']=group['initial_lr']*factor
        opt.step()
        if step%50==0 or step==a.budget:
            pred=predict(model,data['validation'] if a.stage!='smoke' else {k:v[:4] for k,v in data['validation'].items()},device)
            truth=data['validation']['y'] if a.stage!='smoke' else data['validation']['y'][:4]
            score=mse(pred,truth);assert math.isfinite(score)
            entry={'step':step,'validation_MSE_normalized':score,'training_loss':float(loss.detach()),'gradient_norm':float(norm),
                'seconds':elapsed+time.monotonic()-start,'UTC':stamp()};logs.append(entry)
            if score<best:
                best=score;selected=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                save(out/'checkpoint.pt',{'state':best_state,'kind':a.kind,'seed':a.seed,'selected_update':selected})
            dump(out/'progress.json',entry);dump(out/'selection.json',logs);snapshot(step);print(entry,flush=True)
        if STOP:snapshot(step);raise SystemExit(130)
    if a.stage=='smoke':
        # Nonzero initial-output heads intentionally block upstream gradients on step1.
        # Two optimization steps must activate all registered trainable tensors.
        pass
    missing=[k for k,q in model.named_parameters() if q.requires_grad and k not in seen]
    assert not missing,('inactive trainable tensors',missing)
    if a.stage=='smoke' and a.kind!='bone':
        model.eval();batch={k:v[:2] for k,v in tr.items()}
        with torch.no_grad():
            original=model(batch['h'],batch['month'],batch['maps'])
            differences={
                'history':float((model(batch['h']+.1,batch['month'],batch['maps'])-original).abs().max()),
                'maps':float((model(batch['h'],batch['month'],batch['maps']+.1)-original).abs().max()),
                'month':float((model(batch['h'],batch['month']+1,batch['maps'])-original).abs().max())}
            assert all(value>0 and math.isfinite(value) for value in differences.values()),differences
        dump(out/'input_use.json',differences)
    verify(bound)
    dump(out/'complete.json',{'status':'SMOKE_PASS' if a.stage=='smoke' else 'FIT_COMPLETE_NOT_OBSERVATION_SCORED',
        'kind':a.kind,'stage':a.stage,'seed':a.seed,'budget':a.budget,'lr_multiplier':a.multiplier,
        'selected_update':selected,'validation_MSE_normalized':best,'parameters':sum(q.numel() for q in active),
        'missing_gradient_tensors':missing,'seconds':elapsed+time.monotonic()-start,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
        'checkpoint_sha256':sha(out/'checkpoint.pt'),'initial_sha256':initial_digest,'host':socket.gethostname(),
        'python':sys.executable,'UTC':stamp(),'observations_scored':False})
    print('FIT_COMPLETE',job,flush=True)

if __name__=='__main__':main()
