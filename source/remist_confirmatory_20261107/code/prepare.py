"""Server-side integrity checks; split training and observation packets."""
from common import *
guard()
import numpy as np
import torch

def main():
    torch.set_num_threads(1)
    p=read(ROOT/'protocol.json'); sources={name: {**item, 'path': str(ROOT/item['path'])} for name, item in p['inputs'].items()}
    hashes={key:sha(row['path']) for key,row in sources.items()}
    for key,row in sources.items():
        if row.get('sha256'):assert hashes[key]==row['sha256'], key
    d=torch.load(sources['regional']['path'],map_location='cpu',weights_only=False)
    raw=torch.load(sources['spatial']['path'],map_location='cpu',weights_only=False)
    b=torch.load(sources['extended']['path'],map_location='cpu',weights_only=False)
    assert len(d['tr']['history'])==3996 and len(d['va']['history'])==348
    train={'parent':{k:v for k,v in d.items() if k not in ('tests','testtokens')},'scale':b['regional_scale']}
    for src,dst in [('tr','train'),('va','validation')]:
        x=d[src]; maps=raw[src].float()
        h=torch.as_tensor(x['history']).float(); y=torch.as_tensor(x['target']).float()
        mo=torch.as_tensor(x['month']).float()
        assert h.shape[1:]==(24,8) and y.shape[1:]==(12,8) and maps.shape[1:]==(12,24,80)
        assert torch.isfinite(h).all() and torch.isfinite(maps).all()
        assert torch.isfinite(y[:,1:]).all(), 'Same complete state labels required for all methods'
        train[dst]={'h':h,'y':y,'month':mo,'maps':maps,'issue':np.asarray(x['issue']),
                    'product':np.asarray(x['product'])}
    # The old bundle used exactly the same3996 training rows. Its training-only PCA
    # is reused after row-by-row equality, not refitted on validation or observations.
    torch.testing.assert_close(train['train']['h'],b['train']['h'],rtol=0,atol=0)
    torch.testing.assert_close(train['train']['maps'],b['train']['maps'],rtol=0,atol=0)
    train['map_adapter']=b['map_adapter']
    for part in ('train','validation'):
        x=train[part]; a=train['map_adapter']
        x['pc']=((x['maps'].flatten(1)-a['mean'])@a['components'])/a['scale']
    increment=train['train']['y'][:,0]-train['train']['h'][:,-1]
    train['parent']['increment_scale']=increment.std(0).clamp(.03,1.)
    tests={}
    for name in ('SODA','GODAS'):
        old=b['tests'][name]; years=np.asarray(old['initialization'])//12
        mask=(years>=2000)&(years<=2022)
        assert int(mask.sum())==276
        tests[name]={k:(v[mask] if hasattr(v,'__len__') and len(v)==len(mask) else v)
                     for k,v in old.items() if k in ('h','month','maps','y','initialization','last_input','product')}
        tests[name]['y']=tests[name]['y'][:,:12]
        assert torch.isfinite(tests[name]['y'][:,1:]).all()
        dates=tests[name]['initialization']
        assert len(np.unique(dates))==276 and np.all(np.bincount(dates%12,minlength=12)==23)
    (ROOT/'data').mkdir(exist_ok=True)
    save(ROOT/'data/train.pt',train);save(ROOT/'data/tests.pt',tests)
    dump(ROOT/'data/receipt.json',{'status':'PASS','host':socket.gethostname(),'UTC':stamp(),
        'source_sha256':hashes,'source_paths':sources,'train_windows':3996,'validation_windows':348,
        'observation_origins_per_product':276,'observations_scored':False,
        'train_sha256':sha(ROOT/'data/train.pt'),'tests_sha256':sha(ROOT/'data/tests.pt'),
        'PCA':'Inherited train-only PCA after exact train-input equality check',
        'training_increment_scale':train['parent']['increment_scale'].tolist()})
    print('INPUT_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
