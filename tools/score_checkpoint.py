"""Score one checkpoint using the paper cohort and fixed-lead metrics."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

def metrics(pred,truth):
    pred=np.asarray(pred,dtype=np.float64);truth=np.asarray(truth,dtype=np.float64)
    acc=[];mse_leads=[];counts=[]
    for lead in range(11):
        ok=np.isfinite(truth[:,lead]);x=pred[ok,lead];y=truth[ok,lead];counts.append(int(ok.sum()))
        assert np.isfinite(x).all()
        mse_leads.append(float(np.mean((x-y)**2)))
        acc.append(float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else None)
    horizon=0
    for a in acc:
        if a is None or a<.5:break
        horizon+=1
    return {'MSE':float(np.mean((pred-truth)**2)),'mean_ACC':float(np.mean(acc)) if all(v is not None for v in acc) else None,
            'lead_MSE':mse_leads,'lead_ACC':acc,'lead_counts':counts,'H_0.5':horizon}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--predictions', type=Path, help='Optional output directory for forecast arrays')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'source/remist_confirmatory_20261107'
    study = root / 'module_ablation_20261108' if args.ablation else root
    checkpoint = args.checkpoint.resolve()
    output = args.predictions.resolve() if args.predictions else None
    sys.path.insert(0, str(study / 'code'))
    os.chdir(study)

    from common import read, sha
    from train import predict
    if args.ablation:
        from models import make
    else:
        from study_models import make

    receipt = read(root / 'data/receipt.json')
    for key in ('train', 'tests'):
        path = root / f'data/{key}.pt'
        if sha(path) != receipt[f'{key}_sha256']:
            raise ValueError(f'Packet checksum mismatch for {path}')

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'
    data = torch.load(root / 'data/train.pt', map_location='cpu', weights_only=False)
    tests = torch.load(root / 'data/tests.pt', map_location='cpu', weights_only=False)
    fit = torch.load(checkpoint, map_location='cpu', weights_only=True)
    torch.manual_seed(fit['seed'])
    if args.ablation:
        model = make(fit['kind'], data, fit['seed'], device)
    else:
        model = make(fit['kind'], data['parent'], data['train']['maps'], fit['seed'], device)
    model.load_state_dict(fit['state'], strict=True)
    scale = float(np.asarray(data['scale'])[7])
    scores = {}
    for product, batch in tests.items():
        forecast = predict(model, batch, device).numpy()
        truth = batch['y'].numpy()
        scores[product] = metrics(forecast[:, 1:, 7] * scale, truth[:, 1:, 7] * scale)
        if output:
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / f'{product}.npz', prediction=forecast,
                                truth=truth, initialization=batch['initialization'],
                                last_input=batch['last_input'], scale=data['scale'])
    print(json.dumps(scores, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
