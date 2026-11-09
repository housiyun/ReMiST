"""Download pinned official DPOT source and hash-verified Tiny checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import requests

ROOT=Path(__file__).resolve().parent
EXPECTED_SHA256='074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f'
REPO='thu-ml/DPOT'
WEIGHT_URL='https://huggingface.co/hzk17/DPOT/resolve/main/model_Ti.pth'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))

def download(session,url,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return
    part=path.with_suffix(path.suffix+'.part')
    for request in range(12):
        offset=part.stat().st_size if part.exists() else 0
        headers={'Range':f'bytes={offset}-'} if offset else {}
        with session.get(url,headers=headers,stream=True,timeout=(20,90)) as response:
            response.raise_for_status(); total=None
            if response.status_code==206:
                match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                if not match or int(match[1])!=offset: raise ValueError('Unexpected Range response')
                total=int(match[3])
            else:
                offset=0
                if response.headers.get('Content-Length'): total=int(response.headers['Content-Length'])
            with part.open('ab' if offset else 'wb') as out:
                for block in response.iter_content(1024*1024): out.write(block)
        print(json.dumps({'file':path.name,'bytes':part.stat().st_size,'expected_bytes':total}),flush=True)
        if total is None or part.stat().st_size==total:
            part.replace(path); return
    raise RuntimeError('Incomplete response after twelve byte-range requests')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--direct',action='store_true'); ap.add_argument('--source-only',action='store_true'); args=ap.parse_args()
    session=requests.Session(); session.trust_env=not args.direct
    provenance=ROOT/'provenance'; provenance.mkdir(parents=True,exist_ok=True)
    try:
        pin=provenance/'source_pin.json'
        if pin.exists(): commit=json.loads(pin.read_text())['commit']
        else:
            r=session.get(f'https://api.github.com/repos/{REPO}/commits/main',timeout=(20,40)); r.raise_for_status(); commit=r.json()['sha']
            write(pin,{'repository':REPO,'commit':commit})
        archive=provenance/'source.tar.gz'; download(session,f'https://codeload.github.com/{REPO}/tar.gz/{commit}',archive)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts=Path(member.name).parts[1:]
                if not member.isfile() or not parts or '..' in parts: continue
                rel=Path(*parts)
                if rel.suffix not in ('.py','.yaml','.yml','.txt') and rel.name not in ('README.md','LICENSE','AGENTS.md'): continue
                dest=provenance/'source'/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(tar.extractfile(member).read())
        result={'repository':REPO,'commit':commit,'source_archive_sha256':digest(archive),
                'source_files_sha256':{str(f.relative_to(provenance/'source')):digest(f) for f in (provenance/'source').rglob('*') if f.is_file()},
                'checkpoint_url':WEIGHT_URL,'published_checkpoint_sha256':EXPECTED_SHA256}
        write(provenance/'source_manifest.json',result)
        if args.source_only: print('SOURCE_READY',flush=True); return
        weight=ROOT/'weights/model_Ti.pth'; download(session,WEIGHT_URL,weight)
        if digest(weight)!=EXPECTED_SHA256: raise ValueError('Official published checkpoint SHA256 mismatch')
        result.update(status='DOWNLOADED_HASH_VERIFIED',checkpoint_sha256=digest(weight),checkpoint_bytes=weight.stat().st_size,
                      checkpoint_path=str(weight.relative_to(ROOT)),note='Actual torch state loading and forward are separate tests')
        write(provenance/'manifest.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}),flush=True)
    except Exception as exc:
        response=getattr(exc,'response',None)
        error={'status':'BLOCKED','error_type':type(exc).__name__,'http_status':response.status_code if response is not None else None,
               'direct':args.direct,'checkpoint_verified':False,'source_pin_present':(provenance/'source_pin.json').exists()}
        write(provenance/'last_failure.json',error)
        print(json.dumps(error),flush=True); raise SystemExit(1)

if __name__=='__main__': main()
