from pathlib import Path
import hashlib,json,os,socket,sys,datetime
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
SOURCE=PARENT.parent

def guard():
 if Path.cwd()!=ROOT:
  raise RuntimeError(f"Run this script from {ROOT}")

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.part');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def save(p,v):
 import torch
 p=Path(p);q=p.with_name(p.name+'.part');torch.save(v,q);os.replace(q,p)
def bindings():
 paths={ROOT/'protocol.json',PARENT/'protocol.json',*sorted((ROOT/'code').glob('*.py')),*sorted((PARENT/'code').glob('*.py'))}
 for module in tuple(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and Path(f).is_relative_to(SOURCE) and str(f).endswith('.py'):paths.add(Path(f).resolve())
 if (ROOT/'construction_checks.json').exists():paths.add(ROOT/'construction_checks.json')
 return {str(p):sha(p) for p in sorted(paths)}
def verify(bound):
 for p,digest in bound.items():assert sha(p)==digest,('source/input drift',p)
