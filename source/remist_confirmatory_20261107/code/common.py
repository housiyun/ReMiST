"""Metadata and server guard. No numerical work is allowed on the client."""
from pathlib import Path
import hashlib, json, os, socket, sys, datetime
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent

def guard():
    if Path.cwd() != ROOT:
        raise RuntimeError(f"Run this script from {ROOT}")

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())

def dump(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.part')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(q, p)

def save(p, value):
    import torch
    p = Path(p); q = p.with_name(p.name+'.part')
    torch.save(value, q); os.replace(q, p)

def bindings():
    paths = {ROOT/'protocol.json', *sorted((ROOT/'code').glob('*.py'))}
    for module in tuple(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if f and Path(f).is_relative_to(SOURCE) and str(f).endswith('.py'):
            paths.add(Path(f).resolve())
    return {str(p): sha(p) for p in sorted(paths)}

def verify(bound):
    for p, digest in bound.items(): assert sha(p) == digest, ('source/input drift', p)
