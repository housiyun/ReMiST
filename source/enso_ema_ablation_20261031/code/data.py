from pathlib import Path
import numpy as np, torch
import core
read,dump,sha=core.read,core.dump,core.sha
def backbone_parts(base,x,month):
 angle=2*np.pi*month/12;phase=torch.stack([torch.ones_like(angle),angle.sin(),angle.cos()],-1);f=.05*(phase@base.force.T)-torch.cat([.3*base.damp.sigmoid()*x[:,:7],torch.zeros_like(x[:,:1])],-1)
 b=.3*(phase@base.charge).sigmoid();c=.3*(phase@base.discharge).sigmoid();extra=torch.zeros_like(x);extra[:,3]=b*x[:,4:7].mean(-1);extra[:,4:7]=-c[:,None]*x[:,3:4];extra[:,:3]=.1*torch.tanh(x[:,4:7]@base.links.T);extra[:,7]=.5*base.relax.sigmoid()*(2*base.gain.sigmoid()*x[:,3]-x[:,7]);return f+extra,phase[:,1:]
