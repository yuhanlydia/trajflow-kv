"""Target-free ordered token bins; these are not exact 2-D object crops."""
import math
import torch
import torch.nn.functional as F


def pool_chunks(tokens, chunks):
    if tokens.ndim!=2 or tokens.shape[0]==0 or chunks<=0:
        raise ValueError('Nonempty [tokens,width] and positive chunks required')
    return F.adaptive_avg_pool1d(tokens.float().T.unsqueeze(0),chunks)[0].T


def position_encoding(length,width,device):
    pos=torch.arange(length,device=device)[:,None]
    freq=torch.exp(torch.arange(0,width,2,device=device)*(-math.log(10000.)/width))
    out=torch.zeros(length,width,device=device)
    out[:,0::2]=torch.sin(pos*freq)
    out[:,1::2]=torch.cos(pos*freq[:out[:,1::2].shape[1]])
    return out
