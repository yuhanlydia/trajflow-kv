"""Credit-fit diagnostics with correct raw/normalized scales, not GUI success."""
import torch


def _corr(a,b):
    a=a.double().flatten();b=b.double().flatten();a=a-a.mean();b=b-b.mean();den=a.norm()*b.norm()
    return float((a*b).sum()/den) if float(den)>1e-12 else None


def credit_metrics(prediction,target,*,scale,deadzone=.002):
    p=torch.as_tensor(prediction).float();y=torch.as_tensor(target).float()
    if p.shape!=y.shape or not p.numel() or scale<=0:raise ValueError('Aligned nonempty credits and positive scale required')
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite credit')
    raw=p*scale;active=y.abs()>deadzone;neutral=~active
    sign=lambda x:torch.where(x.abs()<=deadzone,torch.zeros_like(x),x.sign())
    truth=sign(y);guess=sign(raw);recalls=[]
    for c in [-1,0,1]:
        m=truth==c
        if m.any():recalls.append(float((guess[m]==c).float().mean()))
    pp=p.reshape(1,-1) if p.ndim==1 else p.reshape(p.shape[0],-1);yy=y.reshape_as(pp)
    cs=[_corr(a,b) for a,b in zip(pp,yy)];cs=[x for x in cs if x is not None]
    return dict(count=p.numel(),raw_rmse=float((raw-y).square().mean().sqrt()),
        normalized_rmse=float((p-y/scale).square().mean().sqrt()),correlation=_corr(raw,y),
        within_prefix_correlation=sum(cs)/len(cs) if cs else None,
        signed_accuracy=float((guess[active]==truth[active]).float().mean()) if active.any() else None,
        balanced_sign_accuracy=sum(recalls)/len(recalls),
        neutral_false_positive_rate=float((guess[neutral]!=0).float().mean()) if neutral.any() else None,
        predicted_positive_fraction=float((guess>0).float().mean()),target_positive_fraction=float((truth>0).float().mean()),
        argmax_block_accuracy=float((pp.argmax(-1)==yy.argmax(-1)).float().mean()))


@torch.no_grad()
def evaluate_credit(controller,compact,rows,device,scale,deadzone):
    prior=controller.training;controller.eval();pred=[];truth=[];records=[]
    try:
        for (f,q),r in zip(compact,rows):
            if 'credits' not in r:continue
            p=controller(f.to(device),q.to(device)).cpu();y=torch.tensor(r['credits'])
            if not len(p):continue
            pred.append(p);truth.append(y)
            records.append(dict(prefix_id=r['prefix_id'],predicted_normalized=p.tolist(),predicted_raw=(p*scale).tolist(),teacher_raw=y.tolist()))
    finally:controller.train(prior)
    if not pred:return dict(available=False)
    if len({len(x) for x in pred})==1:r=credit_metrics(torch.stack(pred),torch.stack(truth),scale=scale,deadzone=deadzone)
    else:
        r=credit_metrics(torch.cat(pred),torch.cat(truth),scale=scale,deadzone=deadzone)
        cs=[_corr(p,y) for p,y in zip(pred,truth)];cs=[v for v in cs if v is not None]
        r['within_prefix_correlation']=sum(cs)/len(cs) if cs else None
        r['argmax_block_accuracy']=sum(int(p.argmax()==y.argmax()) for p,y in zip(pred,truth))/len(pred)
    r.update(available=True,records=records);return r
