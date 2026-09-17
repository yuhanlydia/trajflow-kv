"""Finite-candidate objectives; these are NOT mislabeled as on-policy RL."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def action_objective(scores:torch.Tensor,objective:str,*,target_index:int|None=None,
                     values:torch.Tensor|None=None,reference:torch.Tensor|None=None,
                     trajectory_return:float|None=None,baseline:float=0.) -> torch.Tensor:
    if scores.ndim!=1 or not torch.isfinite(scores).all(): raise ValueError('finite 1-D scores required')
    if objective=='action_sft':
        if target_index is None or not 0<=target_index<len(scores):
            raise ValueError('Action SFT requires a demonstrated action index')
        # scores are mean action-token log-probabilities, NOT candidate probabilities.
        return -scores[target_index]
    logp=F.log_softmax(scores.float(),dim=0)
    if objective in {'ce','successful_ce','memory','memory_shuffled','memory_signflip','unsigned','uniform'}:
        if target_index is None: raise ValueError('CE requires a training target')
        if objective=='successful_ce' and trajectory_return is not None and trajectory_return<=0:
            return scores.sum()*0.
        return -logp[target_index]
    if objective in {'action_advantage','global_return'}:
        if reference is None: raise ValueError('Reference sampling measure must be explicit')
        p=reference.detach().to(scores).float()
        if p.shape!=scores.shape or torch.any(p<0) or not torch.isclose(p.sum(),p.new_tensor(1.),atol=1e-5):
            raise ValueError('Reference must be a normalized candidate distribution')
        if objective=='action_advantage':
            if values is None: raise ValueError('Action advantage needs continuation values')
            q=values.detach().to(scores).float()
            if q.shape!=scores.shape: raise ValueError('Q shape mismatch')
            advantage=q-(p*q).sum()
            return -(p*advantage*logp).sum()
        if target_index is None or trajectory_return is None:
            raise ValueError('Global-return baseline requires logged action and trajectory return')
        # Explicitly an offline weighted-likelihood surrogate, not an unbiased policy gradient.
        return -float(trajectory_return-baseline)*logp[target_index]
    raise ValueError(f'Unknown objective: {objective}')


def credit_loss(prediction:torch.Tensor,target:torch.Tensor,*,scale:float,deadzone:float=.005,
                rank_weight:float=.1,sign_weight:float=0.) -> torch.Tensor:
    if scale<=0 or deadzone<0: raise ValueError('invalid credit calibration')
    if prediction.shape!=target.shape: raise ValueError('credit shape mismatch')
    if prediction.numel()==0: return prediction.sum()*0.
    y=(target.detach().float()/scale).clamp(-5,5)
    loss=F.smooth_l1_loss(prediction.float(),y)
    # Only compare pairs whose measured difference exceeds the neutral threshold.
    delta=target[:,None]-target[None,:]
    valid=torch.triu(torch.abs(delta)>deadzone,diagonal=1)
    if valid.any():
        p=prediction[:,None]-prediction[None,:]
        loss=loss+rank_weight*F.softplus(-delta.sign()[valid]*p[valid]).mean()
    if sign_weight<0:raise ValueError('Sign weight must be nonnegative')
    if sign_weight:
        terms=[]
        for sign in [-1,1]:
            mask=target*sign>deadzone
            if mask.any():terms.append(F.softplus(-sign*prediction[mask].float()).mean())
        if terms:loss=loss+sign_weight*torch.stack(terms).mean()
    return loss


def distillation_loss(student_scores:torch.Tensor,teacher_scores:torch.Tensor,*,weight:float=1.,
                      temperature:float=1.) -> torch.Tensor:
    if weight<0 or temperature<=0: raise ValueError('KL weight must be nonnegative; temperature positive')
    if student_scores.shape!=teacher_scores.shape: raise ValueError('distillation support mismatch')
    q=F.softmax(teacher_scores.detach().float()/temperature,dim=-1)
    logp=F.log_softmax(student_scores.float()/temperature,dim=-1)
    return float(weight)*temperature**2*F.kl_div(logp,q,reduction='sum')


def centered_residual_loss(student:torch.Tensor,base:torch.Tensor,teacher:torch.Tensor) -> torch.Tensor:
    ds=student.float()-base.detach().float(); dt=teacher.detach().float()-base.detach().float()
    return F.mse_loss(ds-ds.mean(-1,keepdim=True),dt-dt.mean(-1,keepdim=True))
