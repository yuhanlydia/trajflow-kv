"""True same-model LoRA policy distillation; no algebraic merge.

Uses one backbone: adapters disabled for the frozen TANGO teacher and enabled for
student updates. Only nonnegative sample weights are allowed. No gold or future
information is added to the teacher's observations.
"""
import argparse,json,random
from pathlib import Path
from contextlib import contextmanager
import torch
from .qwen import QwenEngine
from .schema import read_jsonl,normalize_records,validate_splits
from .runtime import candidate_vjp
from .objectives import distillation_loss,centered_residual_loss,action_objective
from .evaluate import evaluate_records
from .train import selection_score

@contextmanager
def adapters_disabled(model):
    model.disable_adapters()
    try:yield
    finally:model.enable_adapters()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',required=True);p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--rank',type=int,default=16)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--loss',choices=['kl','residual','ce'],default='kl');p.add_argument('--teacher-mode',choices=['tango','base'],default='tango')
    a=p.parse_args();random.seed(a.seed);torch.manual_seed(a.seed)
    rows=normalize_records(read_jsonl(a.train));val=normalize_records(read_jsonl(a.validation))
    validate_splits({'train':rows,'validation':val})
    if any(len(r['candidates'])<2 for r in rows):
        raise ValueError('Finite-candidate distillation needs multiple independent proposals; single demonstration rows are SFT data')
    if any(r.get('split')=='test' for r in rows+val):raise ValueError('Test data cannot be used for distillation')
    from peft import LoraConfig
    e=QwenEngine.from_checkpoint(a.teacher,a.device)
    e.controller.requires_grad_(False);e.bank.requires_grad_(False)
    lora=LoraConfig(r=a.rank,lora_alpha=2*a.rank,target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_dropout=0.,bias='none',task_type='CAUSAL_LM')
    # Transformers' native adapter API keeps decoder names unchanged.
    e.model.add_adapter(lora,adapter_name='tango_student')
    params=[p for p in e.model.parameters() if p.requires_grad]
    if not params:raise RuntimeError('No LoRA parameters became trainable')
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    best=-1.;log=[]
    for epoch in range(a.epochs):
        random.shuffle(rows);losses=[]
        for r in rows:
            weight=float(r.get('distill_weight',1.))
            if weight<0:raise ValueError('Distillation requires nonnegative weights')
            with adapters_disabled(e.model),torch.no_grad():
                c=e.context(r)
                base=torch.stack([e.score(c,x,False) for x in r['candidates']])
                teacher=torch.stack([e.score(c,x,a.teacher_mode=='tango') for x in r['candidates']])
            # Same inputs; student uses no memory hook, and receives neither credit nor role labels.
            opt.zero_grad(set_to_none=True)
            def lf(s):
                if a.loss=='ce':return weight*action_objective(s,'ce',target_index=r['target_index'])
                if a.loss=='kl':return distillation_loss(s,teacher,weight=weight)
                return weight*centered_residual_loss(s,base,teacher)+.05*distillation_loss(s,base)
            loss,_=candidate_vjp(lambda j:e.score(c,r['candidates'][j],False),len(r['candidates']),lf)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(loss)
        result=evaluate_records(e,val,controlled=False)
        score=selection_score(result)
        if score is None:raise ValueError('Validation lacks labels')
        if score>best:
            best=score;e.model.save_pretrained(out/'best_adapter');e.processor.save_pretrained(out/'best_adapter')
        log.append(dict(epoch=epoch,loss=sum(losses)/max(len(losses),1),validation=score))
        (out/'distillation_log.json').write_text(json.dumps(log,indent=2))
    (out/'student.json').write_text(json.dumps(dict(base_model=e.model_path,revision=e.revision,adapter=str(out/'best_adapter'),
        teacher=a.teacher,loss=a.loss,teacher_mode=a.teacher_mode,seed=a.seed,history='same public observations',
        no_hook=True,not_algebraic_merge=True),indent=2))
if __name__=='__main__':main()
