"""One-prefix real-model forward/backward/cache contract test; NOT performance."""
import argparse,json
from pathlib import Path
import torch
from .schema import read_jsonl,normalize_records
from .qwen import QwenEngine


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--data',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(a.config).read_text());row=normalize_records(read_jsonl(a.data))[0]
    if row.get('split')=='test':raise ValueError('Smoke must not use held-out test')
    e=QwenEngine(cfg['model'],cfg['layers'],'k','cuda',cfg['max_pixels'],cfg['max_tokens'],cfg.get('revision'),cfg.get('four_bit',True))
    if e.model_type!='qwen3_vl' or e.num_hidden_layers!=36:raise ValueError('Unexpected model architecture')
    e.setup(cfg['controller'],cfg['hidden'],cfg['rank'],cfg['alpha'],'signed',
            feature_layers=cfg['feature_layers'],feature_chunks=cfg['feature_chunks'])
    e.controller.train();e.bank.train();c=e.context(row,keep_donors=True)
    action=row['candidates'][row['target_index']]
    with torch.no_grad():base=e.score(c,action);zero=e.score(c,action,controlled=True)
    if not torch.allclose(base,zero,atol=2e-4,rtol=1e-4):raise ValueError('Zero-residual bank is not equivalent to base')
    loss=-e.score(c,action,controlled=True);loss.backward()
    grads=[p.grad for p in e.bank.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(x).all() for x in grads):raise ValueError('Missing/nonfinite bank gradients')
    norm=float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
    if norm<=0:raise ValueError('Zero bank gradient; inspect gate and numeric path')
    torch.optim.SGD(e.bank.parameters(),lr=.001).step()
    with torch.no_grad():after=e.score(c,action,controlled=True)
    text=e.generate_context(c,controlled=True,temperature=0.,max_new_tokens=8)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    e.save(out.with_suffix('.pt'),dict(stage='model_contract_smoke',not_a_performance_result=True))
    result=dict(model=e.model_path,revision=e.revision,model_type=e.model_type,layers=e.num_hidden_layers,
        prefix_tokens=int(c.prompt_ids.shape[1]),history_blocks=len(c.spans),feature_shape=list(c.features.shape),
        query_shape=list(c.query.shape),bank_gradient_norm=norm,base_score=float(base),zero_score=float(zero),
        score_after_one_step=float(after),generated_text=text,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        controller_parameters=e.controller.parameter_count,not_a_performance_result=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
