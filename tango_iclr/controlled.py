"""SIGMA Qwen3 controlled run. Train/dev never loads sealed test observations."""
from __future__ import annotations
import argparse,copy,hashlib,json,subprocess,sys
from pathlib import Path
from .schema import digest,read_jsonl


def default_protocol():
    return dict(model='Qwen/Qwen3-VL-8B-Instruct',revision=None,expected_model_type='qwen3_vl',
        train_count=500,validation_count=50,test_count=50,train_template='A',validation_template='B',test_template='D',
        dataset_seed=20260917,data='data/sigma_qwen3_500',output='outputs/sigma_qwen3_500',
        layers=list(range(12,24)),target='k',rank=8,alpha=8.,credit_kind='local_action_logprob',
        donors=1,donor_policy='seeded',seeds=[11,23,37],four_bit=True,max_pixels=100352,max_tokens=8192,
        controller='cross_gru',hidden=256,feature_layers=[12,18,23],feature_chunks=8,
        credit_epochs=20,policy_epochs=5,credit_lr=.0003,policy_lr=.0001,
        credit_sign_weight=.1,credit_rank_weight=.1,credit_deadzone=.002,
        lambda_energy=.01,lambda_kl=0.,lambda_credit=1.,accumulation=8,
        freeze_controller_policy=True,gate_temperature=1.,score_normalization='mean',
        memory_policy_objective='action_sft',group_key='task_id')


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_selection(checkpoints,test_path,protocol,output):
    dest=Path(output)
    if dest.exists():raise FileExistsError('Selection already locked; do not overwrite after viewing test')
    if not checkpoints:raise ValueError('No checkpoints to lock')
    r=dict(schema='sigma.selection.v2',protocol_hash=digest(protocol),test_path=str(Path(test_path).resolve()),
        test_hash=sha256(test_path),checkpoints={k:dict(path=str(Path(v).resolve()),sha256=sha256(v)) for k,v in checkpoints.items()})
    images={}
    for row in read_jsonl(test_path):
        for ip in row.get('history_images',[])+([row['image']] if row.get('image') else []):images[ip]=sha256(ip)
    r['test_image_hashes']=images;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(r,indent=2)+'\n');return r


def verify_selection(record,test_path,protocol):
    if record['protocol_hash']!=digest(protocol):raise ValueError('Locked protocol changed')
    if record['test_hash']!=sha256(test_path):raise ValueError('Locked test changed')
    for item in record['checkpoints'].values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('Locked checkpoint changed')
    for path,h in record.get('test_image_hashes',{}).items():
        if sha256(path)!=h:raise ValueError('Locked test image changed')


def _budget(c):
    b=['--model',c['model'],'--max-pixels',str(c['max_pixels']),'--max-tokens',str(c['max_tokens'])]
    if c.get('four_bit'):b+=['--four-bit']
    if c.get('revision'):b+=['--revision',c['revision']]
    return b


def variants(stage):
    if stage in {'train','smoke'}:return [('action_sft',{}),('ce',{}),('memory',{})]
    return [('memory_pooled',{'controller':'gru'}),('memory_joint',{'freeze_controller_policy':False}),
        ('memory_no_sign',{'credit_sign_weight':0.}),('memory_no_energy',{'lambda_energy':0.}),
        ('memory_ce',{'memory_policy_objective':'ce'}),('memory_contrastive',{'credit_kind':'contrastive_action'}),
        ('memory_two_donors',{'donors':2}),('shuffle_memory',{}),('flip_memory',{}),('unsigned',{}),('uniform',{}),
        ('memory_rank4',{'rank':4}),('memory_rank16',{'rank':16}),('memory_v',{'target':'v'}),('memory_kv',{'target':'both'}),
        ('memory_early',{'layers':list(range(12)),'feature_layers':[0,6,11]}),
        ('memory_late',{'layers':list(range(24,36)),'feature_layers':[24,30,35]})]


def build_plan(cfg,stage):
    if stage not in {'train','smoke','ablations'}:raise ValueError(stage)
    root=Path(cfg['output'])/('smoke' if stage=='smoke' else 'formal');data=Path(cfg['data'])
    tr=data/'train.jsonl';va=data/'validation.jsonl'
    if stage=='smoke':tr=root/'smoke_train.jsonl';va=root/'smoke_validation.jsonl'
    jobs=[];teachers=set();python=sys.executable;base=str(root/'base_validation.json')
    jobs.append(dict(id='base_validation',requires=[str(va)],artifact=base,
        command=[python,'-m','tango_iclr.evaluate','--data',str(va),'--output',base,*_budget(cfg)]))
    for seed in cfg['seeds'][:1] if stage=='smoke' else cfg['seeds']:
        for label,overrides in variants(stage):
            c=copy.deepcopy(cfg);c.update(overrides)
            method=label if label in {'action_sft','ce','shuffle_memory','flip_memory','unsigned','uniform'} else 'memory'
            key=digest({k:c[k] for k in ['model','revision','layers','target','credit_kind','donors','donor_policy','four_bit','max_pixels','max_tokens','score_normalization']})[:14]
            files={};deps=[]
            if method in {'memory','shuffle_memory','flip_memory','unsigned'}:
                for split,path in [('train',tr),('validation',va)]:
                    tid=f'teacher_{key}_{split}';dest=str(root/'labels'/f'{key}_{split}.jsonl');files[split]=dest
                    if tid not in teachers:
                        teachers.add(tid)
                        jobs.append(dict(id=tid,requires=[str(path)],artifact=dest,command=[python,'-m','tango_iclr.teacher',
                            '--data',str(path),'--output',dest,'--target',c['target'],'--layers',*map(str,c['layers']),
                            '--kind',c['credit_kind'],'--donors',str(c['donors']),'--donor-policy',c['donor_policy'],
                            '--normalization',c['score_normalization'],*_budget(c)]))
                    deps.append(tid)
            name=f'{label}_s{seed}';dest=root/name
            c.update(method=method,seed=seed,train=files.get('train',str(tr)),validation=files.get('validation',str(va)),output=str(dest))
            c.pop('data',None)
            if stage=='smoke':c.update(credit_epochs=2,policy_epochs=1)
            jobs.append(dict(id='train_'+name,depends=deps,requires=[c['train'],c['validation']],config=c,artifact=str(dest/'best.pt')))
    return jobs


def _execute(jobs,root):
    import fcntl,yaml
    root.mkdir(parents=True,exist_ok=True);lock=open('/tmp/tango_iclr_gpu.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another SIGMA/TANGO GPU job is running')
    lp=root/'run_status.json';ledger=json.loads(lp.read_text()) if lp.exists() else {}
    code=digest({p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    try:
        for j in jobs:
            missing=[p for p in j.get('requires',[]) if not Path(p).is_file()]
            if missing:raise FileNotFoundError(f'{j["id"]}: {missing}')
            sig=digest([j,code,{p:sha256(p) for p in j.get('requires',[])}]);old=ledger.get(j['id'],{})
            if old.get('status')=='complete' and old.get('signature')==sig and Path(j['artifact']).is_file():
                print('RESUME',j['id'],flush=True);continue
            if old and old.get('signature')!=sig:raise ValueError(f'{j["id"]}: configuration/code changed; use a new output directory')
            cmd=j.get('command')
            if 'config' in j:
                p=root/'configs'/f'{j["id"]}.yaml';p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(yaml.safe_dump(j['config'],sort_keys=False));cmd=[sys.executable,'-m','tango_iclr.train','--config',str(p)]
            log=root/'logs'/f'{j["id"]}.log';log.parent.mkdir(parents=True,exist_ok=True);print('RUN',j['id'],flush=True)
            with log.open('w') as stream:rc=subprocess.call(cmd,stdout=stream,stderr=subprocess.STDOUT)
            ok=rc==0 and Path(j['artifact']).is_file()
            ledger[j['id']]=dict(status='complete' if ok else 'failed',signature=sig,returncode=rc,log=str(log),artifact=j['artifact'])
            lp.write_text(json.dumps(ledger,indent=2)+'\n')
            if not ok:raise RuntimeError(f'{j["id"]} failed: {log}')
    finally:lock.close()


def _model_lock(cfg,execute):
    """Pin the resolved model revision once; all methods share that model."""
    path=Path(cfg['output'])/'model_lock.json'
    if path.exists():
        r=json.loads(path.read_text())
        if r['model']!=cfg['model'] or (cfg.get('revision') and cfg['revision']!=r['revision']):raise ValueError('Model lock differs; use new output')
        cfg['revision']=r['revision'];return
    if not execute:return
    from transformers import AutoConfig
    c=AutoConfig.from_pretrained(cfg['model'],revision=cfg.get('revision'))
    if c.model_type!='qwen3_vl' or c.text_config.num_hidden_layers!=36:raise ValueError('Expected official Qwen3-VL-8B architecture with 36 layers')
    revision=cfg.get('revision') or getattr(c,'_commit_hash',None)
    if revision is None:raise ValueError('Use a revision-pinned Hub model ID; local snapshots need an explicit revision')
    cfg['revision']=revision;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(model=cfg['model'],revision=revision,layers=36),indent=2)+'\n')


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config')
    p.add_argument('--stage',required=True,choices=['prepare','smoke','train','ablations','lock','test','report'])
    p.add_argument('--execute',action='store_true');p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--font')
    p.add_argument('--include-ablations',action='store_true',help='Lock all ablation checkpoints before testing; all must exist')
    a=p.parse_args();cfg=default_protocol()
    if a.config:cfg.update(yaml.safe_load(Path(a.config).read_text()))
    if a.seeds:cfg['seeds']=a.seeds
    if 'Qwen3-VL-7B' in cfg['model']:raise ValueError('Use official Qwen3-VL-8B-Instruct, not a nonexistent 7B alias')
    if (cfg['train_count'],cfg['validation_count'],cfg['test_count'])!=(500,50,50):raise ValueError('Formal sizes are 500/50/50; smoke subsampling is separate')
    root=Path(cfg['output']);data=Path(cfg['data']);formal=root/'formal'
    if a.stage=='prepare':
        if not a.execute:print(json.dumps(cfg,indent=2));return
        from .controlled_data import generate_dataset
        generate_dataset(data,500,50,50,a.font,cfg['dataset_seed']);return
    _model_lock(cfg,a.execute and a.stage in {'smoke','train','ablations'})
    if a.stage in {'smoke','train','ablations'}:
        jobs=build_plan(cfg,a.stage);out=root/('smoke' if a.stage=='smoke' else 'formal');out.mkdir(parents=True,exist_ok=True)
        (out/f'plan_{a.stage}.json').write_text(json.dumps(jobs,indent=2)+'\n')
        print(f'{len(jobs)} jobs, no test execution in {a.stage}')
        if not a.execute:return
        if a.stage=='smoke':
            from .schema import write_jsonl
            write_jsonl(out/'smoke_train.jsonl',read_jsonl(data/'train.jsonl')[:10])
            write_jsonl(out/'smoke_validation.jsonl',read_jsonl(data/'validation.jsonl')[:8])
        _execute(jobs,out);return
    test=data/'test.jsonl';lock_path=formal/'selection.json'
    if a.stage=='lock':
        names=variants('train')+(variants('ablations') if a.include_ablations else [])
        checkpoints={f'{label}_s{s}':str(formal/f'{label}_s{s}'/'best.pt') for s in cfg['seeds'] for label,_ in names}
        if not a.execute:print(json.dumps(checkpoints,indent=2));return
        if not (root/'model_lock.json').exists():raise ValueError('Model revision must be locked before test')
        freeze_selection(checkpoints,test,cfg,lock_path);print('Locked checkpoints and test files');return
    if a.stage=='test':
        if not lock_path.exists():raise FileNotFoundError('Lock selected checkpoints before test')
        r=json.loads(lock_path.read_text());verify_selection(r,test,cfg);jobs=[]
        for label,ck in {'base':None,**r['checkpoints']}.items():
            budget=['--checkpoint',ck['path']] if ck else _budget(cfg)
            out=str(formal/label/'sealed_test.json')
            jobs.append(dict(id='test_'+label,requires=[str(test)]+([ck['path']] if ck else []),artifact=out,
                command=[sys.executable,'-m','tango_iclr.evaluate','--data',str(test),'--output',out,
                         '--normalization',cfg['score_normalization'],*budget]))
        print(f'{len(jobs)} locked test evaluations')
        if a.execute:_execute(jobs,formal/'sealed_evaluation')
        return
    from .controlled_report import summarize
    result=summarize(formal);print(json.dumps(result,indent=2))
    if a.execute:(formal/'controlled_summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
