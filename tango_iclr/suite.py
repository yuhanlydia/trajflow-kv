"""Materialize and execute a sequential, resumable experiment DAG.

Unconfigured data or environments produce BLOCKED, never fake success. GPU jobs
run serially under one process lock. A successful command requires its artifact.
"""
from __future__ import annotations
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
from .schema import digest

def build_jobs(cfg,profile):
    root=Path(cfg['output']); jobs=[];python=sys.executable
    configs=[]
    base=cfg.get('base',{})
    budget=['--max-pixels',str(base.get('max_pixels',100352)),'--max-tokens',str(base.get('max_tokens',8192))]
    if cfg.get('revision'):budget+=['--revision',cfg['revision']]
    if base.get('four_bit',False):budget+=['--four-bit']
    # Base is evaluated with the identical observation/action budget, but no controller prefill.
    base_out=str(root/'base'/'test_candidates.json')
    jobs.append(dict(id='base_test',stage='offline',requires=[cfg['test']],artifact=base_out,
        command=[python,'-m','tango_iclr.evaluate','--model',cfg['model'],'--data',cfg['test'],'--output',base_out,*budget]))
    if profile!='smoke':
        for i,manifest in enumerate(cfg.get('online_manifests',[])):
            base_online=str(root/'base'/f'online_{i}')
            jobs.append(dict(id=f'base_online_{i}',stage='online',requires=[manifest],artifact=str(Path(base_online)/'episodes.jsonl'),
                command=[python,'-m','tango_iclr.online','--model',cfg['model'],'--manifest',manifest,'--output',base_online,
                    '--history-limit',str(cfg.get('history_limit',8)),'--max-steps',str(cfg.get('max_steps',30)),*budget]))
    methods=['ce','successful_ce','global_return','action_advantage','shuffle_memory','flip_memory','unsigned','uniform','memory']
    for seed in cfg.get('seeds',[0,1,2]):
        if profile in {'smoke'}:active=['memory']
        elif profile=='primary':active=['action_sft','memory']
        else:active=cfg.get('methods',methods)
        for method in active:
            conf=dict(cfg.get('base',{}),model=cfg['model'],validation=cfg['validation'],method=method,seed=seed)
            if profile=='smoke':conf.update(credit_epochs=1,policy_epochs=1)
            configs.append((f'{method}_s{seed}',conf))
        if profile in {'ablations','all'}:
            variants=[('mlp',{'controller':'mlp'}),('transformer',{'controller':'transformer'}),
                      ('no_energy',{'lambda_energy':0.}),('no_credit_aux',{'lambda_credit':0.}),
                      ('rank4',{'rank':4}),('rank16',{'rank':16}),
                      ('v_only',{'target':'v'}),('kv',{'target':'both'}),
                      ('early',{'layers':list(range(0,12))}),('late',{'layers':list(range(24,36))}),
                      ('one_donor',{'donors':1}),('local_credit',{'credit_kind':'local_action_logprob'})]
            for label,override in variants:
                conf=dict(cfg.get('base',{}),model=cfg['model'],validation=cfg['validation'],method='memory',seed=seed,**override)
                configs.append((f'abl_{label}_s{seed}',conf))
        if profile=='smoke':break
    teacher_ids={}
    for label,conf in configs:
        target=conf.get('target','k');layers=conf.get('layers',list(range(12,24)))
        kind=conf.get('credit_kind',cfg.get('credit_kind','candidate_value_proxy'));donors=conf.get('donors',3)
        key=digest([cfg['model'],cfg.get('revision'),target,layers,kind,donors,conf.get('max_pixels',100352),conf.get('max_tokens',8192),conf.get('four_bit',False)])[:12];teacher_id='teacher_'+key
        annotated=str(root/'labels'/f'{key}.jsonl')
        needs_credit=conf['method'] in {'memory','shuffle_memory','flip_memory','unsigned'}
        if needs_credit and key not in teacher_ids:
            command=[python,'-m','tango_iclr.teacher','--data',cfg['train'],'--output',annotated,
                '--model',cfg['model'],'--target',target,'--layers',*map(str,layers),'--kind',kind,'--donors',str(donors),
                '--max-pixels',str(conf.get('max_pixels',100352)),'--max-tokens',str(conf.get('max_tokens',8192))]
            if cfg.get('revision'):command+=['--revision',cfg['revision']]
            if conf.get('four_bit',False):command+=['--four-bit']
            if profile=='smoke':command+=['--max-prefixes','10']
            jobs.append(dict(id=teacher_id,stage='teacher',command=command,requires=[cfg['train']],artifact=annotated))
            teacher_ids[key]=teacher_id
        conf.update(train=annotated if needs_credit else cfg['train'],output=str(root/label),revision=cfg.get('revision'))
        job_id='train_'+label;checkpoint=str(root/label/'best.pt')
        jobs.append(dict(id=job_id,stage='train',config=conf,requires=[conf['train'],cfg['validation']],
                         depends=[teacher_id] if needs_credit else [],artifact=checkpoint))
        outfile=str(root/label/'test_candidates.json')
        jobs.append(dict(id='test_'+label,stage='offline',depends=[job_id],requires=[cfg['test'],checkpoint],artifact=outfile,
            command=[python,'-m','tango_iclr.evaluate','--data',cfg['test'],'--checkpoint',checkpoint,'--output',outfile]))
        if profile!='smoke':
            for i,manifest in enumerate(cfg.get('online_manifests',[])):
                online_out=str(root/label/f'online_{i}')
                jobs.append(dict(id=f'online_{i}_{label}',stage='online',depends=[job_id],requires=[manifest,checkpoint],
                    artifact=str(Path(online_out)/'episodes.jsonl'),command=[python,'-m','tango_iclr.online',
                    '--manifest',manifest,'--checkpoint',checkpoint,'--output',online_out,
                    '--history-limit',str(cfg.get('history_limit',8)),'--max-steps',str(cfg.get('max_steps',30)),*budget]))
    if profile=='distill' or (profile=='all' and cfg.get('enable_distillation',False)):
        for seed in cfg.get('seeds',[0,1,2]):
            teacher=str(root/f'memory_s{seed}'/'best.pt')
            for loss in ['kl','residual','ce']:
                dest=str(root/f'distill_{loss}_s{seed}')
                jobs.append(dict(id=f'distill_{loss}_s{seed}',stage='distill',depends=[f'train_memory_s{seed}'],
                    requires=[teacher,cfg['train'],cfg['validation']],artifact=str(Path(dest)/'student.json'),command=[python,'-m','tango_iclr.distill',
                    '--teacher',teacher,'--train',cfg['train'],'--validation',cfg['validation'],'--output',dest,'--seed',str(seed),'--loss',loss]))
                student_id=f'distill_{loss}_s{seed}'
                adapter=str(Path(dest)/'best_adapter')
                offline=str(Path(dest)/'test_candidates.json')
                jobs.append(dict(id='test_'+student_id,stage='offline',depends=[student_id],
                    requires=[adapter,cfg['test']],artifact=offline,command=[python,'-m','tango_iclr.evaluate',
                    '--model',cfg['model'],'--adapter',adapter,'--data',cfg['test'],'--output',offline,*budget]))
                for i,manifest in enumerate(cfg.get('online_manifests',[])):
                    online=str(Path(dest)/f'online_{i}')
                    jobs.append(dict(id=f'online_{i}_{student_id}',stage='online',depends=[student_id],
                        requires=[adapter,manifest],artifact=str(Path(online)/'episodes.jsonl'),
                        command=[python,'-m','tango_iclr.online','--model',cfg['model'],'--adapter',adapter,
                            '--manifest',manifest,'--output',online,'--history-limit',str(cfg.get('history_limit',8)),
                            '--max-steps',str(cfg.get('max_steps',30)),*budget]))
    if profile in {'ablations','all'}:
        spd=str(root/'spd_style'/'projectors.pt')
        jobs.append(dict(id='spd_calibration',stage='calibration',requires=[cfg['train']],artifact=spd,
            command=[python,'-m','tango_iclr.spd_control','--model',cfg['model'],'--data',cfg['train'],'--output',spd]))
        jobs.append(dict(id='spd_test',stage='offline',depends=['spd_calibration'],requires=[spd,cfg['test']],
            artifact=str(root/'spd_style'/'test.json'),command=[python,'-m','tango_iclr.evaluate','--model',cfg['model'],
                '--data',cfg['test'],'--fixed-projectors',spd,'--output',str(root/'spd_style'/'test.json'),*budget]))
        for i,manifest in enumerate(cfg.get('online_manifests',[])):
            dest=str(root/'spd_style'/f'online_{i}')
            jobs.append(dict(id=f'spd_online_{i}',stage='online',depends=['spd_calibration'],requires=[spd,manifest],
                artifact=str(Path(dest)/'episodes.jsonl'),command=[python,'-m','tango_iclr.online','--model',cfg['model'],
                    '--manifest',manifest,'--fixed-projectors',spd,'--output',dest,'--history-limit',str(cfg.get('history_limit',8)),'--max-steps',str(cfg.get('max_steps',30)),*budget]))
    return jobs

def main():
    import yaml,fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--profile',choices=['smoke','primary','core','ablations','distill','all'],default='core')
    p.add_argument('--execute',action='store_true');p.add_argument('--continue-on-error',action='store_true')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    jobs=build_jobs(cfg,a.profile);(root/'suite_plan.json').write_text(json.dumps(jobs,indent=2))
    print(f'{len(jobs)} jobs planned; plan: {root/"suite_plan.json"}')
    if not a.execute:return
    lock=open('/tmp/tango_iclr_gpu.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another TANGO suite is running; serial GPU execution is enforced')
    ledger_path=root/'suite_status.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    blocked=False
    for j in jobs:
        import hashlib
        inputs={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in j.get('requires',[]) if Path(path).is_file()}
        code_signature=digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
        fingerprint=digest([j,inputs,code_signature]);previous=ledger.get(j['id'],{})
        if previous.get('status')=='complete' and previous.get('fingerprint')==fingerprint and Path(j['artifact']).exists():continue
        missing=[x for x in j.get('requires',[]) if not Path(x).exists()]
        unmet=[x for x in j.get('depends',[]) if ledger.get(x,{}).get('status')!='complete']
        if missing or unmet:
            ledger[j['id']]=dict(status='blocked',missing=missing,dependencies=unmet,fingerprint=fingerprint);blocked=True
        else:
            log=root/'logs'/f'{j["id"]}.log';log.parent.mkdir(parents=True,exist_ok=True)
            if 'config' in j:
                config_path=root/'generated_configs'/f'{j["id"]}.yaml';config_path.parent.mkdir(parents=True,exist_ok=True)
                config_path.write_text(yaml.safe_dump(j['config'],sort_keys=False));command=[sys.executable,'-m','tango_iclr.train','--config',str(config_path)]
            else:command=j['command']
            with log.open('w') as stream:code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT)
            status='complete' if code==0 and Path(j['artifact']).exists() else 'failed'
            ledger[j['id']]=dict(status=status,returncode=code,artifact=j['artifact'],log=str(log),fingerprint=fingerprint)
            blocked|=status!='complete'
        ledger_path.write_text(json.dumps(ledger,indent=2));print(j['id'],ledger[j['id']]['status'],flush=True)
        if ledger[j['id']]['status']!='complete' and not a.continue_on_error:break
    if blocked:raise SystemExit(2)
if __name__=='__main__':main()
