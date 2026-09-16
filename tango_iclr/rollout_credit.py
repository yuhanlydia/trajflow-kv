"""Actual paired one-decision KV intervention followed by base-policy continuation.

Requires reproducible reset/replay. Screenshots at the decision must match the
logged state exactly. Mismatches are BLOCKED, never turned into memory labels.
This operator changes ONE decision's history access; it is not persistent TTT.
"""
import argparse,json,copy,random
from pathlib import Path
import torch
from .online import make_environment,frame_hash,observation_record
from .qwen import QwenEngine
from .schema import read_jsonl,write_jsonl

def continuation(engine,record,patch,seed,horizon,history_limit):
    spec=record['environment'];env=make_environment(spec)
    try:
        screen=env.reset();images=[];actions=[]
        for a in record['replay_actions']:
            images.append(screen.copy());actions.append(a);screen=env.step(a)
        if frame_hash(screen)!=record['decision_state_hash']:
            raise RuntimeError('Replay state hash mismatch; paired state is not reproducible')
        torch.manual_seed(seed);goal=env.goal()
        obs=observation_record(goal,screen,images,actions,spec['backend'],history_limit)
        ctx=engine.context(obs,keep_donors=patch is not None)
        action=engine.generate_context(ctx,patch=patch,temperature=.7)
        for step in range(horizon):
            try:new=env.step(action)
            except ValueError:return 0.0
            images.append(screen.copy());actions.append(action);screen=new
            if env.done:break
            action=engine.generate(observation_record(goal,screen,images,actions,spec['backend'],history_limit),temperature=.7)
        return float(env.score())
    finally:env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--device',default='cuda');p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--horizon',type=int,default=20);p.add_argument('--history-limit',type=int,default=8)
    a=p.parse_args();e=QwenEngine(a.model,device=a.device);out=[];blocked=[]
    for r in read_jsonl(a.data):
        try:
            seeds=[10000+k for k in range(a.repeats)]
            base=[continuation(e,r,None,s,a.horizon,a.history_limit) for s in seeds]
            details=[];credits=[]
            for j,donors in enumerate(r['matched_donors']):
                effects=[]
                for d in donors:
                    if d==j:raise ValueError('Self donation forbidden')
                    changed=[continuation(e,r,(j,d),s,a.horizon,a.history_limit) for s in seeds]
                    effects.extend(x-y for x,y in zip(base,changed,strict=True))
                if not effects:raise ValueError('Missing matched donor')
                credits.append(sum(effects)/len(effects));details.append(effects)
            out.append(dict(r,credits=credits,credit_kind='paired_rollout_return',paired_effects=details,
                intervention_scope='one_decision_then_base_continuation',horizon=a.horizon,repeats=a.repeats))
            write_jsonl(a.output,out)
        except Exception as exc:blocked.append({'prefix_id':r['prefix_id'],'error':str(exc)})
    write_jsonl(str(a.output)+'.blocked.jsonl',blocked)
    if blocked:raise SystemExit(2)
if __name__=='__main__':main()
