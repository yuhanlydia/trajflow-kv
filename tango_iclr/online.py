"""Real screenshot-only GUI execution. Native outcomes stay separate from diagnostics."""
from __future__ import annotations
import argparse,hashlib,json,time,traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from .schema import read_jsonl,write_jsonl,digest
from .qwen import QwenEngine
from .runtime import to_browser_action

BROWSER_PROMPT='''You control a browser using screenshots. Return one JSON action, no prose.
Valid actions: {"action_type":"click","x":100,"y":200},
{"action_type":"input_text","text":"text"}, {"action_type":"key","key":"ENTER"},
{"action_type":"scroll","delta_x":0,"delta_y":400}, {"action_type":"wait"},
{"action_type":"drag","x":100,"y":200,"to_x":300,"to_y":200},
{"action_type":"status","text":"Completed"}.
Coordinates are absolute pixels in the original screenshot. Observe after every action.
Do not report completion unless the task is complete.'''
ANDROID_PROMPT='''You control Android using screenshots. Return one JSON action, no prose.
Valid actions: {"action_type":"click","x":100,"y":200},
{"action_type":"input_text","text":"text"}, {"action_type":"scroll","direction":"down"},
{"action_type":"navigate_back"}, {"action_type":"open_app","app_name":"Contacts"},
{"action_type":"wait"}, {"action_type":"status","goal_status":"complete"}.
Coordinates are absolute pixels. Do not claim completion without checking the screen.'''

class BrowserEnvironment:
    def __init__(self,spec):
        import gymnasium as gym
        import browsergym.core
        module=spec.get('registration_module','browsergym.miniwob')
        import importlib;importlib.import_module(module)
        from browsergym.core.action.highlevel import HighLevelActionSet
        self.spec=spec;self.env=gym.make(spec['env_id'],headless=True,
            action_mapping=HighLevelActionSet(subsets=['coord','chat'],multiaction=False).to_python_code,
            **spec.get('env_kwargs',{}))
        self.reward=0.;self.done=False
    def reset(self):
        self.obs,_=self.env.reset(seed=int(self.spec['seed']));self.reward=0.;self.done=False
        return self.observe()
    def observe(self):return Image.fromarray(np.asarray(self.obs['screenshot'],dtype=np.uint8)).convert('RGB')
    def goal(self):return str(self.obs['goal'])
    def step(self,action):
        code=to_browser_action(action)
        self.obs,r,terminated,truncated,_=self.env.step(code);self.reward=float(r);self.done=bool(terminated or truncated)
        return self.observe()
    def score(self):return self.reward
    def close(self):self.env.close()

class AndroidEnvironment:
    def __init__(self,spec):
        # Reuse the actual, inspected upstream HTTP contract, not invented endpoints.
        from trajflow_kv.androidworld_http import AndroidWorldHTTPClient
        self.client=AndroidWorldHTTPClient(spec['base_url']);self.spec=spec;self.done=False
        if not self.client.health():raise RuntimeError('AndroidWorld server health check failed')
    def reset(self):
        self.client.initialize_task(self.spec['task_type'],int(self.spec['task_idx']));self.done=False
        return self.observe()
    def observe(self):return Image.fromarray(self.client.screenshot()).convert('RGB')
    def goal(self):return self.client.goal(self.spec['task_type'],int(self.spec['task_idx']))
    def step(self,action):
        a=json.loads(action)
        if not isinstance(a,dict) or 'action_type' not in a:raise ValueError('Invalid Android JSON')
        if a['action_type']=='status':self.done=True
        else:self.client.execute(a)
        return self.observe()
    def score(self):return self.client.score(self.spec['task_type'],int(self.spec['task_idx']))
    def close(self):self.client.tear_down_task(self.spec['task_type'],int(self.spec['task_idx']))

def make_environment(spec):
    if spec['backend']=='browsergym':return BrowserEnvironment(spec)
    if spec['backend']=='androidworld_http':return AndroidEnvironment(spec)
    raise ValueError('Use the official MemGUI CLI with tango_iclr.serve; no guessed MemGUI reset API')

def frame_hash(image):return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()

def observation_record(goal,current,past_images,past_actions,backend,history_limit):
    start=max(0,len(past_images)-history_limit)
    instruction=(BROWSER_PROMPT if backend=='browsergym' else ANDROID_PROMPT)+f'\nTask: {goal}\nScreen: {current.width}x{current.height}'
    return dict(instruction=instruction,image=current,history_images=past_images[start:],history_actions=past_actions[start:])

def run_episode(engine,spec,directory,controlled=False,history_limit=8,max_steps=30,temperature=0.,fixed_projectors=None):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    env=None;steps=[];images=[];actions=[];invalid=0;start=time.monotonic()
    result=dict(case_id=spec['case_id'],task_id=spec['task_id'],seed=spec['seed'],backend=spec['backend'],
                spec_hash=digest(spec),status='infra_error',success=None,history_limit=history_limit,
                max_steps=max_steps,temperature=temperature,controlled=controlled,action_interface='native_free_generation')
    try:
        env=make_environment(spec);screen=env.reset();goal=env.goal();torch.manual_seed(int(spec.get('decode_seed',spec['seed'])))
        for t in range(max_steps):
            name=root/f'{t:04d}.png';screen.save(name)
            record=observation_record(goal,screen,images,actions,spec['backend'],history_limit)
            action_start=time.monotonic()
            action=engine.generate(record,controlled,temperature,fixed_projectors=fixed_projectors)
            action_latency=time.monotonic()-action_start
            item=dict(step=t,image=str(name.resolve()),history_images=[x['image'] for x in steps][-history_limit:] if history_limit else [],
                      history_actions=actions[-history_limit:] if history_limit else [],instruction=record['instruction'],
                      action=action,state_hash=frame_hash(screen),action_latency_seconds=action_latency)
            steps.append(item)
            try:next_screen=env.step(action)
            except ValueError as exc:
                invalid+=1;item['invalid_action']=str(exc)
                # Invalid output is a MODEL failure, not an infrastructure exclusion.
                result.update(status='complete',success=False,failure_type='invalid_action');break
            images.append(screen.copy());actions.append(action);screen=next_screen
            if env.done:break
        if result['status']!='complete':
            score=env.score();result.update(status='complete',success=bool(score>=spec.get('success_threshold',1.)),native_score=score)
        result.update(goal=goal,steps=len(steps),invalid_actions=invalid,elapsed_seconds=time.monotonic()-start,
            action_latency_median=float(np.median([x['action_latency_seconds'] for x in steps])) if steps else None,
            action_latency_p90=float(np.quantile([x['action_latency_seconds'] for x in steps],.9)) if steps else None)
    except Exception as exc:result.update(status='infra_error',error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:result.update(status='infra_error',success=None,cleanup_error=str(exc))
    (root/'episode.json').write_text(json.dumps(dict(result,trajectory=steps),indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model');p.add_argument('--checkpoint');p.add_argument('--adapter');p.add_argument('--device',default='cuda')
    p.add_argument('--history-limit',type=int,default=8);p.add_argument('--max-steps',type=int,default=30)
    p.add_argument('--temperature',type=float,default=0.);p.add_argument('--fixed-projectors')
    p.add_argument('--max-pixels',type=int,default=100352);p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--revision');p.add_argument('--four-bit',action='store_true')
    a=p.parse_args();specs=read_jsonl(a.manifest)
    if not specs:raise ValueError('Empty native task manifest')
    ids=[s['case_id'] for s in specs]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate case IDs')
    e=QwenEngine.from_checkpoint(a.checkpoint,a.device,a.model) if a.checkpoint else QwenEngine(a.model,device=a.device,max_pixels=a.max_pixels,max_tokens=a.max_tokens,revision=a.revision,four_bit=a.four_bit)
    if a.adapter:
        if a.checkpoint:raise ValueError('Student evaluation must not also use teacher hooks')
        e.model.load_adapter(a.adapter)
    proj=torch.load(a.fixed_projectors,map_location='cpu',weights_only=True)['projectors'] if a.fixed_projectors else None
    policy_hash=digest(dict(model=e.model_path,revision=e.revision,checkpoint_hash=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,adapter=a.adapter,history_limit=a.history_limit,max_steps=a.max_steps,temperature=a.temperature,fixed_projectors_hash=hashlib.sha256(Path(a.fixed_projectors).read_bytes()).hexdigest() if a.fixed_projectors else None))
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);summaries=[]
    for spec in specs:
        safe=hashlib.sha256(spec['case_id'].encode()).hexdigest()[:20];folder=root/safe
        previous=folder/'episode.json'
        if previous.exists():
            old=json.loads(previous.read_text())
            if old['spec_hash']!=digest(spec) or old.get('policy_hash')!=policy_hash:raise ValueError('Resume manifest/policy mismatch: use a fresh output directory')
            if old['status']=='complete':summaries.append({k:v for k,v in old.items() if k!='trajectory'});continue
        result=run_episode(e,spec,folder,bool(a.checkpoint),a.history_limit,a.max_steps,a.temperature,proj)
        result['policy_hash']=policy_hash
        episode=json.loads((folder/'episode.json').read_text());episode['policy_hash']=policy_hash
        (folder/'episode.json').write_text(json.dumps(episode,indent=2))
        summaries.append(result)
        write_jsonl(root/'episodes.jsonl',summaries)
    write_jsonl(root/'episodes.jsonl',summaries)
    blocked=sum(x['status']!='complete' for x in summaries)
    print(json.dumps({'cases':len(summaries),'infra_errors':blocked,'successes':sum(x.get('success') is True for x in summaries)}))
    if blocked:raise SystemExit(2)
if __name__=='__main__':main()
