"""Export observed successful GUI trajectories to action-SFT prefixes.

Accepts our native episode.json or legacy trajectory JSONL, not arbitrary vendor
schemas. Preserves observed actions and screenshots. No synthetic Q, optimal
continuation, stale/useful role, or future observation is invented.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .schema import read_jsonl, write_jsonl, digest, resolve_images


def trajectory_prefixes(record: dict, *, split: str, min_history: int = 2,
                        history_limit: int = 8) -> list[dict]:
    if split not in {'train','validation','test'}:
        raise ValueError('Unknown split')
    if min_history<0 or history_limit<min_history:
        raise ValueError('Require 0 <= min_history <= history_limit')
    if record.get('status','complete')!='complete':return []
    success=record.get('success')
    if success is None:
        score=record.get('return',record.get('native_score'))
        if score is None:raise ValueError('Missing recorded task success/return')
        success=float(score)>=1.
    if success is not True:return []
    task_id=str(record.get('task_id',''))
    if not task_id:raise ValueError('A real task_id is required for grouped splits')
    steps=record.get('trajectory',record.get('steps'))
    if not isinstance(steps,list) or not steps:raise ValueError('No observed trajectory steps')
    episode_id=str(record.get('case_id',record.get('trajectory_id',digest([task_id,steps])[:20])))
    rows=[]
    for i,step in enumerate(steps):
        if i<min_history:continue
        if step.get('invalid_action') or any(steps[j].get('invalid_action') for j in range(i)):
            continue
        if not step.get('image') or 'action' not in step:raise ValueError('Each step needs image and executed action')
        action=step['action']
        if isinstance(action,dict):action=json.dumps(action,sort_keys=True,separators=(',',':'))
        if not isinstance(action,str) or not action.strip():raise ValueError('Empty action')
        start=max(0,i-history_limit)
        past=steps[start:i]
        rows.append(dict(prefix_id=f'{episode_id}:step:{i}',task_id=task_id,trajectory_id=episode_id,
            split=split,family=str(record.get('family',record.get('task_family',record.get('backend','unspecified')))),
            instruction=step.get('instruction',record.get('instruction',record.get('goal',''))),
            image=step['image'],history_images=[x['image'] for x in past],
            history_actions=[x['action'] if isinstance(x['action'],str) else json.dumps(x['action'],sort_keys=True) for x in past],
            candidates=[action],target_index=0,logged_action=action,
            trajectory_return=float(record.get('return',record.get('native_score',1.))),
            target_source='observed_action_in_successful_trajectory',
            data_origin='native_gui_demonstration',candidate_support='single_demonstration',
            critical_step=False))
    return rows


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='JSONL trajectories, episode.json, or directory containing episode.json')
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','validation','test'],required=True)
    p.add_argument('--image-root',default='.');p.add_argument('--min-history',type=int,default=2)
    p.add_argument('--history-limit',type=int,default=8)
    a=p.parse_args();path=Path(a.input)
    if path.is_dir():
        records=[json.loads(f.read_text()) for f in sorted(path.rglob('episode.json'))]
    elif path.suffix=='.json':records=[json.loads(path.read_text())]
    else:records=read_jsonl(path)
    rows=[];used=0
    for record in records:
        batch=trajectory_prefixes(record,split=a.split,min_history=a.min_history,history_limit=a.history_limit)
        used+=bool(batch);rows.extend(batch)
    if not rows:raise ValueError('No usable successful prefixes; provide actual demonstrations, not toy replacements')
    if len({r['prefix_id'] for r in rows})!=len(rows):raise ValueError('Repeated episode/prefix identifiers')
    rows=resolve_images(rows,a.image_root)
    for r in rows:
        for image in [r['image']]+r['history_images']:
            if not Path(image).is_file():raise FileNotFoundError(image)
    write_jsonl(a.output,rows)
    report=dict(input_episodes=len(records),episodes_with_prefixes=used,prefixes=len(rows),
                excluded_episodes=len(records)-used,min_history=a.min_history,history_limit=a.history_limit,
                data_origin='native_gui_demonstration',contains_enumerated_q=False,
                primary_endpoint='native_online_task_success_not_single_candidate_accuracy')
    Path(str(a.output)+'.summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
