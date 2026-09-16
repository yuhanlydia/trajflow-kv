import pytest
from tango_iclr.suite import build_jobs

def config():
    return dict(model='model',train='train.jsonl',validation='val.jsonl',test='test.jsonl',
        output='runs',seeds=[0,1,2],base=dict(layers=list(range(12,24)),target='k',rank=8),
        online_manifests=['cases.jsonl'])

def test_suite_has_controls_and_three_training_seeds():
    jobs=build_jobs(config(),'core')
    train=[j for j in jobs if j['stage']=='train']
    assert len(train)==9*3
    assert {'ce','global_return','action_advantage','memory','shuffle_memory','flip_memory','unsigned','uniform','successful_ce'}=={j['config']['method'] for j in train}
    assert len({j['id'] for j in jobs})==len(jobs)

def test_suite_has_actual_cli_and_dependencies():
    jobs=build_jobs(config(),'core')
    for j in jobs:
        assert j.get('command') or j.get('config')
    for j in jobs:
        for dep in j.get('depends',[]):assert dep in {x['id'] for x in jobs}


def test_distilled_student_has_offline_and_online_evaluation():
    jobs=build_jobs(config(),'distill')
    for seed in [0,1,2]:
        for loss in ['kl','residual','ce']:
            label=f'distill_{loss}_s{seed}'
            for name in ['test_'+label,'online_0_'+label]:
                job=next(j for j in jobs if j['id']==name)
                assert '--adapter' in job['command']
                assert label in job['depends']
