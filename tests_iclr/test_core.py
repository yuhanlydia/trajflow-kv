import copy
import math
import pytest
import torch

from tango_iclr.schema import normalize_records, validate_splits, public_record
from tango_iclr.objectives import action_objective, credit_loss, distillation_loss
from tango_iclr.stats import paired_summary, credit_summary
from tango_iclr.controller import CreditController
from tango_iclr.interventions import visual_spans, patch_span, LowRankBank, decoder_projection_names


def record(i='x', split='train'):
    return dict(prefix_id=i, task_id=i, family='gui', split=split,
                instruction='choose', image='now.png', history_images=['old.png','ref.png'],
                history_actions=['click','wait'], candidates=['A','B'], action_values=[1.,0.],
                target_index=0, credits=[.2,-.1], credit_kind='candidate_value_proxy')


def test_grouping_preserves_prefix():
    base=record()
    rows=[dict(base, action='A', Q=1.), dict(base, action='B', Q=0.)]
    for r in rows: r.pop('candidates'); r.pop('action_values')
    out=normalize_records(rows)
    assert len(out)==1 and out[0]['candidates']==['A','B']


def test_grouping_rejects_candidate_dependent_context():
    base=record(); rows=[dict(base,action='A',Q=1.),dict(base,action='B',Q=0.)]
    for r in rows: r.pop('candidates'); r.pop('action_values')
    rows[1]['instruction']='gold answer B'
    with pytest.raises(ValueError): normalize_records(rows)


def test_public_record_excludes_teacher_fields():
    p=public_record(record())
    assert not {'target_index','credits','action_values','credit_kind','return'} & p.keys()


def test_split_leakage():
    with pytest.raises(ValueError): validate_splits({'train':[record()], 'test':[record(split='test')]})


def test_action_advantage_has_reference_measure():
    s=torch.tensor([.2,-.1],requires_grad=True); p=torch.tensor([.8,.2]); q=torch.tensor([1.,0.])
    loss=action_objective(s,'action_advantage',values=q,reference=p)
    a=q-(p*q).sum()
    assert torch.allclose(loss,-(p*a*torch.log_softmax(s,0)).sum())


def test_distillation_rejects_negative_weights():
    with pytest.raises(ValueError): distillation_loss(torch.zeros(2),torch.zeros(2),weight=-1.)


def test_distillation_kl_zero():
    x=torch.tensor([1.,-1.]); assert abs(float(distillation_loss(x,x)))<1e-6


def test_credit_zero_stability():
    x=torch.zeros(2,requires_grad=True)
    loss=credit_loss(x,torch.zeros(2),scale=.01,deadzone=.005)
    loss.backward(); assert torch.isfinite(x.grad).all()


def test_controller_zero_history():
    m=CreditController(8,hidden=16,kind='gru')
    assert m(torch.empty(0,8),torch.ones(8)).numel()==0


@pytest.mark.parametrize('kind',['mlp','gru','transformer'])
def test_controller_shape_and_gradient(kind):
    m=CreditController(8,hidden=16,kind=kind)
    y=m(torch.randn(3,8),torch.randn(8))
    assert y.shape==(3,) and torch.isfinite(y).all()
    y.square().sum().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_visual_spans():
    ids=torch.tensor([[0,9,9,2,9,9,9,3]])
    assert visual_spans(ids,9)==[(1,3),(4,7)]


def test_visual_spans_batch_rejected():
    with pytest.raises(ValueError): visual_spans(torch.ones(2,3,dtype=torch.long),1)


def test_patch_only_target_and_identity():
    x=torch.arange(24.).reshape(1,6,4)
    y=patch_span(x,(1,3),x[:,4:6].clone())
    assert torch.equal(y[:,:1],x[:,:1]) and torch.equal(y[:,3:],x[:,3:])
    assert torch.equal(patch_span(x,(1,3),x[:,1:3]),x)


def test_patch_mismatch_is_error():
    with pytest.raises(ValueError): patch_span(torch.ones(1,6,4),(1,3),torch.ones(1,1,4))


def test_lowrank_identity_and_gradients():
    bank=LowRankBank({'layers.0.self_attn.k_proj':4},rank=2,alpha=2)
    x=torch.randn(1,6,4)
    y=bank.apply('layers.0.self_attn.k_proj',x,[(1,3)],torch.ones(1))
    assert torch.equal(x,y)
    (y.sum()).backward()
    assert bank.blocks['p0'].up.weight.grad.abs().sum()>0


def test_lowrank_only_history_changes():
    bank=LowRankBank({'layers.0.self_attn.k_proj':4},rank=2,alpha=2)
    with torch.no_grad(): bank.blocks['p0'].up.weight.fill_(.2)
    x=torch.randn(1,6,4)
    y=bank.apply('layers.0.self_attn.k_proj',x,[(1,3)],torch.ones(1))
    assert torch.equal(x[:,:1],y[:,:1]) and torch.equal(x[:,3:],y[:,3:])


def test_dynamic_gate_cannot_merge():
    bank=LowRankBank({'layers.0.self_attn.k_proj':4},rank=2,alpha=2)
    with pytest.raises(ValueError): bank.merge_static({},dynamic=True)


def test_decoder_only_selection():
    class Model(torch.nn.Module):
        def named_modules(self,*a,**kw):
            return iter([('visual.blocks.0.k_proj',torch.nn.Linear(4,4)),
                         ('model.language_model.layers.12.self_attn.k_proj',torch.nn.Linear(4,4))])
    names=decoder_projection_names(Model(),[12],'k')
    assert len(names)==1 and 'language_model' in names[0]


def test_paired_exact_stats():
    a=[dict(case_id=str(i),task_id=str(i),success=True) for i in range(4)]
    b=[dict(case_id=str(i),task_id=str(i),success=False) for i in range(4)]
    x=paired_summary(a,b,n_boot=200)
    assert x['wins']==4 and x['losses']==0 and x['mcnemar_p']==.125
    assert x['delta_pp']==100


def test_missing_pair_fails_closed():
    with pytest.raises(ValueError): paired_summary([dict(case_id='a',task_id='a',success=True)],[])


def test_credit_summary_units():
    out=credit_summary([dict(prefix_id='a',block=0,effect=.2),dict(prefix_id='a',block=1,effect=-.1)])
    assert out['independent_prefixes']==1 and out['block_measurements']==2


def test_legacy_family_seed_group_and_critical_union():
    rows=[dict(prefix_id='p1',task_family='hidden_memory',seed=12,
               instruction='Choose',image='x.png',action='A',Q=1.,is_critical_action=False),
          dict(prefix_id='p1',task_family='hidden_memory',seed=12,
               instruction='Choose',image='x.png',action='B',Q=1.,is_critical_action=True)]
    row=normalize_records(rows)[0]
    assert row['family']=='hidden_memory'
    assert row['task_id']=='hidden_memory:seed:12'
    assert row['critical_step'] is True
    assert row['optimal_indices']==[0,1]


def test_missing_task_group_fails_split_validation():
    row=normalize_records([dict(prefix_id='p',instruction='x',candidates=['A'])])[0]
    with pytest.raises(ValueError,match='task grouping'):
        validate_splits({'train':[row]})


def test_teacher_signature_changes_with_measurement_protocol():
    from types import SimpleNamespace
    from tango_iclr.teacher import generation_signature
    e=SimpleNamespace(model_path='qwen',revision='abc',target='k',layers=[12],max_pixels=100352,max_tokens=8192,four_bit=False)
    a=generation_signature(e,'candidate_value_proxy',3,'mean')
    assert a!=generation_signature(e,'candidate_value_proxy',1,'mean')
    assert a!=generation_signature(e,'local_action_logprob',3,'mean')
    e.revision='different'
    assert a!=generation_signature(e,'candidate_value_proxy',3,'mean')
