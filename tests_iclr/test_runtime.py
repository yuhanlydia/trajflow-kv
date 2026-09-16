import base64, io, copy
import pytest, torch
from PIL import Image
from tango_iclr.runtime import score_tokens, candidate_vjp, prepare_messages, gate_values, to_browser_action

def test_score_is_sum_not_mean():
    z=torch.zeros(1,5,3); ids=torch.tensor([[0,1,2,1,0]])
    total,n=score_tokens(z,ids,3,5)
    assert n==2 and torch.allclose(total,torch.tensor(-2*1.0986123))

def test_vjp_matches_full_graph():
    w=torch.nn.Parameter(torch.tensor([.2,-.5]))
    def scorer(i): return w[i]*2
    def loss(x): return -torch.log_softmax(x,0)[0]
    candidate_vjp(scorer,2,loss)
    got=w.grad.clone(); w.grad=None
    loss(torch.stack([scorer(0),scorer(1)])).backward()
    assert torch.allclose(got,w.grad)

def test_signed_and_unsigned_bounds():
    x=torch.tensor([-100.,0.,100.])
    assert torch.allclose(gate_values(x,'signed'),torch.tensor([-1.,0.,1.]))
    assert (gate_values(x,'unsigned')>=0).all()

def test_json_browser_action_does_not_execute_code():
    with pytest.raises(ValueError): to_browser_action("__import__('os').system('bad')")
    with pytest.raises(ValueError): to_browser_action('{"action_type":"exec","code":"bad"}')
    assert to_browser_action('{"action_type":"click","x":4,"y":7}')=='mouse_click(4.0, 7.0)'

def test_messages_public_only(tmp_path):
    p=tmp_path/'x.png'; Image.new('RGB',(32,32)).save(p)
    row=dict(instruction='work',image=str(p),history_images=[str(p)],history_actions=['wait'],credits=[999],target_index=3)
    msg=prepare_messages(row)
    assert len([c for m in msg for c in m.get('content',[]) if c.get('type')=='image'])==2
    assert '999' not in str(msg) and 'target_index' not in str(msg)

def test_vjp_with_regularizer_matches_joint_graph():
    w=torch.nn.Parameter(torch.tensor([.2,-.5]))
    current={}
    def score(i):
        current['reg']=(w[i]*3).square()
        return w[i]*2
    def loss(x):return -torch.log_softmax(x,0)[0]
    candidate_vjp(score,2,loss,replay_regularizer=lambda:current['reg'])
    actual=w.grad.clone();w.grad=None
    (loss(w*2)+(w*3).square().mean()).backward()
    assert torch.allclose(actual,w.grad)
