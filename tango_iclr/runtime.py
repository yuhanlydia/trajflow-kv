"""Shared, testable candidate scoring and public observation handling."""
from __future__ import annotations
import base64, io, json, math
from pathlib import Path
import torch
from PIL import Image

def score_tokens(logits, input_ids, prompt_len:int, end:int|None=None):
    end=input_ids.shape[1] if end is None else end
    if not 1<=prompt_len<end<=input_ids.shape[1]: raise ValueError('Empty or invalid action span')
    target=input_ids[:,prompt_len:end]
    selected=logits[:,prompt_len-1:end-1].float()
    logp=torch.log_softmax(selected,-1).gather(-1,target[...,None]).squeeze(-1)
    return logp.sum(), int(target.numel())

def candidate_vjp(scorer, count:int, loss_fn, divisor:float=1.,replay_regularizer=None):
    """Exact first-order gradient of a candidate loss, one VLM graph at a time.

    The scorer must be deterministic across the two passes (dropout disabled).
    No optimizer step may occur between scoring and replay.
    """
    with torch.no_grad(): values=torch.stack([scorer(i).detach() for i in range(count)])
    leaf=values.clone().requires_grad_(True)
    loss=loss_fn(leaf)/divisor
    weights,=torch.autograd.grad(loss,leaf)
    for i in range(count):
        if float(weights[i].abs())>0 or replay_regularizer is not None:
            objective=scorer(i)*weights[i]
            if replay_regularizer is not None:objective=objective+replay_regularizer()/(divisor*count)
            objective.backward()
    return float(loss.detach())*divisor, values

def gate_values(credit:torch.Tensor,mode='signed',temperature=1.):
    if temperature<=0: raise ValueError('Gate temperature must be positive')
    if mode=='uniform': return torch.ones_like(credit)
    if mode=='unsigned': return torch.sigmoid(credit/temperature)
    if mode=='signed': return torch.tanh(credit/temperature)
    raise ValueError(mode)

def load_image(value):
    if isinstance(value,Image.Image): return value.convert('RGB')
    if isinstance(value,dict): value=value.get('url',value.get('image'))
    if not isinstance(value,str): raise ValueError('Expected image path or data URI')
    if value.startswith('data:image/'):
        raw=base64.b64decode(value.split(',',1)[1],validate=True)
        if len(raw)>32*1024*1024: raise ValueError('Image exceeds 32 MiB')
        return Image.open(io.BytesIO(raw)).convert('RGB')
    if '://' in value: raise ValueError('Remote image URLs are not fetched; use local path or data URI')
    return Image.open(Path(value)).convert('RGB')

def prepare_messages(record):
    """Only explicitly allowlisted observations; never serialize the training row."""
    if 'messages' in record:
        messages=[]
        for m in record['messages']:
            if m['role'] not in {'system','user','assistant'}: raise ValueError('Unsupported message role')
            parts=m['content']; parts=[{'type':'text','text':parts}] if isinstance(parts,str) else parts
            converted=[]
            for p in parts:
                if p['type']=='text': converted.append({'type':'text','text':str(p['text'])})
                elif p['type'] in {'image','image_url'}:
                    converted.append({'type':'image','image':load_image(p.get('image',p.get('image_url')))})
                else: raise ValueError(f'Unsupported content type {p["type"]}')
            messages.append({'role':m['role'],'content':converted})
        return messages
    parts=[{'type':'text','text':str(record.get('prompt',record.get('instruction','')))}]
    previous=record.get('history_actions',[])
    for j,image in enumerate(record.get('history_images',[])):
        parts += [{'type':'text','text':f'Historical screen {j+1}:'}, {'type':'image','image':load_image(image)}]
        if j<len(previous): parts.append({'type':'text','text':f'Action taken: {previous[j]}'})
    if record.get('image') is not None:
        parts += [{'type':'text','text':'Current screen:'},{'type':'image','image':load_image(record['image'])}]
    parts.append({'type':'text','text':'Choose the next action using the current goal and available history.'})
    return [{'role':'user','content':parts}]

def to_browser_action(text:str) -> str:
    """Translate a strict JSON coordinate action to a BrowserGym primitive.

    Never execute arbitrary model-generated Python. The benchmark action set
    performs a second parse; all string values are escaped with repr.
    """
    try: a=json.loads(text)
    except (ValueError,TypeError) as e: raise ValueError('Action is not strict JSON') from e
    if not isinstance(a,dict): raise ValueError('Expected an action object')
    kind=a.get('action_type')
    def number(k,default=None):
        v=a.get(k,default)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v): raise ValueError(k)
        return float(v)
    if kind=='click': return f'mouse_click({number("x")}, {number("y")})'
    if kind=='scroll': return f'scroll({number("delta_x",0)}, {number("delta_y",0)})'
    if kind=='input_text': return f'keyboard_type({str(a["text"])!r})'
    if kind=='key': return f'keyboard_press({str(a["key"])!r})'
    if kind=='wait': return 'noop(500)'
    if kind=='status': return f'send_msg_to_user({str(a.get("text","Task complete"))!r})'
    if kind=='drag': return f'mouse_drag_and_drop({number("x")}, {number("y")}, {number("to_x")}, {number("to_y")})'
    raise ValueError(f'Unsupported browser action: {kind}')
