"""Explicit-history Qwen engine. Native GPU smoke is required before large runs.

The clean prefill precedes action scoring. Its features do not depend on target
labels or candidate suffixes. Patched donors are captured from the clean prefill,
not from an already-intervened layer. Two prefills are charged to inference cost.
"""
from __future__ import annotations
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
import torch
from .interventions import decoder_projection_names,visual_spans,hooks,patch_span,LowRankBank
from .controller import CreditController
from .runtime import prepare_messages,score_tokens,gate_values

@dataclass
class Context:
    messages:list
    batch:dict
    spans:list
    features:torch.Tensor
    query:torch.Tensor
    donors:dict
    prompt_ids:torch.Tensor

class QwenEngine:
    def __init__(self,model_path,layers=None,target='k',device='cuda',max_pixels=100352,
                 max_tokens=8192,revision=None,four_bit=False):
        from transformers import AutoConfig,AutoProcessor,Qwen2_5_VLForConditionalGeneration
        cfg=AutoConfig.from_pretrained(model_path,revision=revision)
        revision=revision or getattr(cfg,'_commit_hash',None)
        cls=Qwen2_5_VLForConditionalGeneration
        if cfg.model_type=='qwen3_vl':
            from transformers import Qwen3VLForConditionalGeneration
            cls=Qwen3VLForConditionalGeneration
        elif cfg.model_type!='qwen2_5_vl': raise ValueError(f'Unsupported model_type: {cfg.model_type}')
        options=dict(revision=revision,torch_dtype=torch.bfloat16 if device!='cpu' else torch.float32,
                     device_map=device,attn_implementation='sdpa')
        if four_bit:
            from transformers import BitsAndBytesConfig
            options['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
        self.model=cls.from_pretrained(model_path,**options).eval()
        self.model.requires_grad_(False)
        self.processor=AutoProcessor.from_pretrained(model_path,revision=revision,max_pixels=max_pixels)
        self.device=torch.device(device); self.model_path=model_path; self.revision=revision
        self.max_pixels=max_pixels; self.max_tokens=max_tokens; self.target=target; self.four_bit=four_bit
        textcfg=getattr(cfg,'text_config',cfg); n=int(textcfg.num_hidden_layers)
        self.layers=layers or list(range(n//3,2*n//3))
        self.names=decoder_projection_names(self.model,self.layers,target)
        mods=dict(self.model.named_modules()); self.widths={n:mods[n].out_features for n in self.names}
        if len(set(self.widths.values()))!=1: raise ValueError('Controller requires equal K/V projection widths')
        self.feature_dim=next(iter(self.widths.values()))
        self.image_token_id=int(getattr(cfg,'image_token_id',self.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')))
        self.controller=None; self.bank=None; self.gate_mode='signed'; self.temperature=1.; self.metadata={}

    def _batch(self,messages,generation_prompt=False):
        b=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation_prompt,
                return_dict=True,return_tensors='pt')
        if b['input_ids'].shape[1]>self.max_tokens:
            raise ValueError(f'Context exceeds {self.max_tokens} tokens; reduce common history/pixel budget, never silently truncate')
        return b.to(self.device)

    @torch.no_grad()
    def context(self,record,keep_donors=False):
        messages=prepare_messages(record); batch=self._batch(messages,True)
        spans=visual_spans(batch['input_ids'],self.image_token_id)
        if not spans: raise ValueError('No image tokens: this path must remain a visual GUI evaluation')
        # Contract: the last image is the current observation, all preceding images are history.
        history=spans[:-1]; pooled=[]; queries=[]; donors={}
        def callback(name):
            def capture(_m,_i,x):
                pooled.append(torch.stack([x[0,s:e].float().mean(0) for s,e in history]) if history else x.new_empty((0,x.shape[-1]),dtype=torch.float32))
                queries.append(x[0,-1].float())
                if keep_donors: donors[name]=[x[:,s:e].detach().clone() for s,e in history]
                return x
            return capture
        with hooks(self.model,{n:callback(n) for n in self.names}):
            self.model(**batch,use_cache=False,logits_to_keep=1)
        features=torch.stack(pooled).mean(0).detach(); query=torch.stack(queries).mean(0).detach()
        return Context(messages,batch,history,features,query,donors,batch['input_ids'].detach())

    def setup(self,kind='gru',hidden=128,rank=8,alpha=8.,gate_mode='signed'):
        self.controller=CreditController(self.feature_dim,hidden=hidden,kind=kind).to(self.device)
        self.bank=LowRankBank(self.widths,rank,alpha).to(self.device)
        self.gate_mode=gate_mode
        self.metadata=dict(model_path=self.model_path,revision=self.revision,layers=self.layers,target=self.target,
            max_pixels=self.max_pixels,max_tokens=self.max_tokens,kind=kind,hidden=hidden,rank=rank,alpha=alpha,
            gate_mode=gate_mode,four_bit=self.four_bit,feature_dim=self.feature_dim,
            control='pre_rope_history_lowrank_residual',prefill='clean_features_then_controlled_prefill')

    def gates(self,context):
        if self.controller is None: return None
        return gate_values(self.controller(context.features,context.query),self.gate_mode,self.temperature)

    def _callbacks(self,context,controlled=False,patch=None,fixed_projectors=None):
        callbacks={}
        gates=self.gates(context) if controlled else None
        if controlled:
            if self.bank is None: raise ValueError('No trained bank loaded')
            self.bank.reset_energy()
        for n in self.names:
            def apply(_m,_i,x,n=n):
                if x.shape[1]<=1: return x  # cached one-token decode: history is already transformed
                if patch is not None:
                    j,d=patch
                    if not context.donors: raise ValueError('Clean donor capture required')
                    return patch_span(x,context.spans[j],context.donors[n][d])
                if fixed_projectors is not None:
                    p=fixed_projectors[n].to(x.device,dtype=torch.float32)
                    y=x.clone()
                    for s,e in context.spans: y[:,s:e]=(x[:,s:e].float()@p).to(x.dtype)
                    return y
                if controlled: return self.bank.apply(n,x,context.spans,gates)
                return x
            callbacks[n]=apply
        return callbacks

    def score(self,context,action,controlled=False,patch=None,normalization='mean',fixed_projectors=None):
        msgs=context.messages+[{'role':'assistant','content':[{'type':'text','text':str(action)}]}]
        batch=self._batch(msgs,False); pl=context.prompt_ids.shape[1]
        if not torch.equal(batch['input_ids'][:,:pl],context.prompt_ids):
            raise ValueError('Candidate formatting changed prompt tokens; no candidate-dependent feature extraction is permitted')
        with hooks(self.model,self._callbacks(context,controlled,patch,fixed_projectors)):
            indices=torch.arange(pl-1,batch['input_ids'].shape[1]-1,device=self.device)
            out=self.model(**batch,use_cache=False,logits_to_keep=indices)
            targets=batch['input_ids'][:,pl:]
            total=torch.log_softmax(out.logits.float(),-1).gather(-1,targets[...,None]).sum()
            count=targets.numel()
        if normalization=='sum': return total
        if normalization=='mean': return total/count
        raise ValueError(normalization)

    @torch.no_grad()
    def generate(self,record,controlled=False,temperature=0.,max_new_tokens=160,fixed_projectors=None):
        if not controlled and fixed_projectors is None:
            batch=self._batch(prepare_messages(record),True)
            sampling={'do_sample':False} if temperature<=0 else {'do_sample':True,'temperature':temperature,'top_p':1.0}
            ids=self.model.generate(**batch,use_cache=True,max_new_tokens=max_new_tokens,**sampling)
            return self.processor.decode(ids[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        context=self.context(record)
        return self.generate_context(context,controlled,temperature,max_new_tokens,fixed_projectors=fixed_projectors)

    @torch.no_grad()
    def generate_context(self,context,controlled=False,temperature=0.,max_new_tokens=160,patch=None,fixed_projectors=None):
        sampling={'do_sample':False} if temperature<=0 else {'do_sample':True,'temperature':temperature,'top_p':1.0}
        with hooks(self.model,self._callbacks(context,controlled,patch,fixed_projectors=fixed_projectors)):
            ids=self.model.generate(**context.batch,use_cache=True,max_new_tokens=max_new_tokens,**sampling)
        result=self.processor.decode(ids[0,context.prompt_ids.shape[1]:],skip_special_tokens=True).strip()
        return result

    def save(self,path,extra=None):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'metadata':dict(self.metadata,**(extra or {})),
                    'controller':self.controller.state_dict(),'bank':self.bank.state_dict()},p)

    @classmethod
    def from_checkpoint(cls,path,device='cuda',model_override=None):
        p=torch.load(path,map_location='cpu',weights_only=True); m=p['metadata']
        e=cls(model_override or m['model_path'],m['layers'],m['target'],device,m['max_pixels'],m['max_tokens'],m['revision'],m.get('four_bit',False))
        e.setup(m['kind'],m['hidden'],m['rank'],m['alpha'],m['gate_mode'])
        e.controller.load_state_dict(p['controller']);e.bank.load_state_dict(p['bank']);e.metadata=m
        e.controller.eval();e.bank.eval();return e
