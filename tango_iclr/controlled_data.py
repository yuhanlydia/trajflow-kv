"""Fresh deterministic synthetic GUI diagnostic, NOT native application episodes."""
from __future__ import annotations
import argparse,hashlib,json,random
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .schema import write_jsonl,digest

ALPHABET='ABCDEFGHJKLMNPQRSTUVWXYZ23456789'


def make_specs(split,template,count,seed,used=None):
    if split not in {'train','validation','test'} or count<=0:raise ValueError('Invalid split/count')
    if split=='test' and template=='C':raise ValueError('Template C has been analyzed; use fresh D')
    if template not in {'A','B','D'}:raise ValueError('Supported templates A/B/D')
    used=set() if used is None else used;rng=random.Random(seed)
    targets=[i%8 for i in range(count)];rng.shuffle(targets);specs=[]
    for i,target in enumerate(targets):
        codes=[]
        while len(codes)<10:
            c=''.join(rng.choice(ALPHABET) for _ in range(5))
            if c not in used:used.add(c);codes.append(c)
        options=[codes[0],codes[1],*codes[5:10]];rng.shuffle(options);options.insert(target,codes[2])
        key=f'sigma-qwen3-v2:{split}:{seed}:{i}'
        specs.append(dict(prefix_id=key,task_id=key,split=split,template=template,
            family='interference_chain',seed=seed+i,nonce_pool=codes,slot_codes=options,
            target_index=target,critical_step=True,donor_indices=[1,3],
            candidates=[f'select_slot_{k+1}' for k in range(8)],
            instruction='Use the most recent UPDATE for the target record. Reference cards do not change the target. '
                        'Select the current numbered slot whose code equals the final target. '
                        'Answer with select_slot_N, where N is the printed slot number.'))
    return specs


def _font(path,size):
    for p in ([path] if path else ['DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']):
        try:return ImageFont.truetype(str(p),size)
        except OSError:pass
    raise FileNotFoundError('TrueType font required: pass --font /path/to/font.ttf')


def _render(spec,path,block=None,font=None):
    im=Image.new('RGB',(448,448),'white');d=ImageDraw.Draw(im)
    f=_font(font,23);small=_font(font,16);large=_font(font,42);template=spec['template']
    d.rectangle((10,10,438,60),outline='black',width=2)
    d.text((22,24),{'A':'Record console','B':'Record workspace','D':'Record manager'}[template],font=f,fill='black')
    if block is None:
        slots=list(enumerate(spec['slot_codes']))
        if template=='B':slots=slots[::-1]
        if template=='D':slots=slots[2:]+slots[:2]
        for pos,(slot,code) in enumerate(slots):
            left=14+218*(pos%2);top=76+88*(pos//2)
            d.rectangle((left,top,left+204,top+78),outline='black',width=2)
            d.text((left+8,top+6),f'Slot {slot+1}',font=small,fill='black')
            d.text((left+8,top+35),code,font=f,fill='black')
    else:
        role=['INITIAL TARGET','REFERENCE CARD','UPDATE TARGET','REFERENCE CARD','UPDATE TARGET'][block]
        code=spec['nonce_pool'][[0,3,1,4,2][block]];y={'A':125,'B':150,'D':135}[template]
        d.text((28,y),role,font=f,fill='black');d.rectangle((24,y+52,424,y+146),outline='black',width=2)
        d.text((56,y+69),code,font=large,fill='black')
        d.text((28,385),'Reference cards are not updates.' if block in [1,3] else 'Target record value',font=small,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path)


def generate_dataset(output,train_count=500,validation_count=50,test_count=50,font=None,seed=20260917):
    root=Path(output).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a new empty data directory; never overwrite sealed data')
    root.mkdir(parents=True,exist_ok=True);used=set()
    manifest=dict(schema='sigma.controlled.v2',data_kind='synthetic_controlled_not_native_gui',
                  seed=seed,splits={},generator_version=2,image_size=[448,448],test_policy='fresh D; C remains historical')
    hashes={}
    for s,t,n,offset in [('train','A',train_count,0),('validation','B',validation_count,100000),('test','D',test_count,200000)]:
        specs=make_specs(s,t,n,seed+offset,used);rows=[]
        for i,x in enumerate(specs):
            folder=root/'images'/s/f'{i:04d}';hist=[]
            for j in range(5):
                p=folder/f'h{j}.png';_render(x,p,j,font);hist.append(str(p))
            p=folder/'current.png';_render(x,p,None,font)
            r={k:v for k,v in x.items() if k not in {'nonce_pool','slot_codes'}}
            r.update(image=str(p),history_images=hist,history_actions=[],optimal_indices=[x['target_index']]);rows.append(r)
            for name in hist+[str(p)]:hashes[str(Path(name).relative_to(root))]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        write_jsonl(root/f'{s}.jsonl',rows);write_jsonl(root/f'{s}_private_generation.jsonl',specs)
        manifest['splits'][s]=dict(template=t,prefixes=n,seed=seed+offset,
            target_counts=dict(Counter(x['target_index'] for x in rows)),sha256=digest(rows))
    manifest.update(unique_nonce_count=len(used),image_hashes=hashes)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--train-count',type=int,default=500);p.add_argument('--validation-count',type=int,default=50)
    p.add_argument('--test-count',type=int,default=50);p.add_argument('--font');p.add_argument('--seed',type=int,default=20260917)
    a=p.parse_args();m=generate_dataset(a.output,a.train_count,a.validation_count,a.test_count,a.font,a.seed)
    print(json.dumps({k:v for k,v in m.items() if k!='image_hashes'},indent=2))
if __name__=='__main__':main()
