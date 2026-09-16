"""Convert existing logged candidate rows and verify externally assigned splits."""
import argparse,json
from pathlib import Path
from .schema import read_jsonl,write_jsonl,normalize_records,validate_splits,resolve_images

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--test',required=True)
    p.add_argument('--output',required=True);p.add_argument('--image-root',default='.')
    p.add_argument('--group-key',default='task_id');a=p.parse_args()
    splits={}
    for name in ['train','validation','test']:
        rows=resolve_images(normalize_records(read_jsonl(getattr(a,name))),a.image_root)
        for r in rows:
            r['split']=name
            for image in [r.get('image')]+r.get('history_images',[]):
                if image and not Path(image).exists():raise FileNotFoundError(image)
        splits[name]=rows
    manifest=validate_splits(splits,a.group_key);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for name,rows in splits.items():write_jsonl(out/f'{name}.jsonl',rows)
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
