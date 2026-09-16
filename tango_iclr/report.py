"""Generate paired statistics and publication-ready aggregate CSV, without filling missing cells."""
import argparse,csv,json
from pathlib import Path
from .schema import read_jsonl
from .stats import paired_summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    result=paired_summary(read_jsonl(a.method),read_jsonl(a.baseline))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    with dest.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result));writer.writeheader();writer.writerow(result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
