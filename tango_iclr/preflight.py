"""Environment audit only: never installs privileged software automatically."""
import argparse,importlib.util,json,os,platform,shutil,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mobile',action='store_true');p.add_argument('--output',default='preflight.json')
    a=p.parse_args();info={'python':platform.python_version(),'platform':platform.platform(),
        'packages':{x:bool(importlib.util.find_spec(x)) for x in ['torch','transformers','peft','PIL','numpy','yaml','gymnasium']},
        'kvm_present':Path('/dev/kvm').exists(),'kvm_read_write':os.access('/dev/kvm',os.R_OK|os.W_OK),
        'docker':bool(shutil.which('docker')),'nvidia_smi':bool(shutil.which('nvidia-smi'))}
    try:
        import torch
        info.update(cuda_available=torch.cuda.is_available(),torch_version=torch.__version__)
        if torch.cuda.is_available():info['gpu']=[{'name':torch.cuda.get_device_name(i),'memory_GiB':torch.cuda.get_device_properties(i).total_memory/2**30} for i in range(torch.cuda.device_count())]
    except ImportError:info['cuda_available']=False
    info['blocked']=[]
    if not info['cuda_available']:info['blocked'].append('No CUDA; CPU unit tests only')
    for pkg in ['torch','transformers','PIL','numpy','yaml']:
        if not info['packages'][pkg]:info['blocked'].append(f'Missing {pkg}')
    if a.mobile and not info['kvm_read_write']:info['blocked'].append('Mobile runtime requires usable KVM on its environment host')
    destination=Path(a.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
    if info['blocked']:raise SystemExit(2)
if __name__=='__main__':main()
