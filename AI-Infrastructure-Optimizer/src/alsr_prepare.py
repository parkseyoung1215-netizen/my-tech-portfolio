"""Read-only Ollama inspection and isolated public HF checkpoint acquisition."""
import hashlib,importlib.metadata,json,os,platform,struct,subprocess,urllib.request
from pathlib import Path
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path('/private/tmp/alsr_models_20261105')
REPOS=['Qwen/Qwen2.5-0.5B-Instruct','Qwen/Qwen2.5-1.5B-Instruct']

def gguf_metadata(path):
    with Path(path).open('rb') as f:
        def num(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize(fmt)))[0]
        def string():return f.read(num('Q')).decode()
        def value(t):
            if t==8:return string()
            if t==9:
                item,n=num('I'),num('Q');return [value(item) for _ in range(n)]
            return num({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
        if f.read(4)!=b'GGUF':raise ValueError('Not GGUF')
        version,nt,nk=num('I'),num('Q'),num('Q');meta={}
        for _ in range(nk):
            key=string();v=value(num('I'))
            if key.startswith('qwen2.') or key in ('general.name','general.architecture','general.file_type','general.quantization_version'):meta[key]=v
    return dict(version=version,tensors=nt,bytes=Path(path).stat().st_size,metadata=meta)

def prepare():
    os.environ['HF_HOME']='/private/tmp/alsr_hf_home_20261105';os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    from huggingface_hub import HfApi,snapshot_download
    import mlx.core as mx
    result={'platform':platform.platform(),'machine':platform.machine(),'physical_memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'])),
      'mlx_device':mx.device_info(),'packages':{name:importlib.metadata.version(name) for name in ('mlx','mlx-lm','transformers','huggingface-hub','numpy')},'ollama':{},'hf_models':[]}
    base=Path.home()/'.ollama/models'
    for tag in ('0.5b','1.5b'):
        manifest_path=base/'manifests/registry.ollama.ai/library/qwen2.5'/tag;m=json.loads(manifest_path.read_text());layer=next(l for l in m['layers'] if l['mediaType'].endswith('.model'));path=base/'blobs'/layer['digest'].replace(':','-')
        license_path=base/'blobs'/next(l['digest'].replace(':','-') for l in m['layers'] if l['mediaType'].endswith('.license'))
        result['ollama'][tag]={'manifest':m,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'model_path':str(path),**gguf_metadata(path),'license_sha256':hashlib.sha256(license_path.read_bytes()).hexdigest(),'license_header':license_path.read_text()[:100].strip()}
    for repo in REPOS:
        info=HfApi(token=False).model_info(repo);print('Downloading public checkpoint '+repo+' @ '+info.sha,flush=True)
        folder=snapshot_download(repo,revision=info.sha,token=False,cache_dir=str(CACHE),allow_patterns=['*.safetensors','*.json','merges.txt','*.model','LICENSE','README.md'])
        path=Path(folder);config=json.loads((path/'config.json').read_text());result['hf_models'].append({'repo':repo,'revision':info.sha,'path':str(path),'config':config,'files':{str(p.name):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in path.iterdir() if p.is_file()}})
        print('Checkpoint ready '+repo,flush=True)
    result['ollama_api_version']=json.load(urllib.request.urlopen('http://localhost:11434/api/version',timeout=10))
    result['gpu_elapsed_api']='MLX exposes capture, not a per-request GPU elapsed counter; synchronized wall is not GPU time.'
    path=ROOT/'data'/('alsr_setup_'+uuid4().hex+'.json')
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('ALSR SETUP '+str(path),flush=True);return path

if __name__=='__main__':prepare()
