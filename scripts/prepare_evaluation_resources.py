#!/usr/bin/env python3
"""Fetch pinned official scorer weights and estimates for the two smoke cases.

Only publicly accessible files are requested; gated files remain explicit errors.
No reference file is added to an agent input directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import urllib.error
import urllib.request
from urllib.parse import urlparse

MODELS = [
 ('camenduru/dinov3-vitl16-pretrain-lvd1689m','3c276edd87d6f6e569ff0c4400e086807d0f3881','dinov3-vitl16-pretrain-lvd1689m','model.safetensors','dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179',True),
 ('google/tipsv2-l14','52847a71eb02a3082c370fbcd98cd799687bd4d0','tipsv2-l14','model.safetensors','c75707eabca655e641a6c47a59dd26b347290ceadb993b34af184e7f654c5809',True),
 ('depth-anything/Video-Depth-Anything-Large','7aafbcb5c6af0bac741aad2b6471894fb4761afa','video-depth-anything-large','video_depth_anything_vitl.pth','43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa',False),
 ('facebook/cotracker3','bf55ea50d4390e1820a267f131cd6587240fb2c5','cotracker3','scaled_offline.pth','2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834',False),
 ('Ruicheng/moge-3-vitl','184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be','moge-3-vitl','model.pt','9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925',False),
 ('BAAI/Uni3D','3d8233b76aa350d72f6213ecd2123c2026b42355','uni3d-l','modelzoo/uni3d-l/model.pt','540a3cafac52c251cbda1844d109bc598f142b4f5a2118db87cd722b01800c20',False),
]
# Public redistribution: identical official checkpoint and config/license blobs.
DINO_ORIGIN = {'repo': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
 'revision': 'ea8dc2863c51be0a264bab82070e3e8836b02d51',
 'distribution_repo': 'camenduru/dinov3-vitl16-pretrain-lvd1689m',
 'distribution_revision': '3c276edd87d6f6e569ff0c4400e086807d0f3881',
 'conversion': False,
 'git_blobs': {'config.json': 'ab2a8b183fd4fd5213e9898a7f499c7c857b1a86',
  'preprocessor_config.json': '0126173c1ab7b75bbd9102e37de8a56ffe0a012b',
  'LICENSE.md': 'f531b1e6b5ab2318957bbf8ad1bda9f800a23e17'}}
WEIGHT_SIZES = {MODELS[0][4]: 1212559808, MODELS[1][4]: 1951820760, MODELS[5][4]: 614861649}
RANGE_BYTES = 32 * 1024**2
ESTIMATE_REVISION = '671db39a7ca642005478a2d2cd913a3455e1a235'
CASES = ('real/wisa_80k_19_domino_line_triggers', 'synthetic/B_30')


def digest(path):
 d=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024**2),b''): d.update(chunk)
 return d.hexdigest()


def request_for(url, **kwargs):
 request=urllib.request.Request(url,**kwargs)
 token=os.environ.get('HF_TOKEN')
 if token and urlparse(url).hostname=='huggingface.co':
  # Unredirected headers authenticate HF without forwarding credentials to CDNs.
  request.add_unredirected_header('Authorization','Bearer '+token)
 return request


def git_blob(path):
 raw=path.read_bytes()
 return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def fetch(url, target, expected=None, expected_git_blob=None):
 target.parent.mkdir(parents=True,exist_ok=True)
 temp=target.with_name(target.name+'.partial')
 try:
  if not target.exists() and temp.exists() and expected and digest(temp)==expected:
   temp.replace(target)
  if target.exists() and expected_git_blob and git_blob(target)!=expected_git_blob:
   target.replace(target.with_name(target.name+'.rejected'))
  if target.exists() and expected and digest(target)!=expected:
   target.unlink()  # An unverified file must never satisfy scorer readiness.
  if not target.exists():
   failures=0
   while True:
    offset=temp.stat().st_size if temp.exists() else 0
    request=request_for(url,headers={'Range':f'bytes={offset}-{offset+32*1024**2-1}','Accept-Encoding':'identity'})
    try:
     with urllib.request.urlopen(request,timeout=120) as stream:
      header=stream.headers.get('Content-Range','')
      match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',header)
      if stream.status==206:
       if not match or int(match[1])!=offset: raise ValueError('Server returned an unexpected byte range')
       total=int(match[3]);limit=int(match[2])+1
      elif stream.status==200 and offset==0:
       total=int(stream.headers.get('Content-Length','0'));limit=total
      else:
       raise ValueError('Server does not support the required resumable byte range')
      if total>4*1024**3: raise ValueError('Individual download exceeds 4 GiB')
      with temp.open('ab') as out:
       while chunk:=stream.read(1024**2):
        offset+=len(chunk)
        if offset>4*1024**3: raise ValueError('Individual download exceeds 4 GiB')
        out.write(chunk)
      if limit and offset!=limit: raise OSError(f'Incomplete range: {offset} of {limit} bytes')
      if not total or offset==total: break
    except urllib.error.HTTPError:
     raise
    except (OSError,TimeoutError) as exc:
     failures+=1
     if failures>=4: raise exc
   if expected and digest(temp)!=expected:
    actual=digest(temp);temp.unlink()
    raise ValueError('Official SHA256 mismatch: '+actual)
   if expected_git_blob and git_blob(temp)!=expected_git_blob:
    raise ValueError('Pinned metadata Git blob mismatch')
   temp.replace(target)
  return {'git_blob':git_blob(target) if expected_git_blob else None,'path':str(target),'status':'verified' if expected or expected_git_blob else 'downloaded','bytes':target.stat().st_size,'sha256':digest(target),'url':url}
 except urllib.error.HTTPError as exc:
  return {'path':str(target),'status':'unavailable','http_status':exc.code,'url':url}
 except Exception as exc:
  return {'path':str(target),'status':'error','reason':type(exc).__name__+': '+str(exc),'url':url}



def fetch_parallel(url, target, expected, size, workers=4):
 """Preserve a sequential prefix and resumable ranges; publish only verified bytes."""
 if target.exists():
  return fetch(url,target,expected)
 prefix=target.with_name(target.name+'.partial')
 if prefix.exists() and digest(prefix)==expected:
  return fetch(url,target,expected)
 parts=target.with_name('.'+target.name+'.parts')
 parts.mkdir(parents=True,exist_ok=True)
 start=prefix.stat().st_size if prefix.exists() else 0
 state={'prefix_bytes':start,'size':size,'sha256':expected}
 state_file=parts/'state.json'
 try:
  if size>4*1024**3 or start>size:raise ValueError('Invalid checkpoint size')
  if state_file.exists() and json.loads(state_file.read_text())!=state:
   raise ValueError('Retained range state does not match this checkpoint/prefix')
  state_file.write_text(json.dumps(state))
  spans=[(i,min(i+RANGE_BYTES-1,size-1)) for i in range(start,size,RANGE_BYTES)]
  def download(span):
   begin,end=span;path=parts/f'{begin}-{end}.bin';partial=path.with_suffix('.partial')
   if path.exists() and path.stat().st_size==end-begin+1:return path
   attempts=0
   while True:
    offset=partial.stat().st_size if partial.exists() else 0
    if offset==end-begin+1:break
    try:
     request=request_for(url,headers={'Range':f'bytes={begin+offset}-{end}','Accept-Encoding':'identity'})
     with urllib.request.urlopen(request,timeout=120) as stream:
      match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',stream.headers.get('Content-Range',''))
      if stream.status!=206 or not match or tuple(map(int,match.groups()))!=(begin+offset,end,size):
       raise ValueError('Unexpected ranged checkpoint response')
      with partial.open('ab') as out:
       while chunk:=stream.read(1024**2):out.write(chunk)
     if partial.stat().st_size!=end-begin+1:raise OSError('Incomplete range')
    except urllib.error.HTTPError:raise
    except OSError:
     attempts+=1
     if attempts>=4:raise
   partial.replace(path);return path
  with ThreadPoolExecutor(max_workers=workers) as pool:
   list(pool.map(download,spans))
  completed=target.with_name(target.name+'.complete.partial');checksum=hashlib.sha256()
  with completed.open('wb') as out:
   for path in ([prefix] if start else [])+[parts/f'{begin}-{end}.bin' for begin,end in spans]:
    with path.open('rb') as source:
     while chunk:=source.read(1024**2):out.write(chunk);checksum.update(chunk)
  if completed.stat().st_size!=size or checksum.hexdigest()!=expected:
   raise ValueError('Assembled checkpoint does not match official size/SHA256')
  completed.replace(target);prefix.unlink(missing_ok=True);shutil.rmtree(parts)
  return fetch(url,target,expected)
 except Exception as exc:
  return {'path':str(target),'status':'error','reason':type(exc).__name__+': '+str(exc),'url':url}

def prepare(root, include_prepare_models=False):
 if shutil.disk_usage(root).free<10*1024**3: raise ValueError('Need at least 10 GiB free before optional scorer preparation')
 jobs=[];metadata_errors=[]
 for repo,revision,folder,weight,expected,metadata in MODELS:
  if not include_prepare_models and folder not in {'dinov3-vitl16-pretrain-lvd1689m','tipsv2-l14','uni3d-l'}: continue
  filename='model.pt' if folder=='uni3d-l' else weight
  jobs.append((f'https://huggingface.co/{repo}/resolve/{revision}/{weight}',root/'checkpoints'/folder/filename,expected))
  if metadata:
   try:
    with urllib.request.urlopen(request_for(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'),timeout=30) as stream: manifest=json.load(stream)
    for item in manifest.get('siblings',[]):
     name=item['rfilename']
     if (name.endswith(('.json','.py')) or (folder=='dinov3-vitl16-pretrain-lvd1689m' and name=='LICENSE.md')) and not name.startswith('.'):
      if folder=='dinov3-vitl16-pretrain-lvd1689m' and name in DINO_ORIGIN['git_blobs'] and item.get('blobId')!=DINO_ORIGIN['git_blobs'][name]:
       raise ValueError('Public DINO metadata differs from the official pinned blob')
      jobs.append((f'https://huggingface.co/{repo}/resolve/{revision}/{name}',root/'checkpoints'/folder/name,None,item.get('blobId')))
   except Exception as exc:
    metadata_errors.append({'path':str(root/'checkpoints'/folder),'status':'error','reason':'Metadata verification failed: '+str(exc)})
 if include_prepare_models:
  jobs.append(('https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth',root/'checkpoints/raft-large/raft_large_C_T_SKHT_V2.pth','ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322'))
 with urllib.request.urlopen(request_for(f'https://huggingface.co/api/datasets/4DCodeBench/Dataset-Model-Estimation/revision/{ESTIMATE_REVISION}?blobs=true'),timeout=30) as stream: manifest=json.load(stream)
 if manifest['sha']!=ESTIMATE_REVISION: raise ValueError('Estimate revision mismatch')
 for item in manifest.get('siblings',[]):
  name=item['rfilename']
  for case in CASES:
   if name.startswith(case+'/'):
    target=root/'data'/case/'estimates'/name.removeprefix(case+'/')
    jobs.append((f'https://huggingface.co/datasets/4DCodeBench/Dataset-Model-Estimation/resolve/{ESTIMATE_REVISION}/{name}',target,item.get('lfs',{}).get('sha256')))
 jobs.sort(key=lambda job: 'checkpoints' in job[1].parts)
 rows=list(metadata_errors)
 with ThreadPoolExecutor(max_workers=2) as executor:
  futures=[executor.submit(fetch_parallel,*job,size=WEIGHT_SIZES[job[2]]) if job[2] in WEIGHT_SIZES else executor.submit(fetch,*job) for job in jobs]
  for future in as_completed(futures):
   result=future.result()
   rows.append(result)
   print(json.dumps({k:v for k,v in result.items() if k!='url'}),flush=True)
   (root/'preparation.json').write_text(json.dumps({'estimate_revision':ESTIMATE_REVISION,'dinov3_source':DINO_ORIGIN,'files':rows},indent=2))
 return rows


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'data/resources/benchmark')
 parser.add_argument('--include-prepare-models',action='store_true',help='Also download models for recomputing estimates; unnecessary for provided precomputed smoke estimates')
 args=parser.parse_args()
 rows=prepare(args.root.resolve(),args.include_prepare_models)
 raise SystemExit(1 if any(row['status'] in {'error','unavailable'} for row in rows) else 0)
