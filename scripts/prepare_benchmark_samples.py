#!/usr/bin/env python3
"""Prepare an explicit sample pool, private references and official estimates.

Only public videos are registered as agent inputs. This does not run experiments.
Downloads use pinned revisions and official LFS hashes; packed worlds are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from prepare_evaluation_resources import fetch, fetch_parallel, request_for, digest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from orchestration.datasets import import_cases
from orchestration.storage.repository import inside, save


def remote_files(repo, revision):
    url = f'https://huggingface.co/api/datasets/4DCodeBench/{repo}/tree/{revision}?recursive=true&limit=1000'
    files = {}
    while url:
        with urllib.request.urlopen(request_for(url), timeout=60) as response:
            rows, link = json.load(response), response.headers.get('Link', '')
        files.update({row['path']: row for row in rows if row['type'] == 'file'})
        url = next((part.split('<')[1].split('>')[0] for part in link.split(',')
                    if 'rel="next"' in part), None)
    return files


def prepare(selection, root, workers):
    cases = selection['cases']
    prefix = selection.get('id', f'diverse-{len(cases)}')
    if not prefix.replace('-', '').isalnum():
        raise ValueError('Selection id must contain only letters, numbers and hyphens')
    if len({row['case'] for row in cases}) != len(cases):
        raise ValueError('Duplicate sample')
    revisions = selection['repositories']
    sources = {repo: remote_files(repo, rev) for repo, rev in revisions.items()}
    save(root/f'manifests/{prefix}-sources.json', {'revisions': revisions, 'files': sources})
    jobs, videos, packed = [], {}, []

    def add(repo, path, target):
        row = sources[repo][path]
        expected = row.get('lfs', {}).get('oid')
        if not expected:
            raise ValueError('Expected official LFS SHA256: ' + path)
        jobs.append({'repo': repo, 'revision': revisions[repo], 'file': path,
                     'target': target, 'sha256': expected, 'bytes': row['size']})
        return expected

    for row in cases:
        kind, name = row['case'].split('/')
        if kind not in {'real', 'synthetic'} or not name or name in {'.', '..'}:
            raise ValueError('Invalid case: ' + row['case'])
        repo = 'Dataset-Real-World' if kind == 'real' else 'Dataset-Synthetic'
        video = root/'cases'/row['case']/'reference.mp4'
        if f'videos/{name}.mp4' in sources[repo]:
            videos[row['case']] = add(repo, f'videos/{name}.mp4', video)
        else:
            # Official link-only videos have no downloadable benchmark-file hash.
            # Require a frozen local rebuild and its separately archived provenance.
            rebuild = row.get('video_rebuild', {})
            receipt = inside(root, rebuild.get('receipt', 'missing-rebuild-receipt'))
            if (kind != 'real' or not video.is_file() or not receipt.is_file()
                    or digest(video) != rebuild.get('sha256')
                    or digest(receipt) != rebuild.get('receipt_sha256')):
                raise ValueError('Prepare and freeze the official link-only rebuild first: '+row['case'])
            videos[row['case']] = rebuild['sha256']
        if kind == 'real':
            add(repo, f'annotations/{name}/dynamic_mask.npz', root/'data'/row['case']/'annotation/dynamic_mask.npz')
        else:
            path = root/'packed'/f'{name}.h5'
            expected = add(repo, f'worlds/{name}.h5', path)
            packed.append((row['case'], path, expected))
        estimate_prefix = row['case'] + '/'
        estimates = [p for p in sources['Dataset-Model-Estimation'] if p.startswith(estimate_prefix)]
        required = {'semantic_dinov3.npz', 'semantic_tips.npz', 'geophys_dinov3.npz'}
        if kind == 'real': required |= {'depth.h5', 'flow.h5', 'tracks.npz', 'moge.npz'}
        if {p.removeprefix(estimate_prefix) for p in estimates} != required:
            raise ValueError('Unexpected estimate inventory: ' + row['case'])
        for path in sorted(estimates):
            add('Dataset-Model-Estimation', path, root/'data'/row['case']/'estimates'/path.removeprefix(estimate_prefix))

    required = sum(job['bytes'] for job in jobs if not job['target'].is_file())
    world_bytes = sum(path.stat().st_size if path.exists() else
                      sources['Dataset-Synthetic'][f'worlds/{path.stem}.h5']['size'] for _, path, _ in packed)
    if shutil.disk_usage(root).free < required + 3*world_bytes + 5*1024**3:
        raise ValueError('Insufficient free space for downloads, world conversion and reserve')
    report = {'selection': selection, 'planned_download_bytes': sum(j['bytes'] for j in jobs),
              'status': 'downloading', 'files': [], 'worlds': []}
    report_path = root/f'{prefix}-preparation.json'
    save(report_path, report)

    def download(job):
        url = f"https://huggingface.co/datasets/4DCodeBench/{job['repo']}/resolve/{job['revision']}/{job['file']}"
        result = (fetch_parallel(url, job['target'], job['sha256'], job['bytes'])
                  if job['bytes'] >= 128*1024**2 else fetch(url, job['target'], job['sha256']))
        result.update(repo=job['repo'], revision=job['revision'], source_path=job['file'])
        if result['status'] == 'verified' and job['target'].suffix == '.npz':
            with zipfile.ZipFile(job['target']) as archive:
                if archive.testzip() is not None: raise ValueError('NPZ CRC failure: '+job['file'])
            result['zip_crc_ok'] = True
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(download, job) for job in jobs]):
            result = future.result()
            report['files'].append(result)
            save(report_path, report)
            print(json.dumps({'completed': len(report['files']), 'total': len(jobs),
                              'file': result['source_path'], 'status': result['status']}), flush=True)
    if any(r['status'] != 'verified' for r in report['files']):
        report['status'] = 'failed'; save(report_path, report)
        raise RuntimeError('Unverified files; inputs not registered')

    python = PROJECT/'data/resources/perception/python/bin/python3.12'
    env = {**os.environ, 'PYTHONPATH': str(PROJECT/'data/resources/scorer/packages') + ':' +
           str(PROJECT/'data/resources/perception/.venv/lib/python3.12/site-packages')}
    # Use the unmodified official lossless unpacker with the owned HDF5 environment.
    unpack = '''import importlib.util,sys,pathlib,hashlib,shutil,json,numpy as np
source,packed,target=map(pathlib.Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('official_download',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def hashes(folder):
 return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()}
def equivalent(before,after):
 if set(hashes(before))!=set(hashes(after)):return False
 for p in after.rglob('*'):
  if not p.is_file():continue
  q=before/p.relative_to(after)
  if p.suffix!='.npz':
   if p.read_bytes()!=q.read_bytes():return False
  else:
   # ZIP metadata can differ across unpacking runs. Compare native arrays exactly.
   with np.load(p,allow_pickle=False) as a,np.load(q,allow_pickle=False) as b:
    if set(a.files)!=set(b.files):return False
    for key in a.files:
     x,y=a[key],b[key]
     if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y,equal_nan=x.dtype.kind in 'fc'):return False
 return True
if target.exists():
 candidate=target.with_name('world.verification')
 if candidate.exists():raise ValueError('Unfinished verification directory')
 module.unpack(packed,candidate)
 try:
  if not equivalent(target,candidate):raise ValueError('Existing world arrays/files differ from pinned source')
 finally:shutil.rmtree(candidate)
else:module.unpack(packed,target)
print(json.dumps(hashes(target)))
'''
    for case, path, expected in packed:
        target = root/'data'/case/'world'
        result = subprocess.run([str(python), '-c', unpack,
            str(root/'4DCodeBench/scripts/download_data.py'), str(path), str(target)],
            env=env, text=True, capture_output=True, timeout=1200)
        if result.returncode:
            report.update(status='unpack_failed', error={'case': case, 'stderr': result.stderr})
            save(report_path, report)
            raise RuntimeError('Official world unpack failed: '+case+'\n'+result.stderr)
        hashes = json.loads(result.stdout)
        report['worlds'].append({'case': case, 'packed_sha256': expected, 'files_sha256': hashes})
        save(report_path, report)
        print(json.dumps({'world': case, 'status': 'unpacked_verified', 'files': len(hashes)}), flush=True)

    manifest = {**selection, 'cases': [{**row, 'sha256': videos[row['case']],
        'dataset': 'Dataset-Real-World' if row['case'].startswith('real/') else 'Dataset-Synthetic',
        'dataset_revision': revisions['Dataset-Real-World' if row['case'].startswith('real/') else 'Dataset-Synthetic']}
        for row in cases]}
    save(root/f'{prefix}-manifest.json', manifest)
    imported = import_cases(manifest, root, PROJECT)
    report.update(status='prepared', registered_cases=len(imported))
    save(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=PROJECT/'evaluation/benchmark-samples.json')
    parser.add_argument('--root', type=Path, default=PROJECT/'data/resources/benchmark')
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    prepare(json.loads(args.selection.read_text()), args.root.resolve(), args.workers)
