"""Frozen public 4DCodeBench sample registration and explicit split selection."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
from .storage.repository import inside, load, save, sha

SPLITS = {'smoke', 'evolution', 'validation', 'held-out'}


def validate_manifest(manifest, cases_root):
    """An explicit ordered split; never invent an official RSI partition."""
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('cases'), list):
        raise ValueError('Manifest requires schema_version=1 and ordered cases')
    seen, hashes = set(), {}
    for row in manifest['cases']:
        case = PurePosixPath(row['case'])
        if len(case.parts) != 2 or case.parts[0] not in {'real', 'synthetic'} or '..' in case.parts:
            raise ValueError('case must be real/<name> or synthetic/<name>')
        if row['case'] in seen or row.get('split') not in SPLITS:
            raise ValueError('Cases must be unique and have an explicit split')
        seen.add(row['case'])
        video = inside(cases_root, str(case / 'reference.mp4'))
        digest = sha(video)
        if digest != row.get('sha256'):
            raise ValueError('Case video hash differs from frozen manifest: ' + row['case'])
        if digest in hashes and hashes[digest] != row['split']:
            raise ValueError('The same video cannot cross split boundaries')
        hashes[digest] = row['split']
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def import_cases(manifest, root, project):
    """Copy only public input videos. Private references never enter task inputs."""
    root, project = Path(root), Path(project)
    digest = validate_manifest(manifest, root / 'cases')
    previous = load(project / 'data/benchmarks/registry.json', {})
    registered = previous.get('cases', [])
    for row in manifest['cases']:
        for old in registered:
            if old['case'] == row['case'] and (old['sha256'], old['split']) != (row['sha256'], row['split']):
                raise ValueError('An imported case cannot silently change content or split')
            if old['sha256'] == row['sha256'] and old['split'] != row['split']:
                raise ValueError('A previously imported video cannot cross split boundaries')
    cases = []
    for row in manifest['cases']:
        path = f"{row['split']}/4dcodebench/{row['case']}/reference.mp4"
        target = project / 'data/inputs' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != row['sha256']:
            raise ValueError('Refusing to overwrite a changed benchmark input')
        if not target.exists():
            shutil.copyfile(root / 'cases' / row['case'] / 'reference.mp4', target)
        cases.append({**row, 'path': path, 'manifest_sha256': digest})
    # Preserve registrations for older public files: otherwise an old held-out
    # file left on disk could accidentally become an unrestricted manual input.
    merged = {r['path']: r for r in registered}
    merged.update({r['path']: r for r in cases})
    manifests = {**previous.get('manifests', {}), digest: manifest}
    if previous.get('sha256') and previous.get('manifest'):
        manifests.setdefault(previous['sha256'], previous['manifest'])
    save(project / 'data/benchmarks/registry.json', {'manifest': manifest, 'cases': list(merged.values()),
                                                'manifests': manifests, 'sha256': digest})
    return cases
