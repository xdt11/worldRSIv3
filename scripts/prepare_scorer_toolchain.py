#!/usr/bin/env python3
"""Prepare a workspace-owned C toolchain for the official Triton DTW kernels.

Package signatures are checked by apt in a temporary scorer-image container.
The resulting Debian packages, licenses, compiler, headers and SHA256 manifest
stay in data/resources/scorer; the runtime container remains read-only/offline.
"""
import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import subprocess

COMPILER = '''#!/bin/sh
root=/opt/scorer-toolchain
export LD_LIBRARY_PATH="$root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/usr/bin/gcc" --sysroot="$root" \\
  -B"$root/usr/lib/gcc/x86_64-linux-gnu/12/" -B"$root/usr/bin/" \\
  -L"$root/usr/lib/x86_64-linux-gnu" -L/lib/x86_64-linux-gnu \\
  -isystem /opt/resources/perception/python/include/python3.12 "$@"
'''


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def prepare(resources, image):
    scorer = resources / 'scorer'
    target = scorer / 'toolchain'
    (target / 'debs/partial').mkdir(parents=True, exist_ok=True)
    script = '''apt-get update -qq
apt-get install --download-only -y --no-install-recommends -o Dir::Cache::archives=/toolchain/debs gcc libc6-dev
for package in /toolchain/debs/*.deb; do dpkg-deb -x "$package" /toolchain; done
chmod -R a+rX /toolchain
chown "$1:$2" /toolchain
'''
    subprocess.run(['docker', 'run', '--rm', '--pull=never',
        '--mount', f'type=bind,src={target},dst=/toolchain',
        '--env', 'DEBIAN_FRONTEND=noninteractive', image, 'sh', '-eu', '-c', script,
        'prepare-toolchain', str(os.getuid()), str(os.getgid())], check=True)
    wrapper = target / 'cc'
    wrapper.write_text(COMPILER)
    wrapper.chmod(0o755)
    files = {str(p.relative_to(target)): digest(p) for p in sorted(target.rglob('*'))
        if p.is_file() and not p.is_symlink() and p.name != 'identity.json'}
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    manifest = {'image': image, 'image_id': image_id, 'platform': 'Debian 12 amd64',
        'purpose': 'Official CUDA Triton DTW driver C extension; no scorer formula changes', 'sha256': files}
    (target / 'identity.json').write_text(json.dumps(manifest, indent=2))
    identity_path = scorer / 'identity.json'
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text())
    else:
        packages = scorer / 'packages'
        identity = {'schema_version': 1, 'platform': 'Debian 12 amd64', 'image_id': image_id,
                    'packages': {dist.metadata['Name']: dist.version for dist in
                                 sorted(distributions(path=[str(packages)]), key=lambda item: item.metadata['Name'].lower())},
                    'native_sha256': {str(p.relative_to(packages)): digest(p) for p in sorted(packages.rglob('*.so'))},
                    'probe': 'not_run'}
    identity['toolchain_manifest_sha256'] = digest(target / 'identity.json')
    identity_path.write_text(json.dumps(identity, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', type=Path, default=Path(__file__).resolve().parents[1] / 'data/resources')
    parser.add_argument('--image', default='worldrsi:local')
    args = parser.parse_args()
    manifest = prepare(args.resources.resolve(), args.image)
    print(json.dumps({'status': 'prepared', 'files': len(manifest['sha256'])}))
