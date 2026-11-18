"""Reuse local Go2 assets and policy weights, without importing their old simulator."""
from pathlib import Path
import argparse
import json
import shutil
import hashlib
import subprocess
import xml.etree.ElementTree as ET


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def prepare(destination, simulation, renderer):
    config = json.loads((renderer / 'config.json').read_text())
    repo = simulation / 'third_party/go2_rl_lab'
    assets = repo / 'source/go2_rl_lab/go2_rl_lab/assets'
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('urdf', 'meshes'):
        if (assets / name).is_dir(): shutil.copytree(assets / name, destination / 'robot' / name, dirs_exist_ok=True)
    for source, name in ((repo / 'deploy/deploy_mujoco/exported/mixed1.3.2resume_82.5k_0.6929.pt', 'moects.pt'),
                         (repo / 'deploy/deploy_mujoco/config.yaml', 'moects.yaml'),
                         (repo / 'LICENSE', 'Go2-policy.LICENSE')):
        shutil.copyfile(source, destination / name)
    # Convert visual DAE assets once outside the pinned Newton runtime. PLY
    # carries the identical triangle geometry and needs no Collada plugin in
    # each rollout sandbox. Keep the originals and record both identities.
    conversion_packages = destination / 'convert-packages'
    conversion_python = Path(__file__).resolve().parents[1] / 'data/resources/newton_env/bin/python'
    packages = ['pycollada==0.9.3','python-dateutil==2.9.0.post0','six==1.17.0']
    if not (conversion_packages / 'collada/__init__.py').is_file():
        subprocess.run([str(conversion_python),'-m','pip','install','--disable-pip-version-check',
                        '--no-deps','--target',str(conversion_packages),*packages],check=True)
    conversions=[]
    for source in sorted((destination / 'robot').rglob('*.dae')):
        target = source.with_suffix('.ply')
        subprocess.run([str(conversion_python),'-c',
            'import sys;sys.path.insert(0,sys.argv[1]);import trimesh;'
            'trimesh.load(sys.argv[2],force="mesh").export(sys.argv[3])',
            str(conversion_packages),str(source),str(target)],check=True)
        conversions.append({'source':str(source.relative_to(destination)),'source_sha256':digest(source),
                            'target':str(target.relative_to(destination)),'target_sha256':digest(target)})
    # URI resolution is relative to this copied URDF's actual mesh directory.
    for path in (destination / 'robot/urdf').glob('*.urdf'):
        xml=ET.fromstring(path.read_text())
        for mesh in xml.iter('mesh'):
            name=mesh.attrib['filename'].replace('package://go2/meshes/','meshes/')
            mesh.attrib['filename']=str(Path(name).with_suffix('.ply')) if name.endswith('.dae') else name
        path.write_text(ET.tostring(xml,encoding='unicode'))
    torch_packages = Path(config['renderer_python']).resolve().parent.parent / 'lib/python3.12/site-packages'
    paths = {'navigation_python': str((simulation / '.venv/bin/python').absolute()),
             'navigation_repo': str(simulation / 'third_party/NavDP'),
             'navigation_checkpoint': str(simulation / 'models/xnavdp/x-navdp_posttrain.ckpt'),
             'torch_site_packages': str(torch_packages)}
    for name, path in paths.items():
        if not Path(path).exists(): raise ValueError('Missing resource: ' + name)
    files = {str(p.relative_to(destination)): digest(p) for p in destination.rglob('*')
             if p.is_file() and p.name != 'manifest.json'}
    manifest = {'schema_version': 1, 'embodiment': 'unitree_go2', 'navigation': 'xnavdp', 'locomotion': 'moects',
                'host_paths': paths, 'files': files,
                'mesh_conversion': {'packages':packages,'backend':'trimesh Collada-to-PLY geometry export','files':conversions},
                'external_weights': {'xnavdp': {'path': paths['navigation_checkpoint'], 'sha256': digest(Path(paths['navigation_checkpoint']))}},
                'sandbox': {'go2_urdf': '/opt/resources/go2/robot/urdf/go2.urdf',
                            'moects_policy': '/opt/resources/go2/moects.pt', 'moects_config': '/opt/resources/go2/moects.yaml',
                            'torch_site_packages': '/opt/go2-torch'},
                'renderer': {'root': str(renderer), 'config_sha256': digest(renderer / 'config.json')},
                'scope': 'Existing local assets and pretrained policies reused; no Genesis or MuJoCo runtime'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1] / 'data/resources/go2')
    p.add_argument('--simulation', type=Path, required=True,
                   help='Local Go2 and X-NavDP dependency tree described in the README')
    p.add_argument('--renderer', type=Path, required=True,
                   help='Compatible AgentGarten HTTP service directory containing config.json')
    args = p.parse_args()
    result = prepare(args.destination.resolve(), args.simulation.resolve(), args.renderer.resolve())
    print(json.dumps({'status': 'prepared', 'files': len(result['files']), 'manifest': str(args.destination / 'manifest.json')}))
