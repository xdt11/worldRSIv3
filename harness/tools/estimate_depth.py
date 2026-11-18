"""DA3 image-depth tool and its isolated model worker."""
from pathlib import Path
import sys

if __package__:
    from harness.tools._media import PATH_SCHEMA, OUTPUT_PROPERTY, validate_arguments, checked, output_directory, model_device, run_worker
else:
    from _media import PATH_SCHEMA, OUTPUT_PROPERTY, validate_arguments, checked, output_directory, model_device, run_worker

NAME = 'estimate_depth'
DESCRIPTION = (
    'Run DA3 on one image path or 1..8 ordered image paths, exactly one of path/paths. '
    'Returns predicted depth/camera arrays in an NPZ and preview paths; use view_image to inspect previews. '
    'Predicted scale is not measured geometry. CPU is supported; CUDA requires sandbox GPU access. '
    'Extract explicit video frames first.')
ARGUMENTS = {'type': 'object', 'properties': {
    'path': PATH_SCHEMA,
    'paths': {'type': 'array', 'items': PATH_SCHEMA, 'minItems': 1, 'maxItems': 8,
              'description': 'Ordered image paths; use either path or paths.'},
    'output_dir': OUTPUT_PROPERTY,
    'device': {'type': 'string', 'enum': ['cpu', 'cuda'], 'default': 'cpu'},
    'process_res': {'type': 'integer', 'minimum': 1, 'default': 392}},
    'oneOf': [{'required': ['path']}, {'required': ['paths']}], 'additionalProperties': False}
SCHEMA = {'type': 'function', 'name': NAME, 'description': DESCRIPTION, 'strict': False, 'parameters': ARGUMENTS}


def execute(ctx, box, path=None, paths=None, device='cpu', process_res=392, output_dir=None):
    arguments = {'device': device, 'process_res': process_res}
    if path is not None:
        arguments['path'] = path
    if paths is not None:
        arguments['paths'] = paths
    if output_dir is not None:
        arguments['output_dir'] = output_dir
    return box.run_media_tool(NAME, arguments, ARGUMENTS, interpreter='perception-python'), None


def worker(arguments):
    validate_arguments(NAME, arguments, ARGUMENTS, image_only=True)
    import numpy as np
    import torch
    from PIL import Image

    device = model_device(arguments.get('device', 'cpu'))
    resource = Path('/opt/resources/perception')
    sys.path.insert(0, str(resource / 'Depth-Anything-3/src'))
    from depth_anything_3.api import DepthAnything3

    sources = [str(checked(path)) for path in arguments.get('paths', [arguments.get('path')])]
    output = output_directory(arguments, NAME)
    model = DepthAnything3.from_pretrained(str(resource / 'models/DA3-SMALL')).to(device=device).eval()
    with torch.inference_mode():
        prediction = model.inference(sources, process_res=arguments.get('process_res', 392), infer_gs=False)
    data = {key: np.asarray(getattr(prediction, key)) for key in ('depth', 'intrinsics', 'extrinsics')
            if getattr(prediction, key, None) is not None}
    if not np.isfinite(data['depth']).all():
        raise ValueError('DA3 returned non-finite depth')
    path = output / 'depth.npz'
    np.savez_compressed(path, **data)
    previews = []
    for index, depth in enumerate(data['depth']):
        low, high = np.quantile(depth, [.02, .98])
        gray = (np.clip((depth-low) / max(float(high-low), 1e-8), 0, 1)*255).astype(np.uint8)
        preview = output / f'depth_{index:04d}.png'
        Image.fromarray(gray).save(preview)
        previews.append(str(preview))
    return {'path': str(path), 'previews': previews, 'device': device, 'shape': list(data['depth'].shape),
        'source_paths': sources, 'model': 'depth-anything/DA3-SMALL',
        'interpretation': 'Model-predicted geometry and camera estimates. Scale is not independently measured.'}


def main():
    run_worker(NAME, ARGUMENTS, worker, image_only=True)


if __name__ == '__main__':
    main()
