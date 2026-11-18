"""SAM3 text-prompt image segmentation and its isolated CUDA worker."""
from pathlib import Path
import sys

if __package__:
    from harness.tools._media import PATH_SCHEMA, OUTPUT_PROPERTY, validate_arguments, checked, output_directory, model_device, run_worker
else:
    from _media import PATH_SCHEMA, OUTPUT_PROPERTY, validate_arguments, checked, output_directory, model_device, run_worker

NAME = 'segment_image'
DESCRIPTION = (
    'Run SAM3 text-prompt segmentation on one image. Returns predicted mask paths, boxes and confidence scores; '
    'use view_image for selected masks. Empty detections are valid. Requires CUDA sandbox GPU access. '
    'Extract a video frame first; predictions do not establish 3D identity or measured geometry.')
ARGUMENTS = {'type': 'object', 'properties': {
    'path': PATH_SCHEMA, 'output_dir': OUTPUT_PROPERTY,
    'prompt': {'type': 'string', 'minLength': 1, 'description': 'Text describing the objects to segment.'},
    'device': {'type': 'string', 'enum': ['cuda'], 'default': 'cuda'},
    'confidence_threshold': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': .5}},
    'required': ['path', 'prompt'], 'additionalProperties': False}
SCHEMA = {'type': 'function', 'name': NAME, 'description': DESCRIPTION, 'strict': False, 'parameters': ARGUMENTS}


def execute(ctx, box, path, prompt, device='cuda', confidence_threshold=.5, output_dir=None):
    arguments = {'path': path, 'prompt': prompt, 'device': device, 'confidence_threshold': confidence_threshold}
    if output_dir is not None:
        arguments['output_dir'] = output_dir
    return box.run_media_tool(NAME, arguments, ARGUMENTS, interpreter='perception-python'), None


def worker(arguments):
    validate_arguments(NAME, arguments, ARGUMENTS, image_only=True)
    import numpy as np
    import torch
    from PIL import Image

    device = model_device(arguments.get('device', 'cuda'))
    resource = Path('/opt/resources/perception')
    sys.path.insert(0, str(resource / 'sam3'))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    source = checked(arguments['path'])
    prompt = arguments['prompt'].strip()
    output = output_directory(arguments, NAME)
    model = build_sam3_image_model(checkpoint_path=str(resource / 'models/sam3/sam3.pt'),
        device=device, eval_mode=True, load_from_HF=False)
    processor = Sam3Processor(model, device=device,
        confidence_threshold=arguments.get('confidence_threshold', .5))
    with Image.open(source) as original:
        image = original.convert('RGB')
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        result = processor.set_text_prompt(state=processor.set_image(image), prompt=prompt)
    masks = result['masks'].detach().cpu().numpy()
    boxes = result['boxes'].detach().float().cpu().numpy()
    scores = result['scores'].detach().float().cpu().numpy()
    instances = []
    for index, mask in enumerate(masks):
        path = output / f'mask_{index:04d}.png'
        Image.fromarray(np.squeeze(mask).astype(np.uint8)*255).save(path)
        instances.append({'path': str(path), 'box_xyxy': boxes[index].tolist(), 'score': float(scores[index])})
    path = output / 'segmentation.npz'
    np.savez_compressed(path, masks=masks, boxes=boxes, scores=scores)
    return {'instances': instances, 'device': device, 'prompt': prompt, 'source': str(source),
        'model': 'facebook/sam3',
        'interpretation': 'Text-prompt segmentation predictions, not 3D identity or measured geometry.'}


def main():
    run_worker(NAME, ARGUMENTS, worker, image_only=True)


if __name__ == '__main__':
    main()
