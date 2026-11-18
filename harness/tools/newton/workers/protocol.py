"""Shared output contract checks; no renderer, model or metric dependencies."""
import math


def edit_inspection(views=None):
    """A small set of world-space observations, including simultaneous viewpoints."""
    from .camera_math import normalize_camera_pose
    if views is None:
        views = [{'time_s': 0.0}]
    if not isinstance(views, list) or not 1 <= len(views) <= 12:
        raise ValueError('Choose 1..12 representative inspection views')
    samples = []
    for view in views:
        if not isinstance(view, dict) or set(view) - {'time_s', 'camera'}:
            raise ValueError('Inspection views contain time_s and an optional world-space camera')
        t = view.get('time_s')
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0 or (samples and t < samples[-1]['time_s']):
            raise ValueError('Inspection times must be finite, nonnegative and in simulation order')
        sample = {'time_s': t}
        if 'camera' in view:
            normalize_camera_pose(view['camera'])
            sample['camera'] = {key: list(view['camera'][key]) for key in ('position', 'target', 'up')}
            if 'fov_deg' in view['camera']:
                fov = view['camera']['fov_deg']
                if type(fov) not in (int, float) or not 1 < fov < 179:
                    raise ValueError('Inspection vertical fov_deg must be within (1,179)')
                sample['camera']['fov_deg'] = fov
        samples.append(sample)
    return {'schema_version': 1, 'kind': 'scene_edit', 'views': samples, 'width': 640, 'height': 360}


class OutputContractError(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.diagnostic = {'code':code, 'message':message, **details,
                           'signature':{'code':code, 'kind':details.get('kind')}}


def _timeline(values):
    return (isinstance(values,list) and bool(values)
            and all(type(t) in (int,float) and math.isfinite(t) and t >= 0 for t in values)
            and all(a < b for a,b in zip(values,values[1:])))


def validate_output(manifest, contract):
    """Validate output without rewriting timestamps, dropping or inventing frames.

    Image contracts require one frame. Video contracts require every published
    timestamp within their declared tolerance. Unconstrained outputs still need
    finite, nonnegative and strictly increasing timestamps.
    """
    kind = contract.get('kind') if isinstance(contract,dict) else None
    if kind == 'scene_edit':
        plan = edit_inspection(contract.get('views'))
        frames = manifest.get('frames', [])
        if (contract != plan or manifest.get('inspection') != plan or len(frames) != len(plan['views'])
                or any(not math.isclose(frame['timestamp_s'], view['time_s'], rel_tol=0, abs_tol=1e-6)
                       for frame, view in zip(frames, plan['views']))):
            raise OutputContractError('edit_inspection_mismatch', 'Native inspection differs from the submitted views', kind=kind)
        return {'kind':kind, 'frame_count':len(frames), 'valid':True}
    if kind not in ('image','video','unconstrained') or type(contract.get('schema_version',1)) is not int or contract.get('schema_version',1) != 1:
        raise OutputContractError('invalid_output_contract','Unsupported output contract schema or kind',kind=kind)
    expected = contract.get('timestamps_s')
    tolerance = contract.get('timestamp_tolerance_s')
    if kind == 'video' and (not _timeline(expected) or type(tolerance) not in (int,float)
                           or not math.isfinite(tolerance) or tolerance < 0):
        raise OutputContractError('invalid_output_contract',
            'Video output needs a fixed, finite, increasing timeline and nonnegative tolerance',kind=kind)
    frames = manifest.get('frames') if isinstance(manifest,dict) else None
    actual_count = len(frames) if isinstance(frames,list) else 0
    expected_count = len(expected) if kind == 'video' else 1 if kind == 'image' else None
    details = {'kind':kind,'expected_frame_count':expected_count,'actual_frame_count':actual_count,
               'timestamp_tolerance_s':tolerance if kind=='video' else None,
               'contract_path':'/work/output_contract.json'}
    if not isinstance(frames,list) or not frames or any(not isinstance(f,dict) for f in frames):
        raise OutputContractError('invalid_render_manifest','render_manifest.frames must contain frame objects',**details)
    if expected_count is not None and actual_count != expected_count:
        raise OutputContractError('output_frame_count',
            f'Output must match the complete fixed protocol: expected {expected_count} frames, received {actual_count}. '
            'Use every published output timestamp without dropping or adding frames.',**details)
    actual = [f.get('timestamp_s') for f in frames]
    if not _timeline(actual):
        raise OutputContractError('output_timestamps',
            'Render timestamps must be finite, nonnegative, unique and strictly increasing',**details)
    max_error = None
    if kind == 'video':
        errors = [abs(a-b) for a,b in zip(actual,expected)]
        max_error = max(errors)
        mismatches = [{'index':i,'expected_timestamp_s':e,'actual_timestamp_s':a,'error_s':error}
                      for i,(a,e,error) in enumerate(zip(actual,expected,errors)) if error > tolerance]
        if mismatches:
            raise OutputContractError('output_timestamps',
                f'Output timestamps must match the complete fixed protocol within {tolerance:g}s; '
                f'{len(mismatches)} frame timestamps exceed tolerance. Follow the published timestamps_s.',
                **details,first_mismatches=mismatches[:5],max_timestamp_error_s=max_error)
    return {'kind':kind,'frame_count':actual_count,'timestamp_tolerance_s':details['timestamp_tolerance_s'],
            'max_timestamp_error_s':max_error,'valid':True}
