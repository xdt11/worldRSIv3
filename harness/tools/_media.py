"""Small shared media primitives; each public tool owns its schema and worker."""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

PATH_SCHEMA = {'type':'string','minLength':1}
OUTPUT_PROPERTY = {**PATH_SCHEMA, 'description':'Output directory under /work; omitted creates a unique directory.'}
VIDEO_SUFFIXES = {'.mp4','.mov','.mkv','.webm','.avi','.m4v'}


class MediaArgumentError(ValueError):
    def __init__(self,name,message,**details):
        super().__init__(message)
        self.diagnostic = {'code':'invalid_media_arguments','message':message,'tool':name,**details,
            'signature':{'code':'invalid_media_arguments','tool':name,
                         'fields':sorted(details.get('invalid_fields',{}))}}


def _shape_errors(value,schema,path):
    types={'string':lambda x:isinstance(x,str) and bool(x.strip()),
           'integer':lambda x:type(x) is int,
           'number':lambda x:type(x) in (int,float) and math.isfinite(x),
           'array':lambda x:isinstance(x,list)}
    if not types[schema['type']](value):
        return {path:'Expected '+schema['type']+(' with a nonempty value' if schema['type']=='string' else '')}
    if 'enum'in schema and value not in schema['enum']:
        return {path:'Allowed values: '+', '.join(schema['enum'])}
    if ('minimum'in schema and value<schema['minimum']) or ('maximum'in schema and value>schema['maximum']):
        return {path:f"Value must be in [{schema.get('minimum','-inf')}, {schema.get('maximum','inf')}]"}
    if schema['type']=='array':
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',len(value)):
            return {path:f"Expected {schema.get('minItems',0)}..{schema.get('maxItems','unlimited')} items"}
        return {key:reason for index,item in enumerate(value)
            for key,reason in _shape_errors(item,schema['items'],f'{path}[{index}]').items()}
    return {}


def validate_arguments(name,args,schema,*,image_only=False):
    schema=schema.get('parameters',schema)
    required=schema.get('required',[]);allowed=schema['properties']
    if not isinstance(args,dict):
        raise MediaArgumentError(name,'Arguments must be an object',required_arguments=required,allowed_arguments=list(allowed))
    errors={key:'Required argument is missing' for key in required if key not in args}
    errors.update({key:'Unknown argument' for key in args if key not in allowed})
    for key,value in args.items():
        if key in allowed:errors.update(_shape_errors(value,allowed[key],key))
    if 'oneOf'in schema:
        groups=[option['required'] for option in schema['oneOf']]
        if sum(all(field in args for field in group) for group in groups)!=1:
            errors['/'.join(field for group in groups for field in group)]='Provide exactly one of the declared alternatives'
    if image_only:
        paths=args.get('paths',[]) if isinstance(args.get('paths'),list) else [args.get('path')]
        if any(isinstance(path,str) and Path(path).suffix.lower() in VIDEO_SUFFIXES for path in paths):
            errors['path']='This tool accepts images; use extract_frames to decode explicit video frames first'
    if errors:
        raise MediaArgumentError(name,'; '.join(f'{key}: {reason}' for key,reason in errors.items()),
            required_arguments=required,allowed_arguments=list(allowed),invalid_fields=errors)
    return args


def checked(value,*,write=False):
    path=Path(value)
    if not path.is_absolute():path=Path('/work')/path
    path=path.resolve()
    if not (path.is_relative_to('/work') or (not write and path.is_relative_to('/input'))):
        raise ValueError('Media path must be under /work or /input; outputs must be under /work')
    return path


def output_directory(args,name):
    path=checked(args.get('output_dir',f'/work/{name}/{uuid.uuid4().hex}'),write=True)
    path.mkdir(parents=True,exist_ok=True)
    return path


def model_device(device):
    """Initialize reproducible model RNGs; a requested GPU never falls back silently."""
    import os
    import random
    import numpy as np
    import torch
    if 'WORLD_RSI_SEED'in os.environ:
        seed=int(os.environ['WORLD_RSI_SEED']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device not in {'cpu','cuda'}:raise ValueError('device must be cpu or cuda')
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; automatic CPU fallback is disabled')
    return device


def run_worker(name,schema,worker,*,image_only=False):
    """A single tool's CLI: arguments on stdin, result on stdout, library logs on stderr."""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            args=validate_arguments(name,json.load(sys.stdin),schema,image_only=image_only)
            result=worker(args)
    except MediaArgumentError as exc:
        print(json.dumps({'error':exc.diagnostic},ensure_ascii=False),file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            sys.stderr.write(exc.stderr if isinstance(exc.stderr,str) else exc.stderr.decode('utf-8',errors='replace'))
        raise
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
