"""Accept schema-3 worlds; bind native identities only after actual execution."""
import ast
import hashlib
from harness.evidence.archive import uid

import yaml
from harness.core.contracts import ToolError
from harness.tools.world_validation import ContractLoader, validate_world as validate_contract


def validate_world(text):
    if len(text) > 512_000:
        raise ValueError('World description is too large')
    value = yaml.load(text, Loader=ContractLoader)
    if not isinstance(value, dict):
        raise ValueError('World state must be a YAML mapping')
    if value.get('schema_version') != 3:
        raise ValueError('New scene submissions require schema_version: 3; schema 2 is historical only')
    problems = validate_contract(value)
    if problems:
        first = problems[0]
        raise ToolError('invalid_world', first.message, world_path=first.path,
                        signature={'code': 'invalid_world', 'world_path': first.path},
                        issues=[{'path': issue.path, 'message': issue.message} for issue in problems[:20]])
    if value.get('document_role') == 'design_fixture':
        raise ToolError('design_fixture', 'Design examples are not executable world submissions; '
                        'create a world_hypothesis with honest source and implementation claims.')
    return value


def validate_code(code, world):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                   [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        if any(name.split('.')[0] in {'genesis', 'isaaclab', 'isaacsim'} for name in imports):
            raise ValueError('Use the installed Newton/Warp backend; legacy engines are not supported')
    if not any((isinstance(node, ast.ClassDef) and node.name == 'Example') or
               (isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) == 'Example' for alias in node.names))
               for node in tree.body):
        raise ValueError('Scene must expose the native Newton Example(viewer, args) class')
    return {'syntax':'valid', 'mapped_objects':list(world['objects']),
            'runtime_bindings':world.get('runtime_bindings', {}),
            'semantic_alignment':'requires_native_bindings_and_critic',
            'limitations':'Syntax is not evidence of object identity or semantic agreement.'}


def _source(box, inline, path, name):
    if (inline is None) == (path is None): raise ValueError(f'Provide {name} or path, exactly one')
    return box.read_file(path) if path is not None else inline


def set_world_state(ctx, box, yaml=None, path=None):
    yaml = _source(box, yaml, path, 'yaml')
    value = ctx.services.world.validate(yaml)
    digest = hashlib.sha256(yaml.encode()).hexdigest()
    archive = ctx.directory / f'world_{uid("state")}.yaml'
    archive.write_text(yaml)
    box.write_file('world.yaml', yaml)
    (ctx.directory / 'world.yaml').write_text(yaml)
    changed = ctx.state.get('world_hash') != digest
    ctx.state.update(world=value, world_hash=digest, world_path=str(archive.relative_to(ctx.task.directory)),
                     **({'rendered':False,'render_dir':None,'render_hash':None} if changed else {}))
    return {'valid':True, 'world_sha256':digest, 'path':str(archive.relative_to(ctx.task.directory)),
            'render_current':bool(ctx.state.get('rendered')),
            'next_action':'Existing render remains current.' if ctx.state.get('rendered') else
                          'Submit scene code against this world before finishing.',
            'runtime_binding_requirements':value.get('runtime_bindings',{}),
            'semantic_ids':list(value.get('objects',{}))+list(value.get('phenomena',{}))+list(value.get('cameras',{})),
            'solver_ids':list(value.get('solvers',{})),
            'binding_scope':'Declared requirements only; native handles are verified when the scene is built.'}, None
