"""Skill package content policy, independent of version adoption."""
import re
from pathlib import Path
import yaml

ALLOWED = re.compile(r"skills/[a-z0-9][a-z0-9-]{0,63}/(?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:md|py|sh|json|yaml|yml|txt|toml)\Z")


def validate_skills(files):
    """Skills are UTF-8 regular files. Helpers execute only in a task sandbox."""
    for name, value in files.items():
        if not isinstance(name, str) or not ALLOWED.fullmatch(name):
            raise ValueError(f"Evolution path not allowed: {name}; only skills/<name>/... may evolve")
        if not isinstance(value, str) or '\x00' in value or len(value.encode()) > 1024 * 1024:
            raise ValueError("Skill changes must be UTF-8 text up to 1 MiB per file")
    if any(str(parent) in files for name in files for parent in Path(name).parents):
        raise ValueError('A skill path cannot be both a file and a directory')
    for package in {name.split('/')[1] for name in files}:
        entry = f'skills/{package}/SKILL.md'
        content = files.get(entry, '')
        frontmatter = re.fullmatch(r'---\r?\n(.*?)\r?\n---(?:\r?\n|$)(.*)', content, re.S)
        if frontmatter is None:
            raise ValueError(f'{entry} requires YAML frontmatter with name and description')
        try:
            metadata = yaml.safe_load(frontmatter[1])
        except yaml.YAMLError as exc:
            raise ValueError(f'{entry}: invalid skill frontmatter') from exc
        if not isinstance(metadata, dict) or metadata.get('name') != package or not isinstance(metadata.get('description'), str) or not metadata['description'].strip():
            raise ValueError(f'{entry}: name must match its directory and description must be nonempty')
        # Skill descriptions are deliberately single-line index entries.
        # Reject valid-but-unsupported folded/multiline metadata explicitly.
        inline = dict(re.findall(r'(?m)^([A-Za-z_]+):[ \t]*(.*)$', frontmatter[1]))
        description = inline.get('description', '').strip()
        try:
            inline_description = yaml.safe_load('description: ' + description).get('description')
        except yaml.YAMLError:
            inline_description = None
        if (inline.get('name', '').strip() != package or not description
                or description.startswith(('|', '>')) or '\n' in metadata['description'].strip()
                or inline_description != metadata['description']):
            raise ValueError(f'{entry}: skill metadata requires an unquoted name and a one-line description')
        if not frontmatter[2].strip():
            raise ValueError(f'{entry}: skill body must not be empty')
    return files
