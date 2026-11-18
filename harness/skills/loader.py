"""Index metadata from the task's frozen skills; bodies remain on-demand reads."""
from pathlib import Path
import yaml


def index(directory):
    root=Path(directory)
    rows=[]
    for path in sorted(root.glob('*/SKILL.md')):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Frozen skills cannot contain symlinks')
        text=path.read_text()
        if not text.startswith('---\n') or '\n---' not in text[4:]:
            raise ValueError('Skill metadata frontmatter required')
        metadata=yaml.safe_load(text.split('---',2)[1])
        if not isinstance(metadata,dict):
            raise ValueError('Skill frontmatter must be a mapping')
        if metadata.get('name')!=path.parent.name or not isinstance(metadata.get('description'),str):
            raise ValueError('Skill name/description metadata is invalid')
        rows.append(f"- {metadata['name']}: {metadata['description']} Read /skills/{path.parent.name}/SKILL.md")
    return '\n'.join(rows)
