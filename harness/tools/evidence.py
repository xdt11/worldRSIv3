"""Expose actual archived image pixels as a project tool."""
import hashlib
import json
from harness.evidence.archive import inside,uid
from harness.evidence.context import image_item,presentation_metadata,public_source_id,evidence_label


def source_identity(ctx,path):
    record=getattr(ctx.task,'record',{})
    for index,source in enumerate(record.get('inputs',[])):
        if path==source.get('sandbox_path'):
            identity={'source_id':public_source_id(source,index),
                      'source_sha256':source.get('sha256'),'source_path':path,'kind':'public_input'}
            declared=getattr(ctx,'state',{}).get('world',{}).get('sources',{}).get(identity['source_id'],{})
            identity.update(camera_id=declared.get('camera'),source_clock=declared.get('clock'))
            return identity
    return {}


def frame_identity(ctx,path,record):
    identity=source_identity(ctx,path)
    source=getattr(ctx,'state',{}).get('world',{}).get('sources',{}).get(identity.get('source_id'),{})
    time=record.get('decoded_time_s')
    shots=source.get('shots',[])
    if time is not None:
        for index,shot in enumerate(shots):
            first,last=shot['interval_s']
            if first<=time<last or index==len(shots)-1 and time==last:
                identity.update(shot_id=shot['id'],camera_id=shot['camera']);break
    return identity


def provenance(ctx,path):
    index=ctx.task.directory/'work/evidence-index.json'
    entries=json.loads(index.read_text()) if index.is_file() else {}
    return {**source_identity(ctx,path),**entries.get(path,{})}


def store_provenance(ctx,path,record):
    index=ctx.task.directory/'work/evidence-index.json';index.parent.mkdir(parents=True,exist_ok=True)
    entries=json.loads(index.read_text()) if index.is_file() else {}
    entries[path]=record
    index.write_text(json.dumps(entries,ensure_ascii=False,indent=2))


def visual_groups(records,images):
    """Keep metadata adjacent to its actual pixels through native hoisting."""
    result=[]
    for record,pixels in zip(records,images):
        result.extend([{'type':'input_text','text':'EVIDENCE_GROUP '+json.dumps(evidence_label(record),ensure_ascii=False)},pixels])
    return result

def view_image(ctx, box, path=None, paths=None):
    if (path is None)==(paths is None): raise ValueError('Provide path or paths, exactly one')
    selected = [path] if path is not None else paths
    if not isinstance(selected,list) or not selected or any(not isinstance(p,str) for p in selected):
        raise ValueError('Provide one image path or a nonempty list of image paths')
    records, images = [], []
    comparison = getattr(ctx, 'state', {}).get('edit_comparison') or {}
    matched = {view[side]:{'kind':'edit_inspection','side':side,'time_s':view['time_s']}
               for view in comparison.get('views', []) for side in ('before','after')}
    for current in selected:
        raw = inside(ctx.task.directory,current).read_bytes() if current in matched else box.image(current)
        archive = ctx.directory / 'views' / (uid('view') + '.image')
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(raw)
        digest=hashlib.sha256(raw).hexdigest()
        record={**provenance(ctx,current),**matched.get(current,{}),'path':current,'sha256':digest,'evidence_id':'image_'+digest,
            'archive_path':str(archive.relative_to(ctx.task.directory)),
            'presentation':'EXIF-normalized RGB JPEG thumbnail, maximum 1280px; original retained',
            **presentation_metadata(raw)}
        records.append(record)
        images.append(image_item(raw))
    return (records[0],visual_groups(records,images)) if path is not None else ({'images':records},visual_groups(records,images))
