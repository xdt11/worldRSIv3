"""Crop explicit image pixels and archive the resulting evidence."""
from __future__ import annotations

if __package__:
    from harness.tools._media import PATH_SCHEMA,OUTPUT_PROPERTY,checked,output_directory,run_worker,validate_arguments
else:
    from _media import PATH_SCHEMA,OUTPUT_PROPERTY,checked,output_directory,run_worker,validate_arguments

NAME='crop_image'
DESCRIPTION=('Crop one image to pixel coordinates box_xyxy=[left,top,right,bottom], excluding right/bottom edges. '
    'Returns actual cropped pixels and its saved path. Extract a video frame first.')
ARGUMENTS={'type':'object','properties':{'path':PATH_SCHEMA,'output_dir':OUTPUT_PROPERTY,
    'box_xyxy':{'type':'array','items':{'type':'integer','minimum':0},'minItems':4,'maxItems':4,
        'description':'Pixel coordinates [left, top, right, bottom].'}},
    'required':['path','box_xyxy'],'additionalProperties':False}
SCHEMA={'type':'function','name':NAME,'description':DESCRIPTION,'strict':False,'parameters':ARGUMENTS}


def execute(ctx,box,path,box_xyxy,output_dir=None):
    from harness.tools.evidence import view_image,provenance,store_provenance,visual_groups
    args={'path':path,'box_xyxy':box_xyxy}
    if output_dir is not None:args['output_dir']=output_dir
    result=box.run_media_tool(NAME,args,ARGUMENTS)
    target=result.get('result',{}).get('path')
    if result.get('exit_code',0) or result.get('error') or not target:return result,None
    inherited=provenance(ctx,path)
    derivation=[*inherited.get('derivation',[]),{'operation':'crop','parent_path':path,'region_xyxy':box_xyxy}]
    store_provenance(ctx,target,{**inherited,'parent_evidence_id':inherited.get('evidence_id'),
        'parent_path':path,'region_xyxy':box_xyxy,'kind':'reference_crop','derivation':derivation})
    receipt,visual=view_image(ctx,box,path=target)
    receipt.update(parent_path=path,region_xyxy=box_xyxy,kind='reference_crop')
    store_provenance(ctx,target,receipt)
    return {**result,'images':[receipt]},visual_groups([receipt],[p for p in visual if p['type']=='input_image'])


def worker(args):
    validate_arguments(NAME,args,ARGUMENTS,image_only=True)
    from PIL import Image
    source=checked(args['path']);output=output_directory(args,NAME);box=args['box_xyxy']
    with Image.open(source) as image:
        if not (0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height):
            raise ValueError('Crop box must satisfy 0 <= left < right <= width and 0 <= top < bottom <= height')
        path=output/'crop.png';image.crop(box).save(path)
    return {'path':str(path),'source':str(source),'box':box}


def main():run_worker(NAME,ARGUMENTS,worker,image_only=True)

if __name__=='__main__':main()
