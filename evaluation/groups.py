"""Official README families; explicitly local, equal-weight presentation scores."""
import math

FAMILIES = (
    ('perceptual','感知外观','Perceptual',('semantic_dinov3',)),
    ('dynamics_2d','二维动态','2D Dynamics',('flow_distribution','track2d_dtw','dynamic_iou')),
    ('geometry_25d','深度几何','2.5D Geometry',('depth_error',)),
    ('geometry_3d','三维几何','3D Geometry',('scene_3d',)),
    ('dynamics_3d','三维动态','3D Dynamics',('trajectory_dtw','emd_step')),
    ('additional','补充一致性','Additional',('semantic_tips','geophys_dinov3','uni3d_moge_scene','uni3d_point_scene','occupancy_dtw')),
    ('geometry_quality','几何质量','Geometry quality',('mesh_watertight','mesh_manifold','mesh_clean_faces','mesh_no_self_intersection','interpenetration')),
    ('vqa','物理问答','VQA',('vqa_balanced_accuracy','vqa_accuracy')),
)

def metric_groups(metrics):
    groups=[]
    for identity,label,family,keys in FAMILIES:
        rows=[{'key':'4dcodebench/'+key,**metrics.get('4dcodebench/'+key,{'status':'missing','value':None})} for key in keys]
        applicable=[row for row in rows if row['status'] not in {'not_applicable','excluded'}]
        valid=[row for row in applicable if row['status']=='ok' and type(row.get('value')) in (int,float) and math.isfinite(row['value'])]
        # VQA's native balanced accuracy is its total; accuracy remains a subscore.
        scored=valid[:1] if identity=='vqa' else valid
        complete=bool(applicable) and len(valid)==len(applicable)
        total=(sum(1-row['value']/2 if row['key'].endswith('/depth_error') else row['value'] for row in scored)/len(scored)*100
               if complete and scored else None)
        groups.append({'id':identity,'label':label,'family':family,'metrics':rows,'value':total,
            'status':'ok' if total is not None else 'partial' if valid else 'not_applicable' if not applicable else 'unavailable',
            'coverage':{'valid':len(valid),'applicable':len(applicable)},
            'aggregation':'native_balanced_accuracy' if identity=='vqa' else 'local_equal_weight_0_100',
            'note':'界面汇总，非官方总分；depth = 1 − error/2；缺失或错误不静默剔除。' if identity!='vqa' else '原生 balanced accuracy × 100'} )
    return groups
