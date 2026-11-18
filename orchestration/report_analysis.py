"""Descriptive campaign analysis; no mutations, model calls or adoption decisions."""
import math


SKILL_EXPLANATIONS = {
    'rigid-assembly-pose-validation': ('多部件刚体姿态验证',
        '把组件放在同一局部坐标系，对组件中心偏移和朝向施加同一个完整旋转。先验证倾斜姿态，再拟合轨迹，避免只旋转朝向却漏转部件位置。',
        '来源：第一轮磁体牵引重建中，多部件组装的 pitch 与位置变换不一致。它解决组装一致性，不证明物理机制正确。'),
    'static-landmark-calibration': ('先标定静态地标，再拟合运动',
        '用近远处静态角点联合拟合相机和几何比例，确认实际渲染的重投影误差后锁定标定；再拟合运动中心，最后单独调整外观。',
        '来源：第二轮软轮胎下台阶中，静态角点拟合改善了取景；中心轨迹吻合仍不能证明旋转、形变或接触正确。'),
    'event-anchored-motion-fitting': ('以可见事件约束运动相位',
        '记录释放、换面、支撑变化和到达的时间区间，用角点及正反面共同拟合平移与旋转，并在分段边界和实际返回时间验证。',
        '来源：第三轮多米诺场景中，猜测转数、延迟与插值会产生错误的中间相位。该方法补充刚体与相机验证，不恢复不可见的动力学。'),
}


def enrich_report(report, app, campaign, tasks):
    versions=report['versions']
    from evaluation.version_summary import summarize_versions
    report['version_performance']=summarize_versions(versions)
    for row in report['rounds']:
        snapshot=app.versions.snapshot(row['version']) if row.get('version') else {}
        parent=app.versions.snapshot(row['parent']) if row.get('parent') else {}
        skills=[]
        for path,content in snapshot.items():
            if not path.endswith('/SKILL.md') or parent.get(path)==content:continue
            name=path.split('/')[-2]
            title,explanation,origin=SKILL_EXPLANATIONS.get(name,(name,'请展开原文查看适用范围、执行方法与验证要求。','由该轮 task RSI critic 提议。'))
            skills.append({'path':path,'name':name,'title':title,'explanation':explanation,'origin':origin,'content':content})
        row.update(skills=skills,title=(row.get('case') or {}).get('case_key'),
                   component_counts={'skills':sum(path.endswith('/SKILL.md') for path in snapshot)})
    # The media catalog is a set of inputs; repeated test executions belong to
    # their version batches, rather than twelve duplicate dataset thumbnails.
    for group in report['datasets']:
        group['label']='进化集' if group['split']=='evolution' else '测试集'
        group['items']=list({item['case_key']:item for item in reversed(group['items'])}.values())[::-1]
    errors=[]
    for version in versions:
        for case in version['cases']:
            vqa=case.get('vqa') or {}
            if vqa.get('status') in {'error','partial'}:
                errors.append({'version':version['label'],'task_id':case['id'],'case_key':case['case_key'],
                               'error':vqa.get('error') or vqa['status']})
    report['measurement_errors']=errors
    report['metric_analysis']=[]
    if len(versions)>1:
        first,last=versions[0],versions[-1]
        for key in sorted({k for v in versions for k in v['metrics']}):
            values=[v['metrics'].get(key) for v in versions]
            if any(x is None or not math.isfinite(x) for x in (values[0],values[-1])):continue
            delta=values[-1]-values[0]
            report['metric_analysis'].append({'metric':key,'before':values[0],'after':values[-1],'delta':delta,
                'direction':'lower' if key.endswith('/depth_error') else 'higher',
                'coverage':[sum(k.get('metric_results',{}).get(key,{}).get('status')=='ok' for k in v['cases']) for v in versions]})
        def change(key,label):
            a,b=first['metrics'].get('4dcodebench/'+key),last['metrics'].get('4dcodebench/'+key)
            return f'{label} {a:.4f} → {b:.4f}（{b-a:+.4f}）' if a is not None and b is not None else ''
        report['findings']=[
            {'title':'整体表现有改善，也有退化', 'body':'；'.join(filter(None,[change('semantic_dinov3','DINOv3 语义相似度 ↑'),change('track2d_dtw','二维轨迹相似度 ↑'),change('depth_error','深度误差 ↓'),change('dynamic_iou','动态 IoU ↑')]))+'。这些是固定测试集的描述性结果，不能据此断言 RSI 的因果收益。'},
            {'title':'VQA 最新有效判题结果' if not errors else 'VQA 下降混有判题失败', 'body':change('vqa_balanced_accuracy','合并题库 balanced accuracy ↑')+(f'。仍有 {len(errors)} 个测试 task 判题异常，原始失败零分不能解释为质量退化。' if errors else '。已纳入成功补评，原超时记录与补评费用分别留存。')},
            {'title':'成本没有随轮次单调下降','body':'各版测试 task tokens：'+' → '.join(f"{v['usage']['total_tokens']:,}" for v in versions)+f"。{first['label']} 到 {last['label']} 变化 {(last['usage']['total_tokens']/first['usage']['total_tokens']-1)*100:+.1f}%；只含重建与 run critic，不含 task RSI critic 或 VQA 判题。"},
            {'title':'三个技能已进入本实验分支','body':f"{len(report['rounds'])} 轮均保留实际 skill diff、来源 task 与验证要求。自动采纳是调度策略；读过 skill 或生成了新 skill，不等于已证实使用正确、提升质量或节约 tokens。"},
        ]
    performance={row['metric']:row for row in report['version_performance']}
    if len(versions)>1:
        labels={'semantic_dinov3':'语义相似度','track2d_dtw':'二维轨迹','depth_error':'深度误差表现','dynamic_iou':'动态 IoU'}
        normalized=[]
        for key,label in labels.items():
            metric=performance.get('4dcodebench/'+key)
            score=metric['versions'][-1]['value'] if metric else None
            if score is not None:normalized.append(f'{label} {score:.1f}')
        report['findings'][0]['body']='V0 = 100，最终版各指标的逐 task 配对几何指数：'+'；'.join(normalized)+'。高于 100 表示该指标相对基线改善，低于 100 表示退化；不跨指标合成总分。'
        vqa=performance.get('4dcodebench/vqa_balanced_accuracy')
        if vqa:
            report['findings'][1]['body']='VQA balanced accuracy 配对指数（V0=100）：'+' → '.join(f"{v['value']:.1f}" if v['value'] is not None else '测量未完整' for v in vqa['versions'])+('。仍有判题异常，不能把失败零分解释为质量退化。' if errors else '。已纳入成功补评；原生题库合并分另在原始读数区展示。')
    report['caveats'] += ['本实验只有三个测试样本、每版本各一次 task；未设置随机 seed，没有重复实验或置信区间。',
        'VQA 复用官方题库、gates 与计分，但判题模型为 kimi-k3，并非官方发布的 Gemini 判题模型。',
        'V0 完成后切换了流水调度；接续保留新旧协议哈希，V0 与后续版本仅作描述性比较，不冒充严格协议一致的因果实验。']
    if report['timeline']:
        events=sorted([(r['start_s'],1) for r in report['timeline']]+[(r['end_s'],-1) for r in report['timeline']],key=lambda x:(x[0],x[1]))
        count=peak=0
        for _,delta in events:count+=delta;peak=max(peak,count)
        report['totals']['peak_concurrent']=peak
    return report
