"""Paired, dimensionless per-metric version performance. V0 is 100."""
import math


def summarize_versions(versions):
    if not versions:return []
    baseline={row['case_key']:row for row in versions[0]['cases']}
    keys=sorted({key for version in versions for row in version['cases'] for key in row['metric_results'] if key.startswith('4dcodebench/')})
    keys+=['task_tokens']
    result=[]
    for key in keys:
        lower=key in {'task_tokens','4dcodebench/depth_error'}
        entry={'metric':key,'direction':'lower' if lower else 'higher','versions':[]}
        for version in versions:
            by_case={r['case_key']:r for r in version['cases']};pairs=[];ratios=[]
            for case,before in baseline.items():
                after=by_case.get(case,{})
                a={'status':'ok','value':before.get('usage',{}).get('total_tokens')} if key=='task_tokens' else before.get('metric_results',{}).get(key,{})
                b={'status':'ok','value':after.get('usage',{}).get('total_tokens')} if key=='task_tokens' else after.get('metric_results',{}).get(key,{})
                reason=None;ratio=None
                if any(x.get('status')!='ok' or x.get('measurement_status','ok') not in {'ok','gated'} for x in (a,b)):
                    reason='not_applicable' if all(x.get('status') in {'not_applicable','excluded'} for x in (a,b)) else 'missing_or_measurement_error'
                elif any(type(x.get('value')) not in (int,float) or not math.isfinite(x['value']) or x['value']<0 for x in (a,b)):
                    reason='invalid_value'
                elif a['value']==b['value']==0:ratio=1.0
                elif (b['value'] if lower else a['value'])==0:reason='zero_denominator'
                else:ratio=a['value']/b['value'] if lower else b['value']/a['value']
                if ratio is not None:ratios.append(ratio)
                pairs.append({'case':case,'baseline':a.get('value'),'current':b.get('value'),'ratio':ratio,'reason':reason})
            complete=bool(baseline) and len(ratios)==len(baseline) and set(by_case)==set(baseline)
            score=(0.0 if 0 in ratios else 100*math.exp(sum(math.log(r) for r in ratios)/len(ratios))) if complete else None
            entry['versions'].append({'index':version['index'],'value':score,'valid':len(ratios),'required':len(baseline),
                                      'status':'ok' if complete else 'unavailable','pairs':pairs})
        result.append(entry)
    return result
