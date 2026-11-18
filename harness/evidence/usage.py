"""Typed execution events retain the existing on-disk JSON field names."""
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation


@dataclass
class ToolEvent:
    id: str
    call_id: str
    name: str
    arguments: str
    created_at: str
    stage: str
    api_call_id: str | None
    status: str = 'running'
    result: dict | None = None
    finished_at: str | None = None
    feedback: dict | None = None

    def to_dict(self): return asdict(self)


def call_usage(call):
    """One receipt: cache reads partition input, charge is provider-reported, never estimated."""
    usage = call.get('usage') if isinstance(call.get('usage'),dict) else {}
    gateway = call.get('llmgate_usage') if isinstance(call.get('llmgate_usage'),dict) else {}
    keys = ('input_tokens', 'output_tokens', 'total_tokens')
    valid = all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys) and usage.get('complete',True) is True
    if 'cache_split_complete' in usage:
        cached = usage.get('cached_input_tokens') if usage['cache_split_complete'] else None
    else:
        details = usage.get('input_tokens_details')
        cached = details.get('cached_tokens',gateway.get('cache_read_tokens')) if isinstance(details,dict) else gateway.get('cache_read_tokens')
    split = bool(valid and type(cached) is int and 0 <= cached <= usage['input_tokens'])
    if type(gateway.get('cache_read_tokens')) is int and gateway['cache_read_tokens']!=cached: split = False
    raw_charge = next((value for value in (gateway.get('charged_amount'),call.get('charged_amount'),
                                          usage.get('charged_amount')) if value is not None),None)
    try:
        amount = Decimal(str(raw_charge)) if type(raw_charge) in (str,int,float) else None
        if amount is not None and (not amount.is_finite() or amount < 0): amount = None
    except InvalidOperation: amount = None
    currency = gateway.get('currency',call.get('currency',usage.get('charge_currency')))
    return {'input_tokens':usage['input_tokens'] if valid else 0,
            'output_tokens':usage['output_tokens'] if valid else 0,
            'total_tokens':usage['input_tokens']+usage['output_tokens'] if valid else 0,
            'complete':bool(valid), 'api_calls':1,
            'unmeasured_calls':int(not valid),
            'context_rejected_calls':int(call.get('context_rejected_before_generation') is True),
            'cached_input_tokens':cached if split else 0,
            'uncached_input_tokens':usage['input_tokens']-cached if split else 0,
            'cache_split_complete':split, 'cache_split_calls':int(split),
            'charged_amount':format(amount,'f') if amount is not None else None,
            'charged_amount_complete':amount is not None, 'charged_amount_calls':int(amount is not None),
            'charge_currency':currency if isinstance(currency,str) and currency else None}


def combine_usage(usages):
    """Sum independent scopes; callers must de-duplicate task/call identities first."""
    rows = list(usages)
    counts = ('input_tokens','output_tokens','total_tokens','cached_input_tokens','uncached_input_tokens',
              'api_calls','cache_split_calls','charged_amount_calls','unmeasured_calls','context_rejected_calls')
    result = {key:sum(row.get(key,0) for row in rows) for key in counts}
    for key in ('complete','cache_split_complete','charged_amount_complete'):
        result[key] = all(row.get(key,False) for row in rows)
    charged = [row for row in rows if row.get('charged_amount') is not None]
    currencies = {row.get('charge_currency') for row in charged}
    result['charge_currency'] = next(iter(currencies)) if len(currencies)==1 else None
    result['charged_amount'] = (format(sum((Decimal(str(row['charged_amount'])) for row in charged),Decimal(0)),'f')
                                if len(currencies)<=1 and (charged or not rows) else None)
    if len(currencies)>1:
        result.update(charged_amount_complete=False,charged_amount_reason='Different charge currencies; amounts are not combined')
    return result


def summarize_usage(calls):
    calls = list(calls)
    measured = [(call.get('stage'),call_usage(call)) for call in calls]
    stages = {stage:combine_usage(row for name,row in measured if name==stage)
              for stage in ('agent','run_critic','task_critic')}
    return {**combine_usage(row for _,row in measured), 'stages':stages,
            'by_stage':{stage:row['total_tokens'] for stage,row in stages.items()}}


def charge_metric(usage):
    amount = usage.get('charged_amount')
    if amount is not None and usage.get('charged_amount_complete'):
        return {'status':'ok','value':amount,'currency':usage.get('charge_currency')}
    return {'status':'missing','value':None,'known_value':amount,'currency':usage.get('charge_currency'),
            'reason':usage.get('charged_amount_reason') or
                f"Provider charged_amount recorded for {usage.get('charged_amount_calls',0)} / {usage.get('api_calls',0)} calls"}


def compare_charges(baseline, candidate):
    before, after = charge_metric(baseline), charge_metric(candidate)
    comparable = before['status']==after['status']=='ok' and before['currency']==after['currency']
    return {'baseline':before,'candidate':after,
            'delta':format(Decimal(after['value'])-Decimal(before['value']),'f') if comparable else None}
