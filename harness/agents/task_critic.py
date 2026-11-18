"""Read-only task RSI after its frozen quality/token evaluation."""
from .review import prepare as _prepare

def prepare(run,*,propose=True,budget=None,before_call=None,emit=None):
    return _prepare(run,final=True,propose=propose,budget=budget,before_call=before_call,emit=emit)
