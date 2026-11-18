"""Read-only diagnosis of a reconstruction run."""
from .review import prepare as _prepare

def prepare(run,*,budget=None,before_call=None,emit=None):
    return _prepare(run,final=False,budget=budget,before_call=before_call,emit=emit)
