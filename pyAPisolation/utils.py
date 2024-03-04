import numpy as np


DEBUG = True
def debug_wrap(func):
    def wrapper(*args, **kwargs):
        if DEBUG:
            return func(*args, **kwargs)
        else:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return np.nan
    return wrapper

def arg_wrap(argparser):
    """
    Wraps the argparser to catch any exceptions and query the user for the input again
    """
    required = argparser._action_groups.pop()
    optional = argparser._action_groups.pop()
    argparser._action_groups.append(required)
    return None
    