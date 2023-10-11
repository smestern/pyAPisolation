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