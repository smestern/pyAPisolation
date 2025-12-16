import numpy as np
import pandas as pd
import sys
import functools
import inspect
import numpy as np

DEBUG = True

def debug_wrap(func, returns=None):
    """
    Decorator that wraps functions in try-except when DEBUG=False.
    When DEBUG=True, exceptions are raised normally for easier debugging.
    Automatically returns appropriate number of np.nan values based on function signature.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            # Debug mode: let exceptions propagate for full traceback
            return func(*args, **kwargs)
        else:
            # Production mode: catch exceptions and return np.nan
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Determine number of return values from type hints or docstring
                if returns is not None:
                    if isinstance(returns, int):
                        n_returns = returns
                    elif isinstance(returns, tuple) or isinstance(returns, list):
                        return tuple(np.nan for _ in returns)
                sig = inspect.signature(func)
                return_annotation = sig.return_annotation
                
                # Check if return type hint indicates tuple
                if hasattr(return_annotation, '__origin__'):
                    if return_annotation.__origin__ is tuple:
                        # Return tuple of nans matching the type hint
                        n_returns = len(return_annotation.__args__)
                        return tuple(np.nan for _ in range(n_returns))
                
                # Default: single np.nan
                print(f"Warning: {func.__name__} failed with {type(e).__name__}: {e}")
                return np.nan
    return wrapper


class arg_wrap(object):
    """
    Wraps the argparser to catch any exceptions and query the user for the input again
    """
    def __init__(self, argparser, cli_prompt=True, gui_prompt=False, **kwargs):
        #determine which args are needed
        self.argparser = argparser
        self.required_args = [action.dest for action in self.argparser._actions if action.required]
        self.optional_args = [action.dest for action in self.argparser._actions if not action.required]
        self.parse_args = None

        if cli_prompt and gui_prompt:
            raise ValueError("Both cli_prompt and gui_prompt cannot be True")
        elif cli_prompt:
            self.parse_args = self._prompt_cli
        elif gui_prompt:
            self.parse_args = self._prompt_gui

    def __call__(self):
        return self.parse_args()
    
    def _determine_missing_args(self):
        missing_args = []
        sys_input = sys.argv
        for arg in self.required_args:
            if arg not in sys_input:
                missing_args.append(arg)
        return missing_args

    def _prompt_cli(self):
        try: #try to parse the args
            args = self.argparser.parse_args()
        except:
            missing_args = self._determine_missing_args()
            args = self._query_args()
        
        return args
    
    def _query_args(self):
        args = {}
        for arg in self.required_args:
            args[arg] = input(f"Enter the value for {arg}: ")
        for arg in self.optional_args:
            args[arg] = input(f"Enter the value for {arg} (optional): ")
        return args

    def _prompt_gui(self):
        raise NotImplementedError("GUI not yet implemented")