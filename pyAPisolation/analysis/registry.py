"""
Module registry for discovering and retrieving analysis modules.

Usage
-----
::

    from pyAPisolation.analysis import register, get, list_modules

    # Register by class (auto-instantiated)
    register(MyAnalysis)

    # Or register an instance
    register(MyAnalysis(dv_cutoff=10.0))

    # Retrieve
    module = get("my_analysis")

    # List everything
    print(list_modules())
"""

import logging
from typing import Dict, List, Optional

from .base import AnalysisBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------
_registry: Dict[str, AnalysisBase] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(cls_or_instance, **kwargs) -> AnalysisBase:
    """
    Register an analysis module.

    Parameters
    ----------
    cls_or_instance : type | AnalysisBase
        Either an ``AnalysisBase`` **subclass** (will be instantiated) or
        an already-instantiated module.
    **kwargs
        Forwarded to the constructor when *cls_or_instance* is a class.

    Returns
    -------
    AnalysisBase
        The registered instance (useful when used as a decorator).

    Examples
    --------
    ::

        # As a function call
        register(SpikeAnalysis)

        # With overrides
        register(SpikeAnalysis, dv_cutoff=10.0)

        # As a decorator
        @register
        class MyAnalysis(AnalysisBase):
            ...
    """
    # If called with a class, instantiate it
    if isinstance(cls_or_instance, type):
        if not issubclass(cls_or_instance, AnalysisBase):
            raise TypeError(
                f"Expected an AnalysisBase subclass, got {cls_or_instance}"
            )
        instance = cls_or_instance(**kwargs)
    elif isinstance(cls_or_instance, AnalysisBase):
        instance = cls_or_instance
    else:
        raise TypeError(
            f"register() expects an AnalysisBase class or instance, "
            f"got {type(cls_or_instance)}"
        )

    name = instance.name
    if name in _registry:
        logger.info(f"Overwriting existing module '{name}'")
    _registry[name] = instance
    logger.debug(f"Registered analysis module: '{name}' ({instance.display_name})")

    # Return the class so register() works as a decorator
    return cls_or_instance


def get(name: str) -> Optional[AnalysisBase]:
    """
    Look up a registered module by name.

    Parameters
    ----------
    name : str
        The module name (e.g. ``"spike"``).

    Returns
    -------
    AnalysisBase or None
    """
    module = _registry.get(name)
    if module is None:
        logger.warning(f"No module registered with name '{name}'")
    return module


def list_modules() -> List[str]:
    """Return a list of all registered module names."""
    return list(_registry.keys())


def get_all() -> Dict[str, AnalysisBase]:
    """Return a copy of the full registry dict."""
    return dict(_registry)


def clear():
    """Remove all registered modules (mainly for testing)."""
    _registry.clear()
