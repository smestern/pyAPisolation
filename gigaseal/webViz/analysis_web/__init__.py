"""Interactive analysis web app — Flask equivalent of the desktop GUI.

Two deployment profiles controlled by the ``GIGASEAL_WEB_PROFILE`` env var:

* ``public`` — hosted demo. 5 files / 50 MB per session, anonymous, rate-limited.
* ``lab``    — self-hosted lab server. No upload caps, optional bearer auth,
               optional server-side file picker.

The Flask app, session manager, and job runner are all configured from a single
:class:`~gigaseal.webViz.analysis_web.config.WebConfig` dataclass.
"""

from .config import WebConfig, get_config
from .app import create_app

__all__ = ["WebConfig", "get_config", "create_app"]
