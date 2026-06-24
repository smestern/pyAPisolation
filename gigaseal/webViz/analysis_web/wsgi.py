"""WSGI entry point: ``gunicorn gigaseal.webViz.analysis_web.wsgi:app``."""

from .app import create_app

app = create_app()
