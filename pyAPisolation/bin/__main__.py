"""Allow ``python -m pyAPisolation.bin`` to dispatch into the unified CLI."""
from pyAPisolation.cli import main

raise SystemExit(main())
