"""Allow ``python -m gigaseal.bin`` to dispatch into the unified CLI."""
from gigaseal.cli import main

raise SystemExit(main())
