"""Allow running epicur as ``python -m epicur``."""

import sys

from .main import main

sys.exit(main())
