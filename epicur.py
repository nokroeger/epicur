#!/usr/bin/env python3
"""Convenience wrapper to run epicur without pip install."""
import os
import sys

# Ensure the src/ directory is on the Python path
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "src"))

from epicur.main import main

sys.exit(main())
