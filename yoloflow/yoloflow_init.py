"""
Package initialiser for yoloflow.
"""

# ----- 3-line alias so old imports `import tabs.*` still work -------------
import importlib, sys
sys.modules['tabs'] = importlib.import_module('yoloflow.tabs')
# --------------------------------------------------------------------------

from .main import main

def run() -> None:
    """Launch the YOLOflow GUI (same as `python -m yoloflow`)."""
    main()