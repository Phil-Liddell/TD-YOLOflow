"""
yoloflow package initialiser.
"""

# ------------------------------------------------------------------
#  Make old imports like  from tabs.capture_tab import CaptureTab
#  resolve to the packaged  yoloflow.tabs.capture_tab  module.
# ------------------------------------------------------------------
import importlib, sys
sys.modules['tabs'] = importlib.import_module('yoloflow.tabs')
# ------------------------------------------------------------------

from .main import main


def run() -> None:           # optional helper
    """Launch the YOLOflow GUI (same as `python -m yoloflow`)."""
    main()
