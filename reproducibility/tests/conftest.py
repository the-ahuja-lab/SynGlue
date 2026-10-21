import os, sys
# locate the bundle root (marker: RUN_STEPS.md) and add shared/ for _common.py
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
for _ in range(5):
    if os.path.exists(os.path.join(_root, "RUN_STEPS.md")):
        break
    _root = os.path.dirname(_root)
sys.path.insert(0, os.path.join(_root, "shared"))
