"""Legacy entry point kept for backwards compatibility.

``python src/train.py`` still works; the implementation now lives in
:mod:`churn.train`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from churn.train import main  # noqa: E402

if __name__ == "__main__":
    main()
