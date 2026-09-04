"""Legacy entry point kept for backwards compatibility.

``python src/predict.py`` now runs the interactive CLI prompt from
:mod:`churn.cli`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from churn.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["predict", "--interactive"]))
