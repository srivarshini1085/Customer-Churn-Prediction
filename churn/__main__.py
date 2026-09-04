"""Enable ``python -m churn`` as an alias for ``python -m churn.cli``."""

import sys

from churn.cli import main

if __name__ == "__main__":
    sys.exit(main())
