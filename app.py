"""Streamlit entry point. Run with ``streamlit run app.py``.

All logic lives in :mod:`churn.app_ui`; this file only exists because Streamlit
launches a script by path.
"""

from churn.app_ui import main

main()
