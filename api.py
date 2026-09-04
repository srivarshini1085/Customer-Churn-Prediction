"""ASGI entry point for the churn API.

Run with:  ``uvicorn api:app``  (or ``python -m churn serve``).
All logic lives in :mod:`churn.service`.
"""

from churn.service.app import create_app

app = create_app()
