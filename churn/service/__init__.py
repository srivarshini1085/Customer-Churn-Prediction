"""FastAPI serving layer for the churn model."""

from churn.service.app import create_app

__all__ = ["create_app"]
