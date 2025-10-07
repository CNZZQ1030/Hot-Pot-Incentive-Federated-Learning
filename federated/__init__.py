"""
federated/__init__.py
"""
from .client import FederatedClient
from .server import FederatedServer

__all__ = ['FederatedClient', 'FederatedServer']
