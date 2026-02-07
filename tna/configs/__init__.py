"""Configuration module"""

from .model_config import TNAConfig
from .atlas_config import get_atlas_config
from .path_config import PathConfig

__all__ = [
    'TNAConfig',
    'get_atlas_config',
    'PathConfig',
]
