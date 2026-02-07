"""
TNA (Transformer Network Analysis)
A hierarchical graph Transformer framework for psychiatric disorder classification
"""

__version__ = '1.0.0'
__author__ = 'TNA Team'

from .models.tna_model import TNA, DualAtlasTNA

__all__ = [
    'TNA',
    'DualAtlasTNA',
]
