"""
Model services for deepfake detection
"""

from .mesonet import MesoNetModel
from .xception import XceptionModel

__all__ = ['MesoNetModel', 'XceptionModel']