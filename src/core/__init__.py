"""
Core Module - Base Classes and Interfaces
==========================================
Provides base classes and interfaces for the SmartGrocy system.
"""

from src.core.base import BasePipeline, BaseModule, BaseConfig
from src.core.exceptions import SmartGrocyException, PipelineError, ValidationError

__all__ = [
    'BasePipeline',
    'BaseModule', 
    'BaseConfig',
    'SmartGrocyException',
    'PipelineError',
    'ValidationError',
]

