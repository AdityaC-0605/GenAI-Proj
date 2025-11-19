"""Utility modules for Cross-Lingual QA System."""

# Suppress urllib3 warnings early (before any imports)
import warnings
warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*', category=UserWarning)

from src.utils.gradient_accumulation import GradientAccumulator
from src.utils.mixed_precision import MixedPrecisionManager
from src.utils.device_scheduler import DeviceScheduler

__all__ = ['GradientAccumulator', 'MixedPrecisionManager', 'DeviceScheduler']
