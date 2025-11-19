"""Utility to suppress common warnings that don't affect functionality."""

import warnings
import os

# Suppress warnings before urllib3 is imported
warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*', category=UserWarning)

def suppress_urllib3_warnings():
    """
    Suppress urllib3 OpenSSL warnings on macOS.
    
    This warning appears because macOS Python uses LibreSSL instead of OpenSSL,
    but urllib3 v2 prefers OpenSSL. This is harmless and doesn't affect functionality.
    """
    # Suppress warnings before importing urllib3
    warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*', category=UserWarning)
    
    # Also set environment variable to suppress
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:urllib3'
    
    # If urllib3 is already imported, disable its warnings
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
    except (ImportError, AttributeError):
        pass

def setup_warnings():
    """Setup warning filters for common non-critical warnings."""
    suppress_urllib3_warnings()
    
    # Suppress other common warnings if needed
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='transformers')
    warnings.filterwarnings('ignore', message='.*legacy.*', category=UserWarning)

