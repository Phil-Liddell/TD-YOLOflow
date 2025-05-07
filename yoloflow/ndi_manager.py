"""
NDI Manager - Centralized management of NDI connections

This module provides a singleton pattern for managing a single NDI capture instance
that can be shared across multiple tabs. It now uses the ndi_manager_script.py
file for the actual implementation.
"""

import logging
import sys
import os

# Add the parent directory to the path so we can import ndi_manager_script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the full NDI manager implementation
try:
    from ndi_manager_script import (
        initialize_from_config, 
        get_capture_instance, 
        is_connected,
        cleanup_ndi,
        list_ndi_sources
    )
    
    # These functions are now defined in ndi_manager_script.py
    initialize = initialize_from_config
    get_last_error = lambda: "See logs for details"
    is_initialized = is_connected
    is_capture_available = is_connected
    cleanup = cleanup_ndi
    
    logging.info("NDI Manager loaded from ndi_manager_script.py")
except ImportError as e:
    logging.error(f"Error importing ndi_manager_script.py: {e}")
    
    # Fallback to legacy implementation
    import threading
    import time
    from .utils import is_ndi_available, NDICapture

    # Keep track of our global instance
    _ndi_capture_instance = None
    _ndi_initialized = False
    _ndi_lock = threading.Lock()  # Thread lock for accessing the NDI instance
    _config = {}  # Will be populated from main app
    
    # Error tracking
    _last_error = None
    
    def initialize(config):
        """Initialize the NDI manager with configuration from main app"""
        global _config, _ndi_initialized, _last_error
        
        with _ndi_lock:
            if _ndi_initialized:
                return True  # Already initialized
                
            _config = config
            
            try:
                # Check if NDI is available
                if not is_ndi_available():
                    _last_error = "NDI is not available on this system"
                    logging.error(_last_error)
                    return False
                    
                # Get configured NDI source name
                from .utils import get_ndi_source_name
                ndi_source_name = get_ndi_source_name()
                
                # Get configured resolution
                width = config.get('ndi', {}).get('width', 1280)
                height = config.get('ndi', {}).get('height', 720)
                
                # Log initialization attempt
                logging.info(f"NDI Manager: Initializing global NDI capture with source: {ndi_source_name}")
                logging.info(f"NDI Manager: Requested resolution: {width}x{height}")
                
                # Create the NDI capture instance
                global _ndi_capture_instance
                _ndi_capture_instance = NDICapture(
                    sender_name=ndi_source_name,
                    width=width,
                    height=height
                )
                
                # Wait a moment for connection to establish
                time.sleep(0.5)
                
                if _ndi_capture_instance is None:
                    _last_error = "Failed to create NDI capture instance"
                    logging.error(_last_error)
                    return False
                
                logging.info("NDI Manager: Successfully initialized global NDI capture instance")
                _ndi_initialized = True
                return True
                
            except Exception as e:
                _last_error = f"Error initializing NDI Manager: {str(e)}"
                logging.error(_last_error)
                import traceback
                traceback.print_exc()
                return False
        
    def get_capture_instance():
        """Get the global NDI capture instance, initializing if needed"""
        global _ndi_capture_instance, _ndi_initialized, _last_error
        
        # Simple, direct access to existing instance
        if _ndi_initialized and _ndi_capture_instance is not None:
            return _ndi_capture_instance
        
        # Not initialized yet - try to initialize if we have config
        if not _ndi_initialized and _config:
            if initialize(_config):
                return _ndi_capture_instance
        
        # If we get here, initialization failed or hasn't happened yet
        return None
    
    def get_last_error():
        """Return the last error encountered"""
        global _last_error
        return _last_error
    
    def is_initialized():
        """Check if the NDI manager is initialized"""
        global _ndi_initialized
        return _ndi_initialized
    
    def is_capture_available():
        """Check if a valid NDI capture instance is available"""
        global _ndi_capture_instance, _ndi_initialized
        return _ndi_initialized and _ndi_capture_instance is not None
    
    def cleanup():
        """Clean up NDI resources"""
        global _ndi_capture_instance, _ndi_initialized
        
        with _ndi_lock:
            if _ndi_capture_instance is not None:
                try:
                    logging.info("NDI Manager: Cleaning up NDI resources")
                    _ndi_capture_instance.release()
                    _ndi_capture_instance = None
                    _ndi_initialized = False
                    logging.info("NDI Manager: NDI resources released")
                    return True
                except Exception as e:
                    logging.error(f"NDI Manager: Error during cleanup: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return False
            return True  # Nothing to clean up