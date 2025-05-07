#!/usr/bin/env python3
"""
NDI Manager Module - Centralizes NDI initialization for YOLOflow

This module is imported by the main YOLOflow application and initializes
a shared NDI connection that both the Capture and Test tabs can use.
"""

import os
import sys
import time
import yaml
import argparse
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - NDI Manager - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Try to import cyndilib for NDI support
try:
    import cyndilib as ndi
    import numpy as np
    import cv2
    NDI_AVAILABLE = True
    NDI_LIB_AVAILABLE = True
    logging.info("cyndilib module found - NDI access enabled")
except ImportError:
    NDI_AVAILABLE = False
    NDI_LIB_AVAILABLE = False
    logging.error("cyndilib not found - cannot use NDI")

# Global NDI state
ndi_finder = None
ndi_receiver = None
ndi_video_frame = None
ndi_lock = threading.Lock()
ndi_running = False
ndi_thread = None
ndi_frame = None
ndi_ret = False
ndi_connected = False

# Debug flag - set to False to reduce logging overhead and improve performance
DEBUG = False

def debug_print(message):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print(f"[NDI_DEBUG] {message}")

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        return {}
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

def save_config(config):
    """Save configuration to YAML file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving config: {str(e)}")
        return False

def list_ndi_sources():
    """List all available NDI sources"""
    if not NDI_LIB_AVAILABLE:
        logging.error("cyndilib not available - cannot list NDI sources")
        return []
        
    logging.info("Searching for NDI sources...")
    all_sources = []
    
    try:
        # Create a finder
        finder = ndi.Finder()
        finder.open()
        
        # Multiple scan attempts to maximize discovery
        # Scan 1: Quick initial scan
        finder.wait_for_sources(timeout=0.5)
        sources = finder.get_source_names()
        all_sources.extend(sources)
        logging.info(f"Initial scan found {len(sources)} sources")
        
        # Scan 2: Multiple consecutive scans
        for i in range(3):
            finder.wait_for_sources(timeout=0.5)
            sources = finder.get_source_names()
            new_sources = [s for s in sources if s not in all_sources]
            if new_sources:
                logging.info(f"Additional scan found {len(new_sources)} new sources")
                all_sources.extend(new_sources)
                
        # Display detailed source information
        if all_sources:
            logging.info(f"Found {len(all_sources)} total NDI sources:")
            for i, name in enumerate(all_sources):
                source = finder.get_source(name)
                if source:
                    host_name = getattr(source, 'host_name', 'Unknown')
                    stream_name = getattr(source, 'stream_name', 'Unknown')
                    logging.info(f"  {i+1}. {name} (Host: {host_name}, Stream: {stream_name})")
        else:
            logging.warning("No NDI sources found")
            
        # Clean up
        finder.close()
        return all_sources
    except Exception as e:
        logging.error(f"Error listing NDI sources: {str(e)}")
        if 'finder' in locals() and finder:
            try:
                finder.close()
            except:
                pass
        return []

def init_ndi(source_name=None, width=1280, height=720):
    """Initialize NDI with the specified source name"""
    global ndi_finder, ndi_receiver, ndi_video_frame, ndi_running, ndi_thread, ndi_connected
    
    if not NDI_LIB_AVAILABLE:
        logging.error("cyndilib not available - cannot initialize NDI")
        return False
    
    with ndi_lock:
        # Clean up any existing resources
        cleanup_ndi()
        
        # Default to config value if not specified
        if source_name is None:
            config = load_config()
            source_name = config.get('ndi', {}).get('source_name', 'TD-OUT')
            
        logging.info(f"Initializing NDI with source: {source_name}")
        debug_print(f"Init NDI with source: {source_name}")
        
        try:
            # Find NDI sources
            ndi_finder = ndi.Finder()
            ndi_finder.open()
            
            # Scan for sources with multiple attempts
            ndi_finder.wait_for_sources(timeout=1)
            source_names = ndi_finder.get_source_names()
            all_sources = list(source_names)
            
            # Do additional quick scans to find more sources
            for _ in range(2):
                ndi_finder.wait_for_sources(timeout=0.5)
                more_sources = ndi_finder.get_source_names()
                new_sources = [s for s in more_sources if s not in all_sources]
                if new_sources:
                    all_sources.extend(new_sources)
            
            # Use all discovered sources
            source_names = all_sources
            
            if not source_names:
                logging.warning("No NDI sources found")
                return False
                
            # Find target source with intelligent matching
            target_name = None
            target_source = None
            matching_sources = []
            partial_matches = []
            
            # Collect exact and partial matches
            for name in source_names:
                source = ndi_finder.get_source(name)
                if source:
                    stream_name = getattr(source, 'stream_name', '')
                    if stream_name and source_name.lower() == stream_name.lower():
                        matching_sources.append((name, source))
                    elif source_name.lower() in name.lower():
                        partial_matches.append((name, source))
            
            # Determine which source to use
            if matching_sources:
                # Use the first exact stream name match
                target_name, target_source = matching_sources[0]
                logging.info(f"Found exact match for NDI source with stream name: {target_name}")
            elif partial_matches:
                # Use the first partial match if no exact match
                target_name, target_source = partial_matches[0]
                logging.info(f"Found partial match for NDI source: {target_name}")
            else:
                # Fall back to first available source
                source = ndi_finder.get_source(source_names[0])
                if source:
                    target_name, target_source = source_names[0], source
                    logging.warning(f"Target '{source_name}' not found. Using first available source: {target_name}")
            
            # Create and set up receiver if we found a target
            if target_source:
                from cyndilib.video_frame import VideoFrameSync
                from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
                
                # QUALITY IMPROVEMENT: Use highest quality settings for NDI reception
                try:
                    # Try to use explicit video quality receiver settings
                    from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth, RecvFlags
                    
                    # Create receiver with maximum quality settings
                    ndi_receiver = ndi.Receiver(
                        color_format=RecvColorFormat.RGBX_RGBA,  # Full RGBA for best quality
                        bandwidth=RecvBandwidth.highest,         # Maximum bandwidth for quality
                        allow_video_fields=False                 # Progressive frames only
                    )
                    logging.info("NDI receiver created with high-quality settings")
                except (ImportError, AttributeError):
                    # Fallback for older cyndilib versions
                    ndi_receiver = ndi.Receiver(
                        color_format=RecvColorFormat.RGBX_RGBA, 
                        bandwidth=RecvBandwidth.highest
                    )
                    logging.info("NDI receiver created with standard quality settings")
                
                # Set up frame sync
                ndi_video_frame = VideoFrameSync()
                ndi_receiver.frame_sync.set_video_frame(ndi_video_frame)
                
                # Connect to the source
                ndi_receiver.set_source(target_source)
                
                # Create default initial frame
                global ndi_frame, ndi_ret
                ndi_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(ndi_frame, f"Connecting to NDI: {target_name}", 
                           (width//2 - 200, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ndi_ret = True
                
                # Start the frame receiving thread
                ndi_running = True
                ndi_connected = True
                ndi_thread = threading.Thread(target=ndi_update_thread, daemon=True)
                ndi_thread.start()
                
                logging.info(f"NDI initialized successfully with source: {target_name}")
                debug_print(f"NDI initialized with source: {target_name}")
                return True
            else:
                logging.error("Failed to find a suitable NDI source")
                return False
                
        except Exception as e:
            logging.error(f"Error initializing NDI: {str(e)}")
            cleanup_ndi()
            return False

def ndi_update_thread():
    """Background thread to continuously receive frames from NDI"""
    global ndi_frame, ndi_ret, ndi_connected
    
    if not ndi_running or ndi_receiver is None:
        return
        
    logging.info("NDI receiver thread started")
    debug_print("NDI receiver thread started")
    
    try:
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        
        # Main update loop
        while ndi_running:
            try:
                # Capture a video frame
                ndi_receiver.frame_sync.capture_video()
                
                # Check if we got data
                if ndi_video_frame.get_data_size() > 0:
                    # Get the frame data as NumPy array
                    frame_array = ndi_video_frame.get_array()
                    
                    # Get frame dimensions
                    width, height = ndi_video_frame.xres, ndi_video_frame.yres
                    
                    # Reshape flat data to 2D image (RGBA)
                    frame_array = frame_array.reshape(height, width, 4)
                    
                    # QUALITY IMPROVEMENT: Store RGBA directly without conversion to avoid quality loss
                    # Only convert when actually needed in read_frame() method
                    
                    # CRITICAL FIX: Convert to BGR immediately to avoid RGBA/BGR confusion
                    # Convert the frame to BGR (3 channels) format consistently
                    # This ensures all consumers get BGR format which is what OpenCV expects
                    bgr_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
                    
                    # Update the frame with thread safety - store BGR data
                    with ndi_lock:
                        ndi_frame = bgr_frame  # Store BGR format consistently
                        ndi_ret = True
                        ndi_frame_is_rgba = False  # Mark that we're storing BGR format
                        
                        # Mark as connected if not already
                        if not ndi_connected:
                            ndi_connected = True
                            logging.info("Connected to NDI source")
                            debug_print("Connected to NDI source")
                    
                    # CRITICAL FIX: Only count NEW frames by detecting changes
                    # This is the key issue - we were counting loop iterations instead of actual new frames
                    current_time = time.time()
                    
                    # Sampling a small region of the frame to create a signature
                    # Much faster than hashing the entire frame
                    # Take a small sample from the middle of the frame for comparison
                    h, w = frame_array.shape[:2]
                    y_mid, x_mid = h // 2, w // 2
                    
                    # Create a simple signature from a few pixels in the middle
                    # 10x10 grid of pixels from the center of the frame
                    sample_region = frame_array[y_mid-5:y_mid+5, x_mid-5:x_mid+5, 0].flatten()
                    
                    # Store variables on the function object to persist between calls
                    if not hasattr(ndi_update_thread, 'last_sample'):
                        # First frame - initialize
                        ndi_update_thread.last_sample = sample_region
                        ndi_update_thread.last_frame_time = current_time
                        frame_count = 1
                    else:
                        # Check if the sample has changed - only count if different
                        if not np.array_equal(sample_region, ndi_update_thread.last_sample):
                            # This is a genuinely new frame - count it
                            frame_count += 1
                            
                            # Calculate instantaneous FPS between successive frames
                            frame_delta = current_time - ndi_update_thread.last_frame_time
                            if frame_delta > 0:  # Avoid division by zero
                                instant_fps = 1.0 / frame_delta
                                if not hasattr(ndi_update_thread, 'source_fps'):
                                    ndi_update_thread.source_fps = instant_fps
                                else:
                                    # Smooth the FPS estimate with exponential moving average
                                    alpha = 0.1  # Smoothing factor
                                    ndi_update_thread.source_fps = (alpha * instant_fps + 
                                                               (1.0 - alpha) * ndi_update_thread.source_fps)
                            
                            # Update the last frame info
                            ndi_update_thread.last_sample = sample_region
                            ndi_update_thread.last_frame_time = current_time
                    
                    # Report FPS only periodically
                    if current_time - start_time >= 5.0:  # Log every 5 seconds
                        actual_fps = frame_count / (current_time - start_time)
                        source_fps_msg = ""
                        if hasattr(ndi_update_thread, 'source_fps'):
                            source_fps_msg = f" (source estimated at {ndi_update_thread.source_fps:.1f} FPS)"
                        
                        logging.info(f"NDI actual frames received: {actual_fps:.1f} FPS{source_fps_msg}")
                        print(f"NDI actual frames received: {actual_fps:.1f} FPS{source_fps_msg}")
                        frame_count = 0
                        start_time = current_time
            except Exception as e:
                logging.error(f"Error in NDI update loop: {str(e)}")
                time.sleep(0.5)  # Slow down on errors
            
            # ADAPTIVE LOOP TIMING: Based on detected source frame rate
            # Use smart timing that adapts to the actual source frame rate
            
            if hasattr(ndi_update_thread, 'source_fps') and ndi_update_thread.source_fps > 0:
                # We've detected the actual frame rate - use it to calculate optimal delay
                # Use 1/3 of the frame interval for responsive capture while reducing CPU usage
                target_delay = (1.0 / ndi_update_thread.source_fps) / 3.0
                
                # Cap the delay to reasonable bounds
                target_delay = min(max(0.001, target_delay), 0.016)  # Between 1ms and 16ms
                
                # Apply the calculated delay
                time.sleep(target_delay)
            else:
                # Default delay before we have detected the source frame rate
                # This is a compromise that works for most common frame rates (30-120 FPS)
                time.sleep(0.005)  # 5ms default delay
            
    except Exception as e:
        logging.error(f"Critical error in NDI thread: {str(e)}")
        with ndi_lock:
            ndi_connected = False

def read_frame():
    """Read the latest NDI frame (thread-safe) with high quality preservation"""
    global ndi_frame, ndi_ret
    
    # PERFORMANCE DEBUG: Keep track of how often this is called
    if not hasattr(read_frame, 'call_count'):
        read_frame.call_count = 0
        read_frame.last_report_time = time.time()
    
    read_frame.call_count += 1
    current_time = time.time()
    
    # Log every 100 frames or every 5 seconds, whichever is sooner
    if read_frame.call_count % 100 == 0 or (current_time - read_frame.last_report_time) > 5:
        elapsed = current_time - read_frame.last_report_time
        if elapsed > 0:
            calls_per_second = read_frame.call_count / elapsed
            print(f"[PERF_DEBUG] read_frame() called {read_frame.call_count} times "
                 f"({calls_per_second:.1f} calls/sec)")
        read_frame.call_count = 0
        read_frame.last_report_time = current_time
    
    with ndi_lock:
        if ndi_frame is None:
            return False, None
            
        # CRITICAL FIX: We now store BGR data directly, so just return it
        # Use simplified format check for safety
        if ndi_ret and ndi_frame is not None:
            # Check if we're still storing RGBA data (transition period)
            if hasattr(globals(), 'ndi_frame_is_rgba') and globals()['ndi_frame_is_rgba'] and len(ndi_frame.shape) == 3 and ndi_frame.shape[2] == 4:
                # We have RGBA data - convert to BGR
                # MEMORY OPTIMIZATION: Reuse conversion buffer to avoid allocations
                if not hasattr(read_frame, 'bgr_buffer') or read_frame.bgr_buffer.shape[:2] != ndi_frame.shape[:2]:
                    # Create or resize the buffer if needed
                    read_frame.bgr_buffer = np.empty((ndi_frame.shape[0], ndi_frame.shape[1], 3), 
                                                   dtype=np.uint8)
                
                # Convert to BGR using pre-allocated buffer
                cv2.cvtColor(ndi_frame, cv2.COLOR_RGBA2BGR, dst=read_frame.bgr_buffer)
                return True, read_frame.bgr_buffer
            else:
                # Already BGR format, return directly
                # No need to copy - we control the buffer's lifetime
                return True, ndi_frame
        
        # Return a copy of the frame to avoid threading issues
        # We need a copy here because we don't control the lifespan of the returned frame
        return ndi_ret, ndi_frame.copy() if ndi_ret else None

def is_connected():
    """Check if NDI is connected"""
    return ndi_connected

def cleanup_ndi():
    """Clean up all NDI resources"""
    global ndi_finder, ndi_receiver, ndi_video_frame, ndi_running, ndi_thread, ndi_connected
    
    logging.info("Cleaning up NDI resources")
    
    # Stop the update thread
    ndi_running = False
    if ndi_thread and ndi_thread.is_alive():
        try:
            ndi_thread.join(timeout=1.0)
            logging.info("NDI thread stopped")
        except:
            pass
    
    # Release receiver resources
    if ndi_receiver is not None:
        try:
            ndi_receiver.disconnect()
            ndi_receiver = None
            logging.info("NDI receiver released")
        except Exception as e:
            logging.error(f"Error releasing NDI receiver: {e}")
    
    # Close the finder
    if ndi_finder is not None:
        try:
            ndi_finder.close()
            ndi_finder = None
            logging.info("NDI finder closed")
        except Exception as e:
            logging.error(f"Error closing NDI finder: {e}")
    
    ndi_connected = False
    logging.info("All NDI resources released")
    return True


# Functions for direct use by the main YOLOflow application

def initialize_from_config(config):
    """Initialize NDI from config - this is called from main.py"""
    if not config:
        config = load_config()
    
    # Get NDI settings from config
    ndi_settings = config.get('ndi', {})
    source_name = ndi_settings.get('source_name', 'TD-OUT')
    width = ndi_settings.get('width', 1280)
    height = ndi_settings.get('height', 720)
    
    # Initialize NDI
    return init_ndi(source_name, width, height)

def get_capture_instance():
    """Get a capture instance compatible with the tabs' expected interface"""
    # Import here to avoid circular imports
    debug_print("Creating new SharedNDICapture instance")
    
    # CRITICAL PERFORMANCE FIX: Check if we already have instances
    # Store as a global static "singleton" variable to ensure reuse
    if not hasattr(get_capture_instance, '_instances'):
        get_capture_instance._instances = {}
        print("[PERF_DEBUG] First call to get_capture_instance - initializing instances cache")
    else:
        print(f"[PERF_DEBUG] Existing instance cache has {len(get_capture_instance._instances)} entries")
    
    # Create a wrapper that uses our shared NDI connection
    class SharedNDICapture:
        def __init__(self, sender_name=None, width=1280, height=720):
            debug_print(f"Initializing SharedNDICapture for {sender_name}")
            self.lock = threading.Lock()
            self.connected = is_connected()
            self.width = width
            self.height = height
            self.sender_name = sender_name
            self.ret = True
            
            # Create an initial frame
            self.frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(self.frame, "Using shared NDI connection", 
                       (width//2 - 200, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Thread not needed - we'll use the global one
            self.thread = None
            self.running = True
            self.cap = None  # Compatibility with some code that checks for .cap
            
            # CRITICAL PERFORMANCE DEBUG: Print a unique ID for each instance
            # This helps identify if multiple instances are being created unnecessarily
            print(f"[PERF_DEBUG] Created SharedNDICapture wrapper ID={id(self)} for {sender_name}")
        
        def read(self):
            """Read from the shared NDI connection"""
            debug_print("SharedNDICapture.read() called")
            result = read_frame()
            debug_print(f"Read frame result: {result[0]}")
            return result
        
        def isOpened(self):
            """Check if connected to NDI"""
            return is_connected()
        
        def release(self):
            """Release resources - note this doesn't stop the global connection"""
            self.running = False
            print("SharedNDICapture wrapper released")
    
    # Get settings from config
    config = load_config()
    ndi_settings = config.get('ndi', {})
    source_name = ndi_settings.get('source_name', 'TD-OUT')
    width = ndi_settings.get('width', 1280)
    height = ndi_settings.get('height', 720)
    
    # Create a cache key based on the parameters
    cache_key = f"{source_name}_{width}_{height}"
    
    # Check if we already have an instance for these settings
    if cache_key in get_capture_instance._instances:
        print(f"[PERF_DEBUG] Reusing existing NDI capture instance for {source_name}")
        # Return cached instance
        return get_capture_instance._instances[cache_key]
    
    # Create new instance and store in cache
    instance = SharedNDICapture(source_name, width, height)
    get_capture_instance._instances[cache_key] = instance
    print(f"[PERF_DEBUG] Created and cached new NDI capture instance for {source_name}")
    
    # Return the new instance
    return instance


# Script entrypoint for standalone operation
def main():
    """Main function when script is run standalone"""
    parser = argparse.ArgumentParser(description='NDI Manager Script')
    parser.add_argument('--source', type=str, help='NDI source name (default from config.yaml or TD-OUT)')
    parser.add_argument('--width', type=int, default=1280, help='Width of NDI output (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Height of NDI output (default: 720)')
    parser.add_argument('--list', action='store_true', help='List available NDI sources and exit')
    args = parser.parse_args()
    
    if args.list:
        print("Listing available NDI sources:")
        sources = list_ndi_sources()
        print(f"Found {len(sources)} NDI sources")
        return
    
    # Load config and override with command line args if provided
    config = load_config()
    
    # Get source name from args or config
    source_name = args.source
    if source_name is None:
        source_name = config.get('ndi', {}).get('source_name', 'TD-OUT')
    
    # Get dimensions from args or config
    width = args.width if args.width else config.get('ndi', {}).get('width', 1280)
    height = args.height if args.height else config.get('ndi', {}).get('height', 720)
    
    # Initialize NDI
    success = init_ndi(source_name, width, height)
    
    if success:
        print(f"NDI initialized successfully with source: {source_name}")
        print("Running continuously to maintain NDI connection.")
        print("Press Ctrl+C to exit.")
        
        try:
            # Keep running until interrupted
            while True:
                ret, frame = read_frame()
                if ret and frame is not None:
                    # Display info every second
                    time.sleep(1)
                    h, w = frame.shape[:2]
                    print(f"Receiving NDI: {w}x{h} ({time.strftime('%H:%M:%S')})")
                else:
                    print("Waiting for NDI frames...")
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting NDI Manager")
        finally:
            cleanup_ndi()
    else:
        print("Failed to initialize NDI connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()