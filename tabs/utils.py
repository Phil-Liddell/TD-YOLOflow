import os
import cv2
import numpy as np
import threading
import time
import logging
import warnings
from datetime import datetime

# NDI library support

# Try to import cyndilib for NDI support
try:
    import cv2
    import numpy as np
    
    # Try to import cyndilib
    try:
        import cyndilib as ndi
        NDI_AVAILABLE = True
        NDI_LIB_AVAILABLE = True
        print("cyndilib module found - can use direct NDI access")
    except ImportError:
        NDI_AVAILABLE = False
        NDI_LIB_AVAILABLE = False
        print("cyndilib not found - will use demo pattern instead")
except ImportError:
    NDI_AVAILABLE = False
    NDI_LIB_AVAILABLE = False
    pass

# Silence OpenCV warnings (compatible with all versions)
try:
    # Modern OpenCV versions have LOG_LEVEL constants
    if hasattr(cv2, 'LOG_LEVEL_SILENT'):
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    else:
        # Older versions - use numeric value (0)
        cv2.setLogLevel(0)
except Exception:
    # Silently fail if setLogLevel is not available
    pass

# Disable OpenCV depth camera errors for Intel RealSense
# This special string in environment variables silences the D4xx depth camera errors
os.environ["OPENCV_LOG_LEVEL"] = "0"  # Silence OpenCV errors
os.environ["RS_SILENCE_WARNINGS"] = "1"  # Silence RealSense warnings

# Setup minimal logging - only show warnings and higher
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), "yoloflow.log")),
        logging.StreamHandler()
    ]
)
# Filter out non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger('YOLOflow')

# Conditionally import torch - don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch available, version: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using basic segmentation mode")

class VideoCapture:
    """Thread-safe video capture class"""
    def __init__(self, src=0):
        # Redirect OpenCV errors to dev/null
        stderr_fileno = os.dup(2)  # Save stderr
        null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(null_fd, 2)  # Redirect stderr to null
        
        try:
            # Initialize capture with errors suppressed
            self.cap = cv2.VideoCapture(src)
            self.ret, self.frame = self.cap.read()
        finally:
            # Restore stderr
            os.dup2(stderr_fileno, 2)
            os.close(null_fd)
        
        # Default frame if camera not available
        if self.frame is None:
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(self.frame, "No camera available", (80, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.ret = False
        
        self.lock = threading.Lock()
        
        # Start background thread for reading frames
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.running = True
        self.thread.start()
    
    def _update(self):
        """Background thread to continuously update frames"""
        while self.running:
            if not self.cap.isOpened():
                break
                
            # Redirect stderr to null to prevent OpenCV errors
            stderr_fileno = os.dup(2)
            null_fd = os.open(os.devnull, os.O_RDWR)
            os.dup2(null_fd, 2)
            
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
            except Exception:
                # Silently ignore errors during capture
                pass
            finally:
                # Restore stderr
                os.dup2(stderr_fileno, 2)
                os.close(null_fd)
            
            time.sleep(0.01)  # Small delay to prevent CPU hogging
    
    def read(self):
        """Read the latest frame"""
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None
    
    def isOpened(self):
        """Check if the camera is opened"""
        return self.cap.isOpened()
    
    def release(self):
        """Release resources"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


class NDICapture:
    """Video capture using cyndilib for NDI support or demo pattern."""
    # Implementation omitted for brevity
    pass


class FrameProcessor:
    """Process frames with YOLO detection"""
    # Implementation omitted for brevity
    pass

def create_folder_if_not_exists(folder_path):
    """Create a folder if it doesn't exist"""
    # Implementation omitted for brevity
    pass

def get_config_path():
    """Get path to config file"""
    # Get root directory (parent of the directory of this file)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, 'config.yaml')

def load_config():
    """Load configuration from YAML file"""
    try:
        import yaml
        config_path = get_config_path()
        if os.path.exists(config_path):
            logger.debug(f"Loading config from {config_path}")
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        else:
            logger.warning(f"Config file not found at {config_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def is_ndi_available():
    """Check if NDI sources are available on the network"""
    # Attempt to detect NDI sources using cyndilib
    print("\n*** CHECKING FOR NDI SOURCES ***")
    logger.info("Checking for NDI sources")
    
    # Get the target NDI source name
    source_name = get_ndi_source_name()
    print(f"Looking for NDI source: {source_name}")
    
    # Check for NDI sources using cyndilib if available
    if NDI_LIB_AVAILABLE:
        try:
            # Use the finder to get available sources (correct method for v0.0.5)
            finder = ndi.Finder()
            finder.open()  # Open the finder first
            finder.wait_for_sources(timeout=5)  # Wait for sources (5 sec timeout)
            source_names = finder.get_source_names()  # Get list of source names
            
            if source_names:
                print(f"✓ SUCCESS! Found {len(source_names)} NDI sources:")
                for name in source_names:
                    print(f"  - {name}")
                
                # Look for our specific source using improved matching logic
                matching_sources = []
                partial_matches = []
                
                # First collect all sources with matching stream names
                for name in source_names:
                    source = finder.get_source(name)
                    if source:
                        stream_name = getattr(source, 'stream_name', '')
                        if stream_name and source_name.lower() == stream_name.lower():
                            matching_sources.append((name, source))
                        elif source_name.lower() in name.lower():
                            partial_matches.append((name, source))
                
                # Check if we found any matches
                target_found = False
                if matching_sources:
                    if len(matching_sources) == 1:
                        name, _ = matching_sources[0]
                        print(f"✓ SUCCESS! Found unique NDI source with matching stream name: {name}")
                    else:
                        # Multiple matches with same stream name
                        name, _ = matching_sources[0]
                        print(f"✓ SUCCESS! Found {len(matching_sources)} NDI sources with stream name '{source_name}'")
                        print(f"  First match: {name}")
                    target_found = True
                elif partial_matches:
                    name, _ = partial_matches[0]
                    print(f"✓ SUCCESS! Found target NDI source by partial name match: {name}")
                    target_found = True
                
                if not target_found:
                    print(f"× Target source '{source_name}' not found, but will use available sources")
                
                # Clean up
                finder.close()
                print("*** END NDI CHECK ***\n")
                return True
        except Exception as e:
            print(f"Error checking NDI with cyndilib: {e}")
            # Try to clean up if needed
            try:
                if 'finder' in locals() and finder:
                    finder.close()
            except:
                pass
    
    # Still nothing found - will use demo pattern but return True
    print("No NDI sources found - will use demo pattern")
    print("*** END NDI CHECK ***\n")
    return True

def get_ndi_source_name():
    """Get the name of the NDI source to connect to"""
    config = load_config()
    return config.get('ndi', {}).get('source_name', 'TD-OUT')  # Default to 'TD-OUT'

def update_ndi_config(new_source_name):
    """Update the config.yaml file with a new NDI source name"""
    try:
        import yaml
        config_path = get_config_path()
        
        # Load existing config
        config = load_config()
        
        # Update the NDI source name
        if 'ndi' not in config:
            config['ndi'] = {}
        
        config['ndi']['source_name'] = new_source_name
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"✓ Successfully updated config.yaml to use NDI source: {new_source_name}")
        print("  Restart the application for changes to take effect")
        return True
    except Exception as e:
        print(f"× Error updating config: {e}")
        return False

def list_ndi_sources():
    """List all available NDI sources - SIMPLIFIED FOR BETTER DETECTION"""
    if NDI_LIB_AVAILABLE:
        print("\n*** SEARCHING FOR ALL NDI SOURCES ***")
        
        # Try 3 different approaches to maximize chances of finding all sources
        all_sources = []
        
        # APPROACH 1: Quick initial scan
        print("SCAN 1: Fast initial scan...")
        finder1 = ndi.Finder()
        finder1.open()
        finder1.wait_for_sources(timeout=0.5)  # Very short initial wait
        sources1 = finder1.get_source_names()
        print(f"  Found {len(sources1)} sources initially")
        all_sources.extend([s for s in sources1 if s not in all_sources])
        
        # APPROACH 2: Multiple consecutive scans
        print("SCAN 2: Multiple quick consecutive scans...")
        for i in range(3):
            finder1.wait_for_sources(timeout=0.5)
            sources = finder1.get_source_names()
            new_sources = [s for s in sources if s not in all_sources]
            if new_sources:
                print(f"  Scan {i+1}: Found {len(new_sources)} new source(s)")
                all_sources.extend(new_sources)
        finder1.close()
        
        # APPROACH 3: Create a fresh finder and try again
        print("SCAN 3: Fresh finder with longer timeout...")
        finder2 = ndi.Finder()
        finder2.open()
        finder2.wait_for_sources(timeout=2)  # Longer wait with fresh finder
        sources3 = finder2.get_source_names()
        new_sources = [s for s in sources3 if s not in all_sources]
        if new_sources:
            print(f"  Found {len(new_sources)} additional sources")
            all_sources.extend(new_sources)
        
        # Final preparation of source list
        final_sources = []
        print("\nRETRIEVING DETAILED SOURCE INFORMATION:")
        print("--------------------------------------------------------------------------------")
        print("| Index | Full NDI Name                  | Host Name          | Stream Name    |")
        print("| ----- | ------------------------------ | ------------------ | -------------- |")
        print("| NOTE: Only the Stream Name is used for matching in config.yaml             |")
        print("| TIP: If multiple sources have the same Stream Name, the first one is used  |")
        print("--------------------------------------------------------------------------------")
        
        for i, name in enumerate(all_sources):
            # Get detailed source information
            try:
                source = finder2.get_source(name)
                if source:
                    final_sources.append(source)
                    
                    # Extract host name and stream name if possible
                    host_name = getattr(source, 'host_name', 'Unknown')
                    stream_name = getattr(source, 'stream_name', 'Unknown')
                    
                    # Print detailed information
                    print(f"| {i+1:<5} | {name:<30} | {host_name:<18} | {stream_name:<14} |")
            except Exception as src_error:
                # Just print the name if we can't get more details
                print(f"| {i+1:<5} | {name:<30} | {'Error getting details':<34} |")
        
        print("--------------------------------------------------------------------------------")
        
        # If we found sources, provide guidance
        if final_sources:
            print(f"\nSUCCESS! Found {len(final_sources)} total NDI sources.")
            print("\nTo use a specific NDI source in TouchDesigner:")
            print("1. Note the exact Stream Name above (e.g., 'Remote Connection 1')")
            print("2. In TouchDesigner, set your NDI OUT TOP name to match this Stream Name")
            print("3. Alternatively, update config.yaml to set ndi.source_name to match one of these")
        else:
            print("\nNO NDI SOURCES FOUND after multiple scan attempts")
            print("\nTroubleshooting tips:")
            print("1. Make sure your NDI source is active and broadcasting")
            print("2. Check that NDI Tools is installed: https://ndi.video/tools/")
            print("3. Verify with NDI Studio Monitor that sources are visible on the network")
            print("4. Check network firewall settings for NDI traffic (port 5353 for mDNS discovery)")
            
        # Clean up
        finder2.close()
        print("*** END NDI SOURCE LIST ***\n")
        return final_sources
    else:
        print("\n*** AVAILABLE NDI SOURCES ***")
        print("cyndilib not available - using demo pattern")
        print("*** END NDI SOURCE LIST ***\n")
        return []