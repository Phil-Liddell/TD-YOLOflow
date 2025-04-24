"""
YOLOflow TouchDesigner Headless Launcher

Simple script to launch YOLOflow in headless mode and set up shared memory connection.
"""

import os
import subprocess
import time
import sys

# TouchDesigner-specific imports
TD_AVAILABLE = False
try:
    # Only works in TouchDesigner
    import td
    TD_AVAILABLE = True
    print("Running inside TouchDesigner")
except ImportError:
    print("Not running inside TouchDesigner")

def log(message):
    """Print log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def launch_yoloflow():
    """Launch YOLOflow in headless mode and set up shared memory connection"""
    
    # Get directory where this script is located
    if TD_AVAILABLE:
        # In TouchDesigner, use the directory of the current .toe file
        if hasattr(project, 'folder'):
            script_dir = os.path.dirname(project.folder)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        # When running directly, use the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    log(f"Script directory: {script_dir}")
    
    # Path to the auto_inference script
    auto_inference_path = os.path.join(script_dir, "scripts", "auto_inference.py")
    
    if not os.path.exists(auto_inference_path):
        log(f"ERROR: auto_inference.py not found at {auto_inference_path}")
        return False
    
    # Get Python executable
    python_path = sys.executable
    log(f"Using Python: {python_path}")
    
    # Launch the headless process
    try:
        log("Launching YOLOflow headless mode...")
        
        # On Windows, use CREATE_NO_WINDOW flag
        if os.name == 'nt':  # Windows
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            process = subprocess.Popen(
                [python_path, auto_inference_path],
                cwd=script_dir,
                startupinfo=startupinfo
            )
        else:  # macOS, Linux
            process = subprocess.Popen(
                [python_path, auto_inference_path],
                cwd=script_dir
            )
        
        log(f"Headless process started with PID: {process.pid}")
        
        # Wait for shared memory to initialize
        log("Waiting for shared memory configuration...")
        for i in range(10):  # Try for up to 10 seconds
            shm_path = os.path.expanduser("~/Documents/yoloflow_shared_memory.txt")
            if os.path.exists(shm_path):
                break
            log(f"Waiting... ({i+1}/10)")
            time.sleep(1)
            
        if not os.path.exists(shm_path):
            log("ERROR: Shared memory configuration not found")
            return False
            
        # Read shared memory configuration
        log("Reading shared memory configuration...")
        try:
            shm_info = {}
            with open(shm_path, 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    shm_info[key] = value
            
            log("Shared memory information:")
            for key, value in shm_info.items():
                log(f"  {key} = {value}")
                
            # Set up TouchDesigner TOPs if running in TouchDesigner
            if TD_AVAILABLE:
                setup_touchdesigner_tops(shm_info)
                
            return True
            
        except Exception as e:
            log(f"ERROR: Failed to read shared memory config: {e}")
            return False
            
    except Exception as e:
        log(f"ERROR: Failed to launch headless process: {e}")
        return False

def setup_touchdesigner_tops(shm_info):
    """Set up TouchDesigner TOPs for shared memory connection"""
    try:
        # Get the current network
        parent = op.parent()
        
        # Create SharedMem TOP
        shm_top = parent.create(TOP, "YOLOflow_SharedMem")
        shm_top.nodeX = op.nodeX + 200
        shm_top.nodeY = op.nodeY
        shm_top.operator = "sharedmem"
        shm_top.par.Memorynamee = shm_info.get('main_name', '')
        shm_top.par.Sizex = int(shm_info.get('width', 1280))
        shm_top.par.Sizey = int(shm_info.get('height', 720))
        shm_top.par.Format = "RGBA8"
        shm_top.par.Sharingmode = "Read"
        
        # Create NULL TOP to display the result
        null_top = parent.create(TOP, "YOLOflow_Output")
        null_top.nodeX = shm_top.nodeX + 200
        null_top.nodeY = shm_top.nodeY
        null_top.inputConnectors[0].connect(shm_top)
        
        log("TouchDesigner TOPs created successfully")
    except Exception as e:
        log(f"ERROR: Failed to create TouchDesigner TOPs: {e}")

# Run when executed directly or from TouchDesigner
if __name__ == "__main__" or TD_AVAILABLE:
    launch_yoloflow()