import os
import sys
import logging
import warnings
import time
import threading
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
from PyQt5.QtCore import QTimer

# Add config for NDI if it doesn't exist
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
if os.path.exists(config_path):
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
            
        # Add NDI settings if they don't exist
        if 'ndi' not in config:
            config['ndi'] = {
                'source_name': 'TD-OUT',
                'prefer_ndi_over_camera': True,
                'width': 1280,
                'height': 720,
                'fps': 30
            }
            
                
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Updated configuration in config.yaml")
    except Exception as e:
        print(f"Error updating config: {e}")

# Configure logging to only show warnings and errors by default
# This keeps the log files cleaner by reducing less important messages
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s - YOLOflow - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yoloflow.log")),
        logging.StreamHandler()
    ]
)
# Disable propagation of OSC-related logs
logging.getLogger("pythonosc").setLevel(logging.WARNING)
# Filter out non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add startup timestamp for performance tracking
startup_time = time.time()

# Configure PyQt to not show debug output
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'

# OSC server configuration
OSC_IP = "127.0.0.1"  # localhost
OSC_PORT = 9000  # Changed from 8600 to avoid conflicts
osc_server_instance = None
osc_message_count = 0  # Counter for messages

# Track previous OSC values to detect changes
last_osc_values = {}  # Dictionary to store the last value of each address
app_instance = None   # Store reference to app instance for closing

from tabs.capture_tab import CaptureTab
from tabs.review_tab import ReviewTab
from tabs.train_tab import TrainTab
from tabs.test_tab import TestTab

# PERFORMANCE FIX: Import and initialize the centralized NDI manager
# Using a single shared NDI connection is essential for optimal performance
from tabs import ndi_manager
try:
    # Force reload of NDI manager module to ensure fresh state
    import importlib
    importlib.reload(ndi_manager)
    
    # Initialize NDI manager with config at application startup
    # This ensures all tabs use the same shared NDI connection
    print("Initializing central NDI manager at startup...")
    success = ndi_manager.initialize(config)
    
    if ndi_manager.is_initialized():
        print("✓ NDI Manager initialized successfully")
        logging.info("NDI Manager initialized successfully at startup")
    else:
        print(f"× NDI Manager initialization failed: {ndi_manager.get_last_error()}")
        logging.warning(f"NDI Manager initialization failed: {ndi_manager.get_last_error()}")
except Exception as e:
    print(f"× Error initializing NDI Manager: {str(e)}")
    logging.error(f"Error initializing NDI Manager: {str(e)}")
    import traceback
    traceback.print_exc()

class YOLOFlowApp(QMainWindow):
    def __init__(self):
        logging.info("Initializing main application window")
        init_start = time.time()
        
        super().__init__()
        self.setWindowTitle("YOLOflow - Object Detection Workflow Tool (OSC Enabled)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create a status bar
        self.statusBar().showMessage(f"OSC Server listening on {OSC_IP}:{OSC_PORT}")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                padding: 12px 24px;
                border: 1px solid #cccccc;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 140px;
                font-weight: bold;
                font-size: 24px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #eeeeee;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #aaaaaa;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 6px;
                color: #333333;
            }
            QPushButton {
                border: 1px solid #aaaaaa;
                border-radius: 4px;
                padding: 6px 12px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0d8aee;
            }
            QPushButton:pressed {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        logging.info(f"UI styling initialized ({time.time() - init_start:.3f}s)")
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("background-color: white;")
        
        # Create and add only the first tab initially - lazy load others
        logging.info("Creating initial tab (Capture)")
        self.capture_tab = CaptureTab()
        self.tabs.addTab(self.capture_tab, "Capture")
        
        # Create placeholders for other tabs (will be loaded when needed)
        self.review_tab = None
        self.train_tab = None
        self.test_tab = None
        
        # Add placeholder tabs
        self.tabs.addTab(QWidget(), "Review")
        self.tabs.addTab(QWidget(), "Train")
        self.tabs.addTab(QWidget(), "Test")
        
        # Connect tabs to share data and lazy-load
        self.tabs.currentChanged.connect(self.on_tab_change)
        
        # Set central widget
        self.setCentralWidget(self.tabs)
        
        logging.info(f"Main window initialization complete ({time.time() - init_start:.3f}s)")
    
    def on_tab_change(self, index):
        logging.info(f"Switching to tab index {index}")
        tab_switch_start = time.time()
        
        # Lazy-load tab if it hasn't been initialized yet
        if index == 1 and self.review_tab is None:  # Review tab
            logging.info("Lazy-loading Review tab")
            self.review_tab = ReviewTab()
            # Replace placeholder with actual tab
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, self.review_tab, "Review")
            self.tabs.setCurrentIndex(index)
            
            # Get the data directory from the capture tab
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            
            # Use a timer to delay loading until after the tab is fully displayed
            # This prevents UI from appearing frozen during tab switch
            QTimer.singleShot(100, lambda: self.review_tab.load_data(data_dir))
            self.review_tab.status_label.setText("Preparing to load data...")
            logging.info(f"Switching to Review tab, loading data from: {data_dir}")
            
        elif index == 2 and self.train_tab is None:  # Train tab
            logging.info("Lazy-loading Train tab")
            self.train_tab = TrainTab()
            # Replace placeholder with actual tab
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, self.train_tab, "Train")
            self.tabs.setCurrentIndex(index)
            
        elif index == 3 and self.test_tab is None:  # Test tab
            logging.info("Lazy-loading Test tab")
            self.test_tab = TestTab()
            # Replace placeholder with actual tab
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, self.test_tab, "Test")
            self.tabs.setCurrentIndex(index)
            
        logging.info(f"Tab switch completed in {time.time() - tab_switch_start:.3f}s")
        
    def closeEvent(self, event):
        # Clean up resources when closing the application
        self.capture_tab.cleanup()
        
        # Clean up other tabs if initialized
        if self.test_tab is not None:
            self.test_tab.cleanup()
        
        # Clean up centralized NDI manager
        from tabs import ndi_manager
        try:
            ndi_manager.cleanup()
            logging.info("NDI Manager resources released")
        except Exception as e:
            logging.error(f"Error cleaning up NDI Manager: {str(e)}")
            
        super().closeEvent(event)


def osc_message_handler(address, *args):
    """Handler for all incoming OSC messages"""
    global osc_message_count, last_osc_values, app_instance
    
    # Immediately return for ALL common messages
    if address in ["/_samplerate", "/Openyoloflowworkflow", "/Closeyoloflowworkflow"]:
        # For special case of closing the app with value 1.0
        if address == "/Closeyoloflowworkflow" and args and args[0] == 1.0:
            # Force immediate exit
            os._exit(0)
        # Otherwise completely ignore all OSC messages
        return
    
    # Only handle custom messages that aren't the standard ones 
    # (this code will almost never run with normal TouchDesigner usage)
    value = args[0] if args else 0.0
    
    # Update internal tracking only
    last_osc_values[address] = value
    
    # This program receives only standard messages normally,
    # so we won't output anything to avoid console clutter

def start_osc_server():
    """Start the OSC server in a separate thread"""
    global osc_server_instance, OSC_PORT
    
    # Try multiple ports if the first one fails
    max_port_attempts = 5
    for attempt in range(max_port_attempts):
        try:
            # Create OSC dispatcher and set default handler for all messages
            dispatcher = osc_dispatcher.Dispatcher()
            dispatcher.set_default_handler(osc_message_handler)
            
            # Create and start OSC server
            port_to_try = OSC_PORT + attempt  # Try increasing port numbers
            server = osc_server.ThreadingOSCUDPServer((OSC_IP, port_to_try), dispatcher)
            osc_server_instance = server
            
            # Update the global port in case we had to use a different one
            OSC_PORT = port_to_try
            
            # No need to log server start as INFO - it's a normal operation
            print(f"OSC: {OSC_PORT}")
            
            # Start server in a separate thread
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return True
            
        except OSError as e:
            if "Only one usage of each socket address" in str(e) and attempt < max_port_attempts - 1:
                logging.warning(f"Port {port_to_try} is in use, trying port {port_to_try + 1}")
                continue  # Try the next port
            logging.error(f"Failed to start OSC server: {str(e)}")
            print(f"Error starting OSC server: {str(e)}")
            return False
            
        except Exception as e:
            logging.error(f"Failed to start OSC server: {str(e)}")
            print(f"Error starting OSC server: {str(e)}")
            return False
    
    return False  # Failed to start after all attempts

def stop_osc_server():
    """Stop the OSC server if it's running"""
    global osc_server_instance
    if osc_server_instance:
        # No need to log server stop since it's normal behavior
        osc_server_instance.shutdown()
        osc_server_instance = None

def main():
    main_start_time = time.time()
    print("START: main")
    
    # Store app instance for closing from OSC
    global app_instance
    
    app = QApplication(sys.argv)
    print("STARTUP: Application starting")
    
    # Start OSC server
    start_osc_server()
    
    # Create main window
    window = YOLOFlowApp()
    
    # Store reference for OSC-triggered closing
    app_instance = window
    
    # Add cleanup handler for OSC server when application closes
    app.aboutToQuit.connect(stop_osc_server)
    
    # Show the window
    window.show()
    print(f"Ready! ({time.time() - startup_time:.1f}s startup time) - OSC: {OSC_PORT}")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
