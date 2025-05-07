import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class TrackerWorker(QThread):
    """Worker thread for SAM2 tracking to avoid blocking UI"""
    result_ready = pyqtSignal(np.ndarray, tuple)
    
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.current_frame = None
        self.running = True
        self.processing = False
        self.lock = None  # Will be initialized when needed
        
    def set_frame(self, frame):
        """Set a new frame to be processed"""
        # Initialize lock if needed
        if self.lock is None:
            import threading
            self.lock = threading.Lock()
            
        # Thread-safe frame update
        with self.lock:
            self.current_frame = frame.copy()
            if not self.processing:
                self.processing = True
        
    def run(self):
        """Main worker thread loop"""
        import threading
        if self.lock is None:
            self.lock = threading.Lock()
            
        while self.running:
            # Check if we have a frame to process
            process_frame = None
            with self.lock:
                if self.processing and self.current_frame is not None:
                    process_frame = self.current_frame.copy()
            
            # Process frame if available
            if process_frame is not None:
                try:
                    vis_frame, bbox = self.tracker.process_frame(process_frame)
                    if vis_frame is not None:
                        self.result_ready.emit(vis_frame, bbox)
                except Exception as e:
                    print(f"Error in tracker worker: {e}")
                
                # Mark as done processing
                with self.lock:
                    self.processing = False
            else:
                # Sleep to prevent CPU hogging
                time.sleep(0.01)
                
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.wait()