import os
import cv2
import time
import numpy as np
import threading
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QFormLayout, QGroupBox, QFileDialog, QMessageBox,
                             QSlider, QCheckBox, QApplication, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QFileSystemWatcher
from PyQt5.QtGui import QImage, QPixmap
from .utils import NDICapture, is_ndi_available, get_ndi_source_name, load_config, NDI_LIB_AVAILABLE
from pythonosc import udp_client

class TestTab(QWidget):
    def __init__(self):
        super().__init__()
        
        print("TEST TAB: Initializing")
        
        # Video capture
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Model
        self.model = None
        self.model_path = ""
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Class names
        self.class_names = []
        
        # Model information - built as a dictionary for easy lookup
        self.available_models = {}
        
        self.last_annotated = None      # stores the most‑recent labelled frame
        self.raw_passthrough = True     # controls whether raw frames are allowed out
        
        # Load config first
        self.config = load_config()
        
        # NDI configuration
        self.ndi_available = is_ndi_available()
        self.prefer_ndi = self.config.get('ndi', {}).get('prefer_ndi_over_camera', True)
        self.using_ndi = False
        self.inference_active = False
        
        print(f"TEST TAB: NDI available: {self.ndi_available}, prefer NDI: {self.prefer_ndi}")
        
        # Auto-load a model on startup - this will be done after UI setup
        QTimer.singleShot(1000, self.auto_init_all)
        
        # Shared memory configuration - always enabled
        self.shm_sender = None
        self.ndi_output_width = self.config.get('ndi', {}).get('width', 1280)
        self.ndi_output_height = self.config.get('ndi', {}).get('height', 720)
        self._debug_printed = False  # For one-time debug printing
        
        # Don't initialize shared memory immediately - we'll do it when needed
        self.shm_sender = None
        
        # Models directory
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        
        # OSC detection client for sending detections to external applications
        self.osc_det_client = udp_client.SimpleUDPClient("127.0.0.1", 8860)
        
        # Set up file system watcher to monitor models directory
        self.fs_watcher = QFileSystemWatcher()
        self.fs_watcher.addPath(self.models_dir)
        self.fs_watcher.directoryChanged.connect(self.on_directory_changed)
        
        # Set up UI
        self.setup_ui()
        
        # Populate model dropdown after UI is set up
        QTimer.singleShot(500, self.populate_model_dropdown)
        
        # If NDI is preferred, start it automatically after a short delay
        if self.ndi_available and self.prefer_ndi:
            QTimer.singleShot(1000, self.start_camera)
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)  # Add more spacing between elements
        
        # Common style for all group boxes
        # Note: Some variables still use 'spout' in their name for backward compatibility
        group_box_style = """
            QGroupBox {
                font-weight: bold;
                font-size: 18px;  /* Much larger section labels */
                border: 1px solid #bbb;
                border-radius: 6px;
                margin-top: 16px; /* More margin for larger font */
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 8px;
            }
        """
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_group.setStyleSheet(group_box_style)
        model_layout = QFormLayout()
        model_layout.setSpacing(10)  # Add more spacing between form elements
        
        # Model dropdown with improved arrow
        self.model_dropdown = QComboBox()
        self.model_dropdown.setStyleSheet("""
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 6px 12px;
                background-color: #f5f5f5;
                min-height: 32px;
                min-width: 300px;
                font-weight: bold;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-left: none;
                margin-right: 4px;
            }
            QComboBox::down-arrow {
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="%23555" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>');
                width: 20px;
                height: 20px;
            }
            QComboBox::down-arrow:hover {
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="%230d8aee" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>');
            }
            QComboBox:hover {
                background-color: #e5e5e5;
                border-color: #bbbbbb;
            }
            QComboBox:focus {
                border-color: #2196F3;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                selection-background-color: #2196F3;
                selection-color: white;
                outline: none;
            }
        """)
        self.model_dropdown.currentIndexChanged.connect(self.on_model_selected)
        
        # Model control buttons layout
        model_buttons_layout = QHBoxLayout()
        
        # Load button - for external models
        self.browse_model_btn = QPushButton("Load External Model")
        self.browse_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 32px;
            }
            QPushButton:hover {
                background-color: #0d8aee;
            }
            QPushButton:pressed {
                background-color: #0b7dda;
            }
        """)
        self.browse_model_btn.clicked.connect(self.browse_model)
        
        # Refresh button
        self.refresh_models_btn = QPushButton("Refresh Model List")
        self.refresh_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 32px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.refresh_models_btn.clicked.connect(self.populate_model_dropdown)
        
        # Add model control buttons to layout
        model_buttons_layout.addWidget(self.browse_model_btn)
        model_buttons_layout.addWidget(self.refresh_models_btn)
        
        # Add components to form layout
        model_layout.addRow("Model:", self.model_dropdown)
        model_layout.addRow("", model_buttons_layout)
        
        # Auto-detect model on startup
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)  # Default 0.25
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -8px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #0d8aee;
            }
            QSlider::sub-page:horizontal {
                background: #aad4f5;
                border: 1px solid #999999;
                border-radius: 4px;
            }
        """)
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.conf_label = QLabel("0.25")
        self.conf_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 6px;
                background-color: #f5f5f5;
                min-width: 60px;
                font-weight: bold;
                text-align: center;
            }
        """)
        self.conf_label.setAlignment(Qt.AlignCenter)
        threshold_layout.addWidget(self.conf_slider, 4)
        threshold_layout.addWidget(self.conf_label, 1)
        model_layout.addRow("Confidence Threshold:", threshold_layout)
        
        # IOU threshold
        iou_layout = QHBoxLayout()
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)  # Default 0.45
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)
        self.iou_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -8px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #0d8aee;
            }
            QSlider::sub-page:horizontal {
                background: #aad4f5;
                border: 1px solid #999999;
                border-radius: 4px;
            }
        """)
        self.iou_slider.valueChanged.connect(self.update_iou_threshold)
        self.iou_label = QLabel("0.45")
        self.iou_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 6px;
                background-color: #f5f5f5;
                min-width: 60px;
                font-weight: bold;
                text-align: center;
            }
        """)
        self.iou_label.setAlignment(Qt.AlignCenter)
        iou_layout.addWidget(self.iou_slider, 4)
        iou_layout.addWidget(self.iou_label, 1)
        model_layout.addRow("IoU Threshold:", iou_layout)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Video input group (renamed from Camera Settings)
        input_title = "Video Input" if self.ndi_available else "Camera Settings"
        camera_group = QGroupBox(input_title)
        camera_group.setStyleSheet(group_box_style)
        camera_layout = QFormLayout()
        camera_layout.setSpacing(10)  # Add more spacing between form elements
        
        # Removed NDI source notice as requested
        
        # Change button text based on whether we're using NDI
        start_label = "Start Feed" if self.ndi_available else "Start Camera"
        stop_label = "Stop Feed" if self.ndi_available else "Stop Camera"
        
        # Removed feed status indicator completely
        
        # Add inference control buttons
        inference_buttons_layout = QHBoxLayout()
        self.start_inference_btn = QPushButton("Start Inference")
        self.start_inference_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;  /* Green color */
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #45a049;  /* Darker green on hover */
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.stop_inference_btn = QPushButton("Stop Inference")
        self.stop_inference_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;  /* Red color */
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #d32f2f;  /* Darker red on hover */
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_inference_btn.clicked.connect(self.start_inference)
        self.stop_inference_btn.clicked.connect(self.stop_inference)
        self.start_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(False)
        inference_buttons_layout.addWidget(self.start_inference_btn)
        inference_buttons_layout.addWidget(self.stop_inference_btn)
        camera_layout.addRow("Inference:", inference_buttons_layout)
        
        camera_group.setLayout(camera_layout)
        main_layout.addWidget(camera_group)
        
        # Display options with better styling
        display_group = QGroupBox("Display Options")
        display_group.setStyleSheet(group_box_style)
        display_layout = QVBoxLayout()
        display_layout.setSpacing(10)  # Add more spacing
        
        # Style the checkboxes
        checkbox_style = """
            QCheckBox {
                spacing: 8px;
                font-size: 14px;
                min-height: 28px;
                padding: 4px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 1px solid #999;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: #f5f5f5;
            }
            
            QCheckBox::indicator:unchecked:hover {
                background-color: #e5e5e5;
            }
            
            QCheckBox::indicator:checked {
                background-color: #2196F3;
                border: 1px solid #0d8aee;
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>');
            }
            
            QCheckBox::indicator:checked:hover {
                background-color: #0d8aee;
            }
        """
        
        # Create a horizontal layout for the checkboxes
        check_box_layout = QHBoxLayout()
        
        # Show FPS checkbox
        self.show_fps_checkbox = QCheckBox("Show FPS Counter")
        self.show_fps_checkbox.setChecked(True)
        self.show_fps_checkbox.setStyleSheet(checkbox_style)
        check_box_layout.addWidget(self.show_fps_checkbox)
        
        # Show Labels checkbox
        self.show_labels_checkbox = QCheckBox("Show Object Labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.setStyleSheet(checkbox_style)
        check_box_layout.addWidget(self.show_labels_checkbox)
        
        # Add the horizontal layout to the display layout
        display_layout.addLayout(check_box_layout)
        
        # Removed TouchDesigner Shared Memory status indicator
        
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)
        
        # Camera view with improved styling
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(640, 480)  # Standard size
        self.camera_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expanding
        self.camera_view.setStyleSheet("""
            QLabel {
                border: 2px solid #bbb;
                border-radius: 8px;
                padding: 8px;
                background-color: #f8f8f8;
                margin: 10px 0;
            }
        """)
        main_layout.addWidget(self.camera_view)
        
        # Completely removed status bar
        
        self.setLayout(main_layout)
    
    # Removed populate_camera_list method as we always use NDI now
    
    def update_conf_threshold(self, value):
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"{self.conf_threshold:.2f}")
    
    def update_iou_threshold(self, value):
        self.iou_threshold = value / 100.0
        self.iou_label.setText(f"{self.iou_threshold:.2f}")
        
    def auto_init_all(self):
        """Auto-initialize everything needed for the test tab to work"""
        print("TEST TAB: Auto-initializing all components")
        
        # 1. First try to find and load a model
        try:
            # Look for any .pt file in the models directory
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                if model_files:
                    # Use the first model found
                    model_path = os.path.join(models_dir, model_files[0])
                    print(f"TEST TAB: Auto-loading model {model_path}")
                    
                    # Set model path in UI
                    self.model_path = model_path
                    self.model_path_input.setText(os.path.basename(model_path))
                    
                    # Load the model
                    import torch
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    
                    # Try to load classes
                    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
                    classes_txt = os.path.join(data_dir, "classes.txt")
                    if os.path.exists(classes_txt):
                        with open(classes_txt, 'r') as f:
                            self.class_names = [line.strip() for line in f.readlines()]
                    
                    # If no classes found, use defaults
                    if not self.class_names:
                        self.class_names = ["object"]
                    
                    print(f"TEST TAB: Model loaded with {len(self.class_names)} classes")
        except Exception as e:
            print(f"TEST TAB: Error auto-loading model: {e}")
        
        # 2. Now automatically start the camera with our own direct NDI connection
        QTimer.singleShot(500, self.start_camera)
        
    def cleanup(self):
        """Clean up resources when the tab is closed"""
        # Stop camera if running
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clean up shared memory resources
        self.cleanup_shared_memory()
    
    def populate_model_dropdown(self):
        """Populate the model dropdown with available models"""
        # Clear existing items
        self.model_dropdown.clear()
        self.available_models = {}
        
        # Add a placeholder item
        self.model_dropdown.addItem("Select a model...")
        
        # Scan for models in various locations
        model_found = False
        
        # First category: Models in the models directory
        standard_models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith(".pt"):
                    model_path = os.path.join(self.models_dir, file)
                    # Try to get nice display name
                    display_name = self.get_model_display_name(file)
                    standard_models.append((display_name, model_path))
        
        if standard_models:
            # Add a category separator
            self.model_dropdown.insertSeparator(self.model_dropdown.count())
            self.model_dropdown.addItem("--- Standard Models ---")
            self.model_dropdown.insertSeparator(self.model_dropdown.count())
            
            # Sort standard models (put default models first)
            standard_models.sort(key=lambda x: (not x[0].startswith("Nano"), 
                                               not x[0].startswith("Small"),
                                               not x[0].startswith("Medium"), 
                                               not x[0].startswith("Large"),
                                               not x[0].startswith("XLarge"),
                                               x[0]))
            
            # Add standard models to dropdown
            for display_name, model_path in standard_models:
                self.model_dropdown.addItem(display_name)
                self.available_models[display_name] = model_path
            model_found = True
        
        # Second category: Trained models
        trained_models = []
        
        # Check train directory
        runs_dir = os.path.join(os.path.dirname(self.models_dir), "runs/train")
        if os.path.exists(runs_dir):
            train_subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                     if os.path.isdir(os.path.join(runs_dir, d))]
            
            # Sort by modification time (newest first)
            if train_subdirs:
                train_subdirs.sort(key=os.path.getmtime, reverse=True)
                for subdir in train_subdirs:
                    weights_path = os.path.join(subdir, "weights/best.pt")
                    if os.path.exists(weights_path):
                        run_name = os.path.basename(subdir)
                        display_name = f"Trained: {run_name}"
                        trained_models.append((display_name, weights_path))
        
        # If there are trained models, add them to the dropdown
        if trained_models:
            # Add a category separator
            self.model_dropdown.insertSeparator(self.model_dropdown.count())
            self.model_dropdown.addItem("--- Trained Models ---")
            self.model_dropdown.insertSeparator(self.model_dropdown.count())
            
            for display_name, model_path in trained_models:
                self.model_dropdown.addItem(display_name)
                self.available_models[display_name] = model_path
            model_found = True
        
        # Select default model if available
        if model_found:
            # Try to find a default model
            default_index = -1
            
            # First try to find Large model (preference order: Large, Medium, Small, Nano, XLarge)
            for i in range(self.model_dropdown.count()):
                text = self.model_dropdown.itemText(i)
                if text == "Large":
                    default_index = i
                    break
            
            # If no Large model, try Medium
            if default_index == -1:
                for i in range(self.model_dropdown.count()):
                    text = self.model_dropdown.itemText(i)
                    if text == "Medium":
                        default_index = i
                        break
            
            # If no Medium model, try Small
            if default_index == -1:
                for i in range(self.model_dropdown.count()):
                    text = self.model_dropdown.itemText(i)
                    if text == "Small":
                        default_index = i
                        break
                        
            # If still not found, try to pick the first non-separator, non-category model
            if default_index == -1:
                for i in range(self.model_dropdown.count()):
                    text = self.model_dropdown.itemText(i)
                    if text and text not in ("Select a model...", "--- Standard Models ---", "--- Trained Models ---") and not self.model_dropdown.itemData(i):
                        default_index = i
                        break
            
            # Select the default model
            if default_index != -1:
                self.model_dropdown.setCurrentIndex(default_index)
        else:
            print("No models found in directory")
    
    def get_model_display_name(self, filename):
        """Convert model filename to a nicer display name"""
        # Remove file extension
        name = os.path.splitext(filename)[0]
        
        # Check for Nano/Small/Medium/Large/XLarge models
        if name.endswith("n"):
            return "Nano"
        elif name.endswith("s"):
            return "Small"
        elif name.endswith("m"):
            return "Medium"
        elif name.endswith("l"):
            return "Large"
        elif name.endswith("x"):
            return "XLarge"
        elif "trained" in name.lower():
            # For trained models, use the timestamp part
            parts = name.split("_")
            if len(parts) > 2:
                return f"Trained: {parts[-1]}"
        
        # Default: use the filename
        return filename
    
    def on_model_selected(self, index):
        """Handle model selection from dropdown"""
        if index <= 0 or self.model_dropdown.itemText(index) in ("--- Standard Models ---", "--- Trained Models ---"):
            return
        
        selected_model = self.model_dropdown.currentText()
        if selected_model in self.available_models:
            model_path = self.available_models[selected_model]
            self.model_path = model_path
            # Update status to show selected model
            # self.status_label.setText(f"Status: Loading model {selected_model}...")
            self.load_model()
    
    def on_directory_changed(self, path):
        """Handler for when a monitored directory changes"""
        logging.debug(f"Directory changed: {path}")
        
        # Check which directory changed and update accordingly
        if path == self.models_dir:
            # Models directory changed, update the dropdown
            self.populate_model_dropdown()
    
    def auto_detect_model(self):
        """Auto-detect model in the models directory"""
        # Use the new populate_model_dropdown method
        self.populate_model_dropdown()
    
    def browse_model(self):
        """Browse for a model file starting in the models directory"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model", self.models_dir,
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        
        if file_path:
            self.model_path = file_path
            display_name = f"External: {os.path.basename(file_path)}"
            
            # Add to available models if not already there
            if display_name not in self.available_models:
                self.available_models[display_name] = file_path
                self.model_dropdown.addItem(display_name)
                self.model_dropdown.setCurrentText(display_name)
            
            # Update status label with loading indication
            # self.status_label.setText(f"Status: Loading external model {os.path.basename(file_path)}...")
            
            # Load the model
            self.load_model()
    
    def load_model(self):
        """Load the YOLO model with optimizations"""
        if not self.model_path or not os.path.exists(self.model_path):
            return
            
        try:
            # Update UI to show loading status
            model_name = os.path.basename(self.model_path)
            # self.status_label.setText(f"Status: Loading model {model_name}...")
            QApplication.processEvents()  # Update UI
            
            # Import only once needed
            import torch
            from ultralytics import YOLO
            
            # Load model with optimizations
            self.model = YOLO(self.model_path)
            
            # Try to use GPU if available, with improved error handling
            try:
                if torch.cuda.is_available():
                    print(f"TEST TAB: Using GPU for inference: {torch.cuda.get_device_name(0)}")
                    # Make sure model is on GPU with proper error handling
                    self.model.to('cuda')
                    print("Model successfully loaded on GPU")
                else:
                    print("TEST TAB: No GPU available, using CPU for inference")
            except Exception as gpu_err:
                print(f"TEST TAB: Error initializing GPU: {gpu_err}, falling back to CPU")
                # Ensure model runs on CPU if GPU fails
                try:
                    self.model.to('cpu')
                    print("Model successfully loaded on CPU instead")
                except:
                    pass
            
            # Load class names from various possible locations
            self.class_names = self._load_class_names()
            
            # If no classes found after all attempts, use a default
            if not self.class_names:
                self.class_names = ["object"]
            
            # Set success status
            # self.status_label.setText(f"Status: Model loaded with {len(self.class_names)} classes")
            
            # Enable the Start Inference button
            self.start_inference_btn.setEnabled(True)
            
        except Exception as e:
            self.model = None
            # self.status_label.setText(f"Status: Error loading model - {str(e)}")
            # Disable inference button 
            self.start_inference_btn.setEnabled(False)
    
    def _load_class_names(self):
        """Helper to load class names from various possible locations"""
        class_names = []
        
        # Try multiple locations for class names
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        model_dir = os.path.dirname(self.model_path)
        
        # Places to look (in order of preference)
        locations = [
            (os.path.join(data_dir, "classes.txt"), "text"),
            (os.path.join(model_dir, "classes.txt"), "text"),
            (os.path.join(model_dir, "dataset.yaml"), "yaml"),
            (os.path.join(data_dir, "dataset.yaml"), "yaml"),
        ]
        
        # Try each location
        for location, file_type in locations:
            if os.path.exists(location):
                try:
                    if file_type == "text":
                        with open(location, 'r') as f:
                            class_names = [line.strip() for line in f.readlines()]
                        print(f"TEST TAB: Loaded {len(class_names)} classes from {location}")
                        if class_names:
                            break
                    elif file_type == "yaml":
                        import yaml
                        with open(location, 'r') as f:
                            data = yaml.safe_load(f)
                            if 'names' in data:
                                class_names = data['names']
                                print(f"TEST TAB: Loaded {len(class_names)} classes from {location}")
                                if class_names:
                                    break
                except Exception as e:
                    print(f"TEST TAB: Error loading classes from {location}: {e}")
        
        return class_names
    
    def start_camera(self):
        """Get the NDI feed from the global NDI manager"""
        # First check if we already have a running capture
        if self.cap is not None:
            try:
                # Release existing capture before creating a new one
                self.cap.release()
                self.cap = None
                print("TEST TAB: Released existing NDI capture")
                # Small delay to ensure proper release
                time.sleep(0.5)
            except Exception as e:
                print(f"TEST TAB: Error releasing existing capture: {e}")
        
        try:
            print("TEST TAB: Creating direct NDI capture")
            
            # CRITICAL PERFORMANCE FIX: Try using centralized NDI manager for shared connection
            # instead of creating a separate capture for each tab
            try:
                from . import ndi_manager
                
                # First test if NDI manager is already initialized
                if ndi_manager.is_initialized():
                    print("TEST TAB: Using centralized NDI manager for shared connection")
                    self.cap = ndi_manager.get_capture_instance()
                    if self.cap is None:
                        raise ValueError("NDI manager returned None instance")
                    print(f"TEST TAB: Using shared NDI manager instance successfully")
                else:
                    config = self.config
                    # Initialize the NDI manager
                    print("TEST TAB: Initializing NDI manager with config")
                    ndi_manager.initialize(config)
                    self.cap = ndi_manager.get_capture_instance()
                    if self.cap is None:
                        raise ValueError("Could not get NDI instance after initialization")
                    print(f"TEST TAB: Initialized and using NDI manager successfully")
            except Exception as e:
                print(f"TEST TAB: Unable to use NDI manager: {e}, falling back to direct NDI")
                
                # Fallback to direct NDI creation if manager fails
                from .utils import NDICapture
                config = self.config
                
                # Get NDI settings from config
                ndi_settings = config.get('ndi', {})
                source_name = ndi_settings.get('source_name', 'TD-OUT')
                width = ndi_settings.get('width', 1280)
                height = ndi_settings.get('height', 720)
                
                print(f"TEST TAB: Creating direct NDI capture with source={source_name}, {width}x{height}")
                
                # Create a new NDICapture instance directly
                self.cap = NDICapture(
                    sender_name=source_name,
                    width=width,
                    height=height
                )
            
            print(f"TEST TAB: NDI capture created directly, connected: {self.cap.isOpened()}")
            
            self.using_ndi = True
            
            # CRITICAL FIX: Use much higher UI refresh rate to ensure smooth display
            # Try 60 FPS for test tab to reduce perceived stutter
            viewport_fps = 60  # Higher refresh rate for smoother display
            interval = 1000 // viewport_fps  # 16.6ms interval
            
            # Enable inference controls
            self.start_inference_btn.setEnabled(True)
            
            # Start the timer
            self.timer.start(interval)
            # self.status_label.setText(f"Status: NDI feed active")
            
            # Don't auto-start inference - keep it off by default
            self.inference_active = False
                
        except Exception as e:
            print(f"TEST TAB: ERROR creating NDI capture: {str(e)}")
            import traceback
            traceback.print_exc()
            # self.status_label.setText(f"Status: Error starting NDI feed: {str(e)}")
            
            # Try again in 5 seconds
            QTimer.singleShot(5000, self.start_camera)
            # Status styling removed
    
    def initialize_memory_mapped_file(self, filename, size):
        """Initialize a memory-mapped file for sharing data with TouchDesigner"""
        import os
        import time
        import numpy as np
        import mmap
        
        try:
            # Create directory for the mapped file if it doesn't exist
            os.makedirs(os.path.expanduser("~/Documents/YOLOflow"), exist_ok=True)
            
            # Full path to the memory-mapped file
            filepath = os.path.expanduser(f"~/Documents/YOLOflow/{filename}")
            
            # Clean up existing file if it exists
            if os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                    print(f"Removed existing memory-mapped file: {filepath}")
                except Exception as e:
                    print(f"Warning: Could not remove existing file: {e}")
            
            # Create the file and ensure it has the right size
            with open(filepath, 'wb') as f:
                # Fill with zeros to allocate the space
                f.write(b'\0' * size)
                f.flush()
                
            print(f"Created memory-mapped file: {filepath}")
            
            # Open the file for shared memory access
            fd = os.open(filepath, os.O_RDWR)
            
            # Create the memory map
            mm = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
            
            # Store info about this mapped file
            self.mmap_info = {
                'filepath': filepath,
                'fd': fd,
                'mmap': mm,
                'size': size
            }
            
            return True
        
        except Exception as e:
            print(f"Error creating memory-mapped file: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def initialize_shared_memory(self):
        """Initialize shared memory for frame sharing with TouchDesigner using OSC signals and unique memory names"""
        try:
            from multiprocessing import shared_memory
            import numpy as np
            from pythonosc import udp_client
            import os
            import random
            
            # Set up OSC client for sending memory address updates to port 8060
            self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8060)
            
            # Generate unique names using timestamp and random number to avoid conflicts
            current_time = int(time.time())
            random_suffix = random.randint(1000, 9999)
            self.SHARED_MEM_NAME = f"YOLOflow_main_{current_time}_{random_suffix}"
            self.SYNC_MEM_NAME = f"YOLOflow_sync_{current_time}_{random_suffix}"
            
            # Calculate buffer sizes
            buffer_size = self.ndi_output_width * self.ndi_output_height * 4  # RGBA: 4 bytes per pixel
            sync_buffer_size = 1  # Single byte for synchronization
            
            print(f"Creating new shared memory segments with unique names:")
            print(f"- Main: {self.SHARED_MEM_NAME}")
            print(f"- Sync: {self.SYNC_MEM_NAME}")
            
            # Create shared memory segments
            self.shm = shared_memory.SharedMemory(create=True, size=buffer_size, name=self.SHARED_MEM_NAME)
            self.sync_shm = shared_memory.SharedMemory(create=True, size=sync_buffer_size, name=self.SYNC_MEM_NAME)
            
            # Create frame as numpy array mapped to shared memory
            self.shared_frame = np.ndarray(
                (self.ndi_output_height, self.ndi_output_width, 4),
                dtype=np.uint8,
                buffer=self.shm.buf
            )
            
            # Initialize sync byte to 0 (no frame ready)
            self.sync_shm.buf[0] = 0
            
            # Write config file for TouchDesigner to use
            config_path = os.path.expanduser("~/Documents/yoloflow_shared_memory.txt")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(f"main_name={self.SHARED_MEM_NAME}\n")
                f.write(f"sync_name={self.SYNC_MEM_NAME}\n")
                f.write(f"width={self.ndi_output_width}\n")
                f.write(f"height={self.ndi_output_height}\n")
                f.write(f"channels=4\n")
            
            # Reset frame count and timing
            if hasattr(self, 'frame_count'):
                delattr(self, 'frame_count')
            if hasattr(self, 'start_time'):
                delattr(self, 'start_time')
            
            # Mark as initialized
            self.shm_sender = True
            
            # Send initial OSC message with memory info to TouchDesigner on port 8060
            self.osc_client.send_message("/yoloflow/memory", [
                self.SHARED_MEM_NAME,
                self.SYNC_MEM_NAME,
                int(self.ndi_output_width),
                int(self.ndi_output_height),
                4  # RGBA channels
            ])
            
            print("\n*** SHARED MEMORY WITH OSC SIGNALING INITIALIZED ***")
            print(f"Main memory name: {self.SHARED_MEM_NAME}")
            print(f"Sync memory name: {self.SYNC_MEM_NAME}")
            print(f"Video format: {self.ndi_output_width}x{self.ndi_output_height} RGBA")
            print(f"OSC messages sent to: 127.0.0.1:8060")
            print(f"OSC address: /yoloflow/memory")
            print(f"OSC message format: [main_name, sync_name, width, height, channels]")
            print(f"Config file written to: {config_path}")
            print("*******************************************\n")
            return True
            
        except Exception as e:
            print(f"Error initializing shared memory: {str(e)}")
            import traceback
            traceback.print_exc()
            self.shm_sender = None
            return False
    
    # Removed toggle_ndi_output method as shared memory is always enabled
    
    def send_frame_to_shared_memory(self, frame):
        """Send a frame using shared memory with sync buffer for TouchDesigner"""
        # Skip if not initialized or not properly set up
        if self.shm_sender is None or not hasattr(self, 'shm') or not hasattr(self, 'shared_frame'):
            return
        
        try:
            import numpy as np
            
            # Only log once
            if not hasattr(self, '_shm_log_done'):
                self._shm_log_done = True
                print("\nShared memory frame sending started with sync buffer")
                print(f"Shape: {frame.shape} -> {self.ndi_output_width}x{self.ndi_output_height}")
            
            # Check if receiver has consumed the previous frame (sync_flag == 0)
            if hasattr(self, 'sync_shm') and self.sync_shm.buf[0] != 0:
                # Receiver hasn't read the previous frame yet, skip this frame
                return
                
            # 1. Resize frame if needed
            if frame.shape[1] != self.ndi_output_width or frame.shape[0] != self.ndi_output_height:
                frame = cv2.resize(frame, (self.ndi_output_width, self.ndi_output_height))
            
            # 2. Convert to RGBA (TouchDesigner prefers RGBA)
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            
            # 3. Copy the frame data to shared memory
            self.shared_frame[:] = frame_rgba[:]
            
            # 4. Set sync flag to 1 to indicate new frame is available
            if hasattr(self, 'sync_shm'):
                self.sync_shm.buf[0] = 1
            
            # Log FPS occasionally
            if not hasattr(self, 'frame_count') or not hasattr(self, 'start_time'):
                self.frame_count = 0
                self.start_time = time.time()
            
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            # Log FPS every 100 frames
            if self.frame_count % 100 == 0:
                fps = self.frame_count / elapsed
                print(f"Shared memory sending at {fps:.1f} FPS")
        
        except Exception as e:
            if not hasattr(self, '_shm_error_shown'):
                self._shm_error_shown = True
                print(f"Shared memory error: {str(e)}")
                import traceback
                traceback.print_exc()
                # Try to reinitialize shared memory with a new name
                print("Attempting to reinitialize shared memory with a new name...")
                self.cleanup_shared_memory()
                QTimer.singleShot(1000, self.initialize_shared_memory)  # Try again after 1 second
                return
    
    def cleanup_shared_memory(self):
        """Clean up shared memory resources"""
        if self.shm_sender is not None:
            try:
                print("Cleaning up shared memory resources...")
                
                # Store names for logging
                main_name = getattr(self, 'SHARED_MEM_NAME', "unknown")
                sync_name = getattr(self, 'SYNC_MEM_NAME', "unknown")
                
                # Clean up main shared memory
                if hasattr(self, 'shm') and self.shm is not None:
                    try:
                        # Close the shared memory
                        self.shm.close()
                        # Unlink to remove from system
                        self.shm.unlink()
                        print(f"- Main shared memory {main_name} unlinked")
                    except Exception as e:
                        print(f"- Warning: Error cleaning up main shared memory: {str(e)}")
                    self.shm = None
                
                # Clean up sync shared memory
                if hasattr(self, 'sync_shm') and self.sync_shm is not None:
                    try:
                        # Close the sync shared memory
                        self.sync_shm.close()
                        # Unlink to remove from system
                        self.sync_shm.unlink()
                        print(f"- Sync shared memory {sync_name} unlinked")
                    except Exception as e:
                        print(f"- Warning: Error cleaning up sync shared memory: {str(e)}")
                    self.sync_shm = None
                
                # Clear shared frame reference
                if hasattr(self, 'shared_frame'):
                    self.shared_frame = None
                
                # Clear FPS tracking stats
                if hasattr(self, 'frame_count'):
                    delattr(self, 'frame_count')
                if hasattr(self, 'start_time'):
                    delattr(self, 'start_time')
                if hasattr(self, '_shm_log_done'):
                    delattr(self, '_shm_log_done')
                if hasattr(self, '_shm_error_shown'):
                    delattr(self, '_shm_error_shown')
                
                # Clean up OSC socket if needed
                if hasattr(self, 'osc_client'):
                    if hasattr(self.osc_client, '_sock') and self.osc_client._sock is not None:
                        try:
                            self.osc_client._sock.close()
                            print("- OSC socket closed")
                        except:
                            pass
                
                # Set to None
                self.shm_sender = None
                print("Shared memory resources successfully released")
                
            except Exception as e:
                print(f"Error cleaning up shared memory: {str(e)}")
                import traceback
                traceback.print_exc()
                # Ensure sender is set to None even if cleanup fails
                self.shm_sender = None
                self.shm = None
                self.sync_shm = None
                self.shared_frame = None
    
    def stop_camera(self):
        """Stop the camera"""
        if self.cap is not None:
            print("TEST TAB: Stopping camera")
            self.timer.stop()
            self.cap.release()
            self.cap = None
            
            # Status text and styling removed
    
    def start_inference(self):
        """Start model inference on the video feed"""
        if self.cap is None or not self.cap.isOpened():
            # self.status_label.setText("Status: Waiting for video feed")
            return
            
        if self.model is None:
            # self.status_label.setText("Status: Please load a model first")
            return
        
        # Always clean up and recreate shared memory for each inference session
        # to avoid leaving orphaned memory segments
        if self.shm_sender is not None:
            # Clean up existing shared memory before creating new one
            self.cleanup_shared_memory()
        
        # Create new shared memory segments with unique names
        try:
            if self.initialize_shared_memory():
                print("New shared memory segments initialized successfully")
                # self.status_label.setText("Status: Shared memory initialized, starting inference...")
            else:
                print("Failed to initialize shared memory - will continue without TouchDesigner output")
                # self.status_label.setText("Status: Shared memory failed, starting inference...")
        except Exception as e:
            print(f"Error initializing shared memory: {str(e)}")
            # self.status_label.setText("Status: Shared memory error, starting inference...")
            # Continue even if shared memory fails
                
        self.inference_active = True
        self.raw_passthrough = False      # block raw frames
        
        # Update UI controls
        self.start_inference_btn.setEnabled(False)
        self.stop_inference_btn.setEnabled(True)
        
        # Disable model selection while inference is running
        self.model_dropdown.setEnabled(False)
        self.refresh_models_btn.setEnabled(False)
        self.browse_model_btn.setEnabled(False)
        self.conf_slider.setEnabled(True)  # Keep these enabled to adjust on the fly
        self.iou_slider.setEnabled(True)
        
        # Status messages removed
        print("Inference started - sending frames to TouchDesigner via shared memory")
    
    def stop_inference(self):
        """Stop model inference but keep the video feed running"""
        self.inference_active = False
        self.raw_passthrough = True       # allow raw frames again
        
        # Clean up shared memory resources when stopping inference
        if self.shm_sender is not None:
            self.cleanup_shared_memory()
            print("Shared memory resources released during inference stop")
        
        # Update UI controls
        self.start_inference_btn.setEnabled(True)
        self.stop_inference_btn.setEnabled(False)
        
        # Re-enable model selection when inference is stopped
        self.model_dropdown.setEnabled(True)
        self.refresh_models_btn.setEnabled(True)
        self.browse_model_btn.setEnabled(True)
        
        # Status messages removed
    
    def update_frame(self):
        """Update the camera frame with model predictions"""
        # IMPROVED FRAME PROCESSING STRATEGY
        # Use a flag-based approach with timeouts to avoid deadlocks
        current_time = time.time()
        
        # Check if previous frame processing is taking too long (more than 100ms)
        # This prevents the UI from freezing if processing gets stuck
        if hasattr(self, '_processing_start_time') and hasattr(self, '_processing_frame'):
            if self._processing_frame and (current_time - self._processing_start_time) > 0.1:
                # Force reset if processing is taking too long
                print("WARNING: Frame processing timed out - resetting flags")
                self._processing_frame = False
            
        # Balanced frame skipping - skip if still processing previous frame
        if hasattr(self, '_processing_frame') and self._processing_frame:
            return
            
        # Set processing flags with timeout protection
        self._processing_frame = True
        self._processing_start_time = current_time
        
        # Handle potential camera issues with better error recovery
        if self.cap is None:
            self._processing_frame = False
            QTimer.singleShot(500, self.start_camera)
            return
            
        if not self.cap.isOpened():
            self._processing_frame = False
            QTimer.singleShot(500, self.start_camera)
            return
        
        # Get the next frame with error handling
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._processing_frame = False
                QTimer.singleShot(500, self.start_camera)
                return
        except Exception as e:
            print(f"Error reading NDI frame: {e}")
            self._processing_frame = False
            QTimer.singleShot(500, self.start_camera)
            return
        
        # Optimized FPS calculation - keep it simple with minimal overhead
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            # Use a larger smoothing factor (0.2) for more responsive FPS updates
            # while still maintaining stability
            instantaneous_fps = 1.0 / max(0.001, current_time - self.last_frame_time)
            
            # Initialize smoothed FPS if needed
            if not hasattr(self, 'smoothed_fps'):
                self.smoothed_fps = instantaneous_fps
            else:
                # Update using exponential moving average (EMA)
                alpha = 0.2  # Slightly higher smoothing factor for more responsive display
                self.smoothed_fps = alpha * instantaneous_fps + (1-alpha) * self.smoothed_fps
            
            fps = self.smoothed_fps
        else:
            fps = 30.0  # Default for first frame
            self.smoothed_fps = fps
            
        self.last_frame_time = current_time
        
        # ADVANCED FRAME SKIPPING: Use an adaptive approach based on detected speed
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        
        self.frame_count += 1
        
        # IMPORTANT: Use source FPS estimate from NDI manager if available
        source_fps = 30.0  # Default assumption
        
        # Better frame skipping strategy based on source frame rate
        process_this_frame = False
        
        # FIXED: Process more frames to ensure smoother inference
        # Process every other frame (15 FPS effective at 30 FPS source)
        # This provides a better balance between performance and responsiveness
        if self.frame_count % 2 == 0:
            process_this_frame = True
            
        # Cache fps for display
        self.last_fps = fps
        
        # FIXED: Always make a copy for display to ensure inference results appear
        # We need to ensure display_frame is always a separate copy to prevent
        # modification of the original frame which could cause threading issues
        display_frame = frame.copy()
        
        # Run inference if active and model is loaded
        detected_boxes = []
        if self.inference_active and self.model is not None and process_this_frame:
            try:
                # Run inference with optimizations
                import torch  # Import here to avoid potential issues
                
                # CRITICAL FIX: Ensure we always use a fresh copy with 3 BGR channels for inference
                # Debug the frame format extensively to diagnose issues
                print(f"DEBUG: Input frame shape: {frame.shape}")
                
                # GUARANTEE 3-channel BGR for YOLO by making an explicit copy and conversion
                # Don't try to reuse the frame directly - create a guaranteed fresh copy
                if frame is not None:
                    # Always make a deep copy to avoid any reference issues
                    frame_copy = frame.copy()
                    
                    # Handle 4-channel RGBA input
                    if len(frame_copy.shape) == 3 and frame_copy.shape[2] == 4:
                        # Explicit new allocation for inference frame
                        inference_frame = np.empty((frame_copy.shape[0], frame_copy.shape[1], 3), dtype=np.uint8)
                        cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR, dst=inference_frame)
                        print("DEBUG: Converted 4-channel RGBA to 3-channel BGR for inference")
                    elif len(frame_copy.shape) == 3 and frame_copy.shape[2] == 3:
                        # Already 3 channels, but ensure it's BGR
                        inference_frame = frame_copy  # Should be BGR already
                        print("DEBUG: Using existing 3-channel frame for inference")
                    else:
                        # Unexpected format
                        print(f"ERROR: Unexpected frame format for inference: {frame_copy.shape}")
                        return
                    
                    # Final check - MUST be 3 channels
                    if len(inference_frame.shape) != 3 or inference_frame.shape[2] != 3:
                        print(f"FATAL: Failed to get 3-channel frame for inference: {inference_frame.shape}")
                        return
                else:
                    # Invalid frame, can't process
                    print("ERROR: Frame is None, cannot process")
                    return
                
                # OPTIMIZATION: Use dynamic batch size based on available memory
                # This helps balance between speed and stability
                results = self.model.predict(
                    inference_frame,  # Use properly converted frame 
                    conf=self.conf_threshold, 
                    iou=self.iou_threshold,
                    verbose=False,
                    half=True,  # Use half precision (FP16) for faster inference
                    device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
                )
                
                # Process results
                if results and len(results) > 0:
                    # Get the first result (only one image was processed)
                    result = results[0]
                    
                    # Get boxes and process them
                    boxes = result.boxes.cpu().numpy()
                    detected_boxes = boxes
                    
                    # Predefined color palette for better visibility (moved outside loop for efficiency)
                    color_palette = [
                        (0, 255, 0),    # Green
                        (255, 0, 0),    # Blue (BGR format)
                        (0, 0, 255),    # Red
                        (0, 255, 255),  # Yellow
                        (255, 0, 255),  # Magenta
                        (255, 255, 0),  # Cyan
                        (128, 0, 255),  # Purple
                        (0, 128, 255),  # Orange
                        (128, 255, 0),  # Lime
                        (0, 255, 128)   # Teal
                    ]
                    
                    # cache image size once -------------------------
                    h, w = inference_frame.shape[:2]

                    # Use enumerate to get the index for each detection
                    for idx, box in enumerate(boxes):
                        # -------- YOLO raw outputs -----------------
                        x1, y1, x2, y2 = box.xyxy[0]          # ndarray(4,) of float32
                        conf           = box.conf[0]
                        cls_id         = int(box.cls[0])      # already python int

                        # -------- cast to python scalars ----------
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        conf           = float(conf)

                        u_min = float(x1) / w                 # normalised
                        v_min = float(y1) / h
                        u_max = float(x2) / w
                        v_max = float(y2) / h

                        # Get class name if available
                        if cls_id < len(self.class_names):
                            cls_name = self.class_names[cls_id]
                        else:
                            cls_name = f"Class {cls_id}"
                            
                        # Send OSC message for this detection with idx
                        addr = f"/yoloflow/detections/{idx}"
                        self.osc_det_client.send_message(
                            addr,
                            [cls_name, float(conf),
                             int(x1), int(y1), int(x2), int(y2),
                             float(u_min), float(v_min), float(u_max), float(v_max)]
                        )
                        
                        # ───────── draw UI (unchanged) ────
                        # Select color from palette
                        color = color_palette[cls_id % len(color_palette)]
                        
                        # Simpler bounding box drawing for performance
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label if enabled
                        if self.show_labels_checkbox.isChecked():
                            label = f"{cls_name}: {conf:.2f}"
                            # Calculate text size for background
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            text_width = text_size[0] + 10
                            text_height = text_size[1] + 10
                            
                            # Draw label background
                            cv2.rectangle(
                                display_frame, 
                                (x1, y1 - text_height - 5), 
                                (x1 + text_width, y1), 
                                color, 
                                -1
                            )
                            
                            # Draw white text for better visibility
                            cv2.putText(
                                display_frame, 
                                label, 
                                (x1 + 5, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, 
                                (255, 255, 255), 
                                2
                            )
                    
                    # Send the total number of detections for this frame
                    self.osc_det_client.send_message("/yoloflow/detections/count", [int(len(boxes))])
                    
                    # tell the receiver the pack is complete (keep as is)
                    self.osc_det_client.send_message("/yoloflow/frame_end", [int(self.frame_count)])
                    
                    # self.status_label.setText(f"Status: Inference active - {len(boxes)} objects detected")
            except Exception as e:
                print(f"TEST TAB: Error during inference: {e}")
                # self.status_label.setText(f"Status: Inference error: {str(e)}")
        
        # ------------------------------------------------------------------
        # 3.  Decide what we will output (display + shared memory)
        # ------------------------------------------------------------------
        if self.inference_active:
            if process_this_frame:
                # =======================
                #  Run YOLO + draw boxes
                # =======================
                annotated = display_frame     # keep your existing drawing code
                self.last_annotated = annotated.copy()
                out_frame = annotated
            else:
                # Skip inference on this tick; re‑use last annotated frame
                out_frame = self.last_annotated if self.last_annotated is not None else frame
        else:
            # Not running inference → always show/pass‑through raw frame
            out_frame = frame

        # ---------------------------------
        # 4.  Overlays common to all modes
        # ---------------------------------
        if self.inference_active:
            cv2.putText(out_frame, f"FPS: {fps:.1f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # ---------------------------------
        # 5.  Publish to TouchDesigner + UI
        # ---------------------------------
        self.send_frame_to_shared_memory(out_frame)
        self._update_qt_view(out_frame)     # moved Qt stuff into a small helper
        self._processing_frame = False
    
    def _update_qt_view(self, bgr):
        if not hasattr(self, '_rgb_buffer') or self._rgb_buffer.shape[:2] != bgr.shape[:2]:
            self._rgb_buffer = np.empty_like(bgr)
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=self._rgb_buffer)
        h, w = bgr.shape[:2]
        img = QImage(self._rgb_buffer.data, w, h, 3*w, QImage.Format_RGB888)
        self.camera_view.setPixmap(QPixmap.fromImage(img).scaled(
            self.camera_view.size(), Qt.KeepAspectRatio))