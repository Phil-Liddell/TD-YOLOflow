import os
import cv2
import numpy as np
import logging
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QLineEdit, QFormLayout, QGroupBox, QMessageBox,
                             QCheckBox, QSlider, QToolButton, QFrame, QSizePolicy, QScrollArea,
                             QGridLayout, QStackedLayout, QApplication)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent
from .utils import (VideoCapture, FrameProcessor, NDICapture, 
                   create_folder_if_not_exists, is_ndi_available, get_ndi_source_name, load_config)
from .dino_point_core import DinoCore

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

class CollapsibleBox(QWidget):
    """Custom collapsible box widget for expandable sections"""
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)
        
        self.toggle_animation = None
        
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)
        
        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)
        
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content_area.setMaximumHeight(self.content_area.widget().sizeHint().height() if checked else 0)
        
    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
        content = QWidget()
        content.setLayout(layout)
        self.content_area.setWidget(content)
        self.content_area.setWidgetResizable(True)
        
    def toggleExpand(self, expand):
        self.toggle_button.setChecked(expand)
        self.on_pressed()

class CaptureTab(QWidget):
    data_updated = pyqtSignal(str)  # Signal emitted when new data captured
    
    def __init__(self):
        super().__init__()
        
        init_start = time.time()
        logging.info("Initializing CaptureTab")
        
        # Initialize video capture (lazy initialization)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Track if we're using NDI
        self.using_ndi = False
        
        # Load config first
        self.config = load_config()
        
        # Check if NDI is available and enabled in config
        self.ndi_available = is_ndi_available()
        self.prefer_ndi = self.config.get('ndi', {}).get('prefer_ndi_over_camera', True)
        
        # Initialize basic tracker settings
        logging.info("Setting up tracker configuration")
        self.dino_tracker = DinoCore()    # Main tracker - default params
        self.anchor_pt = None             # (cx, cy) in full‑res coords
        
        # Setup tracking flags
        self.is_tracking = False
        self.is_recording = False
        self.frame_count = 0
        self.saved_frame_count = 0
        self.recording_fps = 5  # Default recording FPS
        
        # Drawing and dragging variables
        self.drawing = False       # For initial drawing
        self.draw_mode = False     # Drawing mode active
        self.dragging = False      # For dragging existing box
        self.drag_mode = False     # Dragging mode active
        self.drag_offset_x = 0     # For calculating drag position
        self.drag_offset_y = 0
        self.start_point = None
        self.end_point = None
        self.current_frame = None
        self.current_box = None    # Current bounding box coordinates
        
        # Output paths - set this before everything else
        logging.info("Setting up output directories")
        self.base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
        logging.info(f"Base output directory: {self.base_output_dir}")
        create_folder_if_not_exists(self.base_output_dir)
        
        # Use data directory directly for images and labels
        self.images_dir = os.path.join(self.base_output_dir, "images")
        self.labels_dir = os.path.join(self.base_output_dir, "labels")
        create_folder_if_not_exists(self.images_dir)
        create_folder_if_not_exists(self.labels_dir)
        
        logging.info(f"Directories setup: Images={os.path.exists(self.images_dir)}, Labels={os.path.exists(self.labels_dir)}")
        
        # Initialize class-related attributes BEFORE setting up UI
        self.current_class = ""  # Will be set when selected or drawing starts
        self.class_list = ["Banana", "Raspberry Pi", "Headphones"]
        
        # Load existing classes from file if available
        classes_file = os.path.join(self.base_output_dir, "classes.txt")
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r') as f:
                    saved_classes = [line.strip() for line in f.readlines() if line.strip()]
                if saved_classes:
                    self.class_list = saved_classes
                    logging.info(f"Loaded {len(saved_classes)} classes from {classes_file}")
            except Exception as e:
                logging.error(f"Error loading classes file: {str(e)}")
        
        # Track camera devices - will be populated on demand        
        self.available_cameras = []
        
        # Setup UI AFTER initializing all variables
        logging.info("Setting up UI components")
        ui_start = time.time()
        self.setup_ui()
        logging.info(f"UI setup completed in {time.time() - ui_start:.3f}s")
        
        logging.info(f"CaptureTab initialization complete ({time.time() - init_start:.3f}s)")
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        
        # Camera view with overlay container for controls
        camera_container = QWidget()
        camera_container.setContentsMargins(8, 8, 8, 8)
        camera_container_layout = QVBoxLayout(camera_container)
        camera_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a proper stacked layout with absolute positioning
        camera_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container has relative positioning
        relative_container = QWidget()
        relative_container.setStyleSheet("border: 1px solid #ccc; background-color: transparent;")
        relative_layout = QVBoxLayout(relative_container)
        relative_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main camera view
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(640, 800)
        self.camera_view.setStyleSheet("border: none;")
        self.camera_view.setMouseTracking(True)  # Enable mouse tracking
        
        # Connect mouse events for drawing
        self.camera_view.mousePressEvent = self.mouse_press_event
        self.camera_view.mouseMoveEvent = self.mouse_move_event
        self.camera_view.mouseReleaseEvent = self.mouse_release_event
        
        # Add camera view to container
        relative_layout.addWidget(self.camera_view)
        
        # Create the drawing button for top of camera view
        self.draw_box_btn = QPushButton("DRAW BOUNDING BOX")
        self.draw_box_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9900;  /* Bright orange for high visibility */
                color: white;
                font-weight: bold;
                border-radius: 12px;
                padding: 15px 20px;
                font-size: 20px;  /* Much larger font */
                border: 3px solid #e68a00;
                letter-spacing: 1px;
                text-transform: uppercase;
            }
            QPushButton:hover {
                background-color: #ffad33;
                border: 3px solid #e68a00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.draw_box_btn.clicked.connect(self.toggle_draw_mode)
        
        # Add button to an event filter for top of camera view
        camera_container_layout.addWidget(relative_container)
        
        # Now place the button on top using geometry
        self.draw_box_btn.setParent(relative_container)
        self.draw_box_btn.setGeometry(20, 20, 300, 100)  # x, y, width, height
        
        # Add the container to the main layout
        main_layout.addWidget(camera_container)
        
        # Controls section - using two-column layout
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(8, 10, 8, 10)
        controls_layout.setSpacing(10)
        
        # LEFT COLUMN: Camera + Class options
        left_column = QVBoxLayout()
        left_column.setSpacing(5)
        
        # 1. VIDEO INPUT SETTINGS
        input_title = "Video Input"
        camera_box = QGroupBox(input_title)
        camera_layout = QVBoxLayout(camera_box)
        camera_layout.setSpacing(5)
        camera_layout.setContentsMargins(8, 10, 8, 10)
        
        # Video source status label
        if self.ndi_available:
            video_info = QLabel(f"Video stream available")
            video_info.setStyleSheet("""
                color: #2196F3;
                font-weight: bold;
                padding: 4px;
                background-color: #e3f2fd;
                border-radius: 4px;
            """)
            camera_layout.addWidget(video_info)
        
        # Camera selection row (hidden if prefer_ndi is true)
        camera_select_layout = QHBoxLayout()
        camera_select_layout.setSpacing(5)
        camera_select_layout.addWidget(QLabel("Device:"))
        self.camera_dropdown = QComboBox()
        self.populate_camera_list()
        camera_select_layout.addWidget(self.camera_dropdown, 1)  # Give dropdown more space
        
        # Only add the camera selection if not preferring NDI or if NDI is not available
        if not (self.ndi_available and self.prefer_ndi):
            camera_layout.addLayout(camera_select_layout)
        else:
            # Hide all camera controls completely
            self.camera_dropdown.setVisible(False)
        
        # FPS selection with improved styling
        fps_layout = QHBoxLayout()
        fps_layout.setSpacing(5)
        fps_label = QLabel("Recording FPS:")
        fps_label.setStyleSheet("font-weight: bold; color: #333;")
        fps_layout.addWidget(fps_label)
        
        self.fps_dropdown = QComboBox()
        self.fps_dropdown.addItems(["1 FPS", "2 FPS", "3 FPS", "5 FPS", "10 FPS", "15 FPS", "20 FPS", "30 FPS"])
        self.fps_dropdown.setCurrentIndex(3)  # Default to 5 FPS
        self.fps_dropdown.setStyleSheet("""
            QComboBox {
                border: 1px solid #aaa;
                border-radius: 3px;
                padding: 2px 8px;
                min-width: 100px;
                background-color: #f8f8f8;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #aaa;
            }
        """)
        fps_layout.addWidget(self.fps_dropdown, 1)
        camera_layout.addLayout(fps_layout)
        
        # Camera control buttons with improved styling
        camera_buttons_layout = QHBoxLayout()
        camera_buttons_layout.setSpacing(8)
        
        # Removed NDI feed status indicator
        
        # Always start camera immediately
        QTimer.singleShot(100, self.start_camera)
        
        # Tracking options 
        tracking_box = QGroupBox("Tracking Options")
        tracking_layout = QVBoxLayout(tracking_box)
        tracking_layout.setContentsMargins(8, 8, 8, 8)
        
        # Tracking mode header
        tracking_header = QLabel("Tracking Mode:")
        tracking_header.setStyleSheet("font-weight: bold; color: #333; margin-bottom: 6px; font-size: 14px;")
        tracking_layout.addWidget(tracking_header)
        
        # ① replace the old checkbox section
        tracking_mode_layout = QHBoxLayout()
        self.track_checkbox = QCheckBox("Tracking")           # new label
        self.track_checkbox.setChecked(False)                 # default OFF
        self.track_checkbox.toggled.connect(self.toggle_tracking)
        tracking_mode_layout.addWidget(self.track_checkbox)
        tracking_layout.addLayout(tracking_mode_layout)
        
        # Parameter adjustments section header
        self.params_header = QLabel("Tracking Parameters:")
        self.params_header.setStyleSheet("font-weight: bold; color: #999; margin: 10px 0 6px 0; font-size: 14px;")  # Greyed out by default
        tracking_layout.addWidget(self.params_header)
        
        # Define slider styles - reused for all sliders
        self.slider_enabled_style = """
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3366cc;
                border: 1px solid #5c5c5c;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #3366cc;
                border-radius: 3px;
            }
        """
        
        self.slider_disabled_style = """
            QSlider::groove:horizontal {
                height: 6px;
                background: #e8e8e8;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #cccccc;
                border: 1px solid #aaaaaa;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #cccccc;
                border-radius: 3px;
            }
        """
        
        # ② Smoothing slider
        smooth_layout = QGridLayout()
        smooth_layout.setSpacing(5)

        self.smooth_label = QLabel("Smoothing:")
        self.smooth_label.setStyleSheet("color:#999;")
        smooth_layout.addWidget(self.smooth_label, 0, 0)

        self.smooth_val = QLabel("0.30")
        self.smooth_val.setAlignment(Qt.AlignRight)
        self.smooth_val.setStyleSheet("font-weight:bold; color:#999; min-width:40px;")
        smooth_layout.addWidget(self.smooth_val, 0, 1)

        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0,100)           # map 0‑1
        self.smooth_slider.setValue(30)              # 0.30 default
        self.smooth_slider.setEnabled(False)         # disabled until tracking ON
        self.smooth_slider.setStyleSheet(self.slider_disabled_style)
        self.smooth_slider.valueChanged.connect(self.update_smoothing)
        smooth_layout.addWidget(self.smooth_slider, 0, 2)

        tracking_layout.addLayout(smooth_layout)
        
        # Add tracking box to left column
        left_column.addWidget(camera_box)
        left_column.addWidget(tracking_box)
        
        # --------------------------------------------------------------------
        # --------------------------------------------------------------------
        # 2. CLASS SETTINGS  (⇦ drop this whole block in place of your grid code)
        # --------------------------------------------------------------------
        class_box = QGroupBox("Class")
        class_layout = QVBoxLayout(class_box)
        class_layout.setContentsMargins(8, 10, 8, 10)
        class_layout.setSpacing(6)

        # ── row 1 : "Current" ────────────────────────────────────────────────
        current_row = QHBoxLayout()
        current_row.setSpacing(6)

        lbl_current = QLabel("Current Class:")
        lbl_current.setFixedWidth(140)                # Make even wider
        current_row.addWidget(lbl_current)

        self.class_dropdown = QComboBox()
        self.class_dropdown.addItems(self.class_list)
        self.class_dropdown.setMinimumWidth(150)   # Make dropdown wider
        self.class_dropdown.setMaximumWidth(200)  # But not too wide
        self.class_dropdown.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # FIXED SIZE - NEVER EXPAND
        current_row.addWidget(self.class_dropdown) # NO STRETCH FACTOR
        
        # Add a spacer to push the button to the right
        current_row.addStretch(1)

        self.remove_class_btn = QPushButton("Remove")
        self.remove_class_btn.setFixedWidth(180)      # Make button wider
        self.remove_class_btn.setToolTip("Remove selected class from the list")
        self.remove_class_btn.clicked.connect(self.remove_selected_class)
        self.remove_class_btn.setStyleSheet("""
            QPushButton {
                background-color: #777777;
                color: white;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        current_row.addWidget(self.remove_class_btn)
        class_layout.addLayout(current_row)

        # ── row 2 : "Add new" ────────────────────────────────────────────────
        add_row = QHBoxLayout()
        add_row.setSpacing(6)

        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("Add new class…")
        self.new_class_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        add_row.addWidget(self.new_class_input, 1)
        
        # Add a spacer to push the button to the right
        add_row.addStretch(1)

        self.add_class_btn = QPushButton("Add Class")
        self.add_class_btn.setFixedWidth(180)  # Make button wider
        self.add_class_btn.clicked.connect(self.add_new_class)
        # FORCE VISIBLE TEXT WITH HIGH CONTRAST
        self.add_class_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        add_row.addWidget(self.add_class_btn)
        class_layout.addLayout(add_row)
        
        # Add spacing
        spacer = QWidget()
        spacer.setMinimumHeight(5)
        class_layout.addWidget(spacer)
        
        # Add class box to left column
        left_column.addWidget(class_box)
        
        # RIGHT COLUMN: Record section
        right_column = QVBoxLayout()
        right_column.setSpacing(5)
        
        # Record controls with status and button
        record_box = QGroupBox("Record")
        record_layout = QVBoxLayout(record_box)
        record_layout.setSpacing(8)
        record_layout.setContentsMargins(10, 12, 10, 12)
        
        # Status display with improved style
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)                      # Wrap text instead of stretching
        self.status_label.setSizePolicy(QSizePolicy.Expanding, 
                                       QSizePolicy.Fixed)         # Never force width growth
        self.status_label.setMinimumWidth(260)                    # Set minimum width
        self.status_label.setMaximumWidth(320)                    # Set maximum width to match record button
        self.status_label.setStyleSheet("""
            font-weight: bold; 
            font-size: 11pt;
            color: #333333;
            background-color: #f8f8f8;
            border-radius: 4px;
            padding: 4px;
            border: 1px solid #e0e0e0;
        """)
        record_layout.addWidget(self.status_label)
        
        # Add a bit of spacing
        record_layout.addSpacing(5)
        
        # Extremely large record button with improved styling
        self.record_btn = QPushButton("RECORD")
        self.record_btn.setCheckable(True)
        self.record_btn.setMinimumHeight(320)  # Twice as tall (2x 160)
        self.record_btn.setMinimumWidth(320)   # Square
        self.record_btn.setStyleSheet("""
            QPushButton {
                font-size: 38px;
                font-weight: bold;
                background-color: #f0f0f0;
                border: 5px solid #aaa;
                border-radius: 20px;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border: 5px solid #999;
            }
            QPushButton:checked {
                background-color: #ff4040;
                color: white;
                border: 5px solid #cc0000;
                /* removed text-shadow for better compatibility */
            }
            QPushButton:disabled {
                background-color: #e0e0e0;
                color: #a0a0a0;
                border: 5px solid #cccccc;
            }
        """)
        # Disable record button initially until box is drawn
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_btn)
        
        # Add record box to right column
        right_column.addWidget(record_box)
        
        # Add columns to main control layout - use 2:1 ratio
        controls_layout.addLayout(left_column, 2)  # Left gets 2/3 width
        controls_layout.addLayout(right_column, 1)  # Right gets 1/3 width
        
        # Set up separator line above controls
        top_separator = QFrame()
        top_separator.setFrameShape(QFrame.HLine)
        top_separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(top_separator)
        main_layout.addSpacing(5)
        
        # Add the controls container to the main layout
        main_layout.addWidget(controls_container)
        
        
        # Set the main layout
        self.setLayout(main_layout)
        
        # Update the class dropdown with available classes
        self.update_class_dropdown()
        
    def populate_camera_list(self):
        logging.info("Populating camera list")
        scan_start = time.time()
        
        self.camera_dropdown.clear()
        
        # Skip camera scanning completely if NDI is available and preferred
        if self.ndi_available and self.prefer_ndi:
            self.camera_dropdown.addItem("Using Video Stream")
            logging.info("Using NDI - skipping camera scan")
            return
        
        # Use cached camera list if available
        if self.available_cameras:
            logging.info(f"Using cached camera list with {len(self.available_cameras)} devices")
            self.camera_dropdown.addItems(self.available_cameras)
            return
        
        # First add a placeholder while scanning
        self.camera_dropdown.addItem("Scanning...")
        QApplication.processEvents()  # Update UI
        
        # Check for available cameras (0-3 for faster startup, can be expanded)
        # Limiting to first few indices for faster startup
        available_cameras = []
        for i in range(4):  # Reduced from 10 to 4 for faster startup
            try:
                # Use thread-safe VideoCapture with short timeout
                cap = cv2.VideoCapture(i)
                # Only wait a short time to check if camera opens
                if cap.isOpened():
                    available_cameras.append(str(i))
                cap.release()
            except Exception as e:
                logging.warning(f"Error checking camera {i}: {str(e)}")
        
        # Update UI with results
        self.camera_dropdown.clear()
        if available_cameras:
            self.available_cameras = available_cameras
            self.camera_dropdown.addItems(available_cameras)
            logging.info(f"Found {len(available_cameras)} cameras in {time.time() - scan_start:.3f}s")
        else:
            self.camera_dropdown.addItem("No cameras found")
            logging.info("No cameras found")
    
    def update_class_dropdown(self):
        """Update the class dropdown with current class list"""
        current_text = self.class_dropdown.currentText()
        self.class_dropdown.clear()
        self.class_dropdown.addItems(self.class_list)
        
        # Try to restore the previous selection if it still exists
        if current_text in self.class_list:
            self.class_dropdown.setCurrentText(current_text)
        elif self.class_list:
            # Set to first item if available
            self.class_dropdown.setCurrentIndex(0)
    
    def add_new_class(self):
        """Add a new class to the list"""
        new_class = self.new_class_input.text().strip()
        if new_class and new_class not in self.class_list:
            self.class_list.append(new_class)
            self.update_class_dropdown()
            self.class_dropdown.setCurrentText(new_class)
            self.new_class_input.clear()
            
            # Save class list to file
            self.save_classes_to_file()
        elif not new_class:
            self.status_label.setText("Status: Please enter a class name")
        else:
            self.status_label.setText(f"Status: Class '{new_class}' already exists")
    
    def remove_selected_class(self):
        """Remove the currently selected class from the list"""
        current_class = self.class_dropdown.currentText()
        if not current_class:
            self.status_label.setText("Status: No class selected to remove")
            return
            
        if len(self.class_list) <= 1:
            self.status_label.setText("Status: Cannot remove the last class")
            return
            
        if current_class in self.class_list:
            self.class_list.remove(current_class)
            self.update_class_dropdown()
            self.status_label.setText(f"Status: Removed class '{current_class}'")
            
            # Save updated class list to file
            self.save_classes_to_file()
        else:
            self.status_label.setText(f"Status: Class '{current_class}' not found")
    
    def save_classes_to_file(self):
        """Save the current class list to classes.txt"""
        try:
            classes_file = os.path.join(self.base_output_dir, "classes.txt")
            with open(classes_file, 'w') as f:
                f.write('\n'.join(self.class_list))
            print(f"Saved classes to: {classes_file}")
        except Exception as e:
            print(f"Error saving classes file: {str(e)}")
    
    def start_camera(self):
        try:
            # Get the most current NDI availability status
            self.ndi_available = is_ndi_available()
            
            # Try the centralized NDI manager first if available
            from . import ndi_manager
            if self.ndi_available and ndi_manager.is_initialized():
                logging.info("Using centralized NDI manager for video input")
                try:
                    # Release any existing capture
                    if self.cap is not None:
                        self.cap.release()
                    
                    # Get NDI capture instance from manager
                    self.cap = ndi_manager.get_capture_instance()
                    
                    if self.cap is None:
                        raise ValueError("NDI manager returned None capture instance")
                        
                    # Get configured details
                    ndi_source_name = get_ndi_source_name()
                    width = self.config.get('ndi', {}).get('width', 1280)
                    height = self.config.get('ndi', {}).get('height', 720)
                    
                    self.using_ndi = True
                    
                    # Set viewport timer to 60 FPS (16ms interval) for smooth display with NDI
                    viewport_fps = 60
                    interval = 1000 // viewport_fps
                    
                    # Store FPS for recording calculations
                    self.recording_fps = self.config.get('ndi', {}).get('fps', 30)
                    
                    # Hide camera selection while using NDI
                    self.camera_dropdown.setVisible(False)
                    
                    # Show appropriate status based on whether we're connected to NDI
                    if hasattr(self.cap, 'connected') and self.cap.connected:
                        self.status_label.setText(f"Status: Video feed connected")
                        self.status_label.setStyleSheet("""
                            font-weight: bold; 
                            font-size: 11pt;
                            color: #333333;
                            background-color: #e3f2fd;
                            border-radius: 4px;
                            padding: 4px;
                            border: 1px solid #2196F3;
                        """)
                    else:
                        self.status_label.setText(f"Status: Video feed connected")
                        self.status_label.setStyleSheet("""
                            font-weight: bold; 
                            font-size: 11pt;
                            color: #333333;
                            background-color: #fff3cd;
                            border-radius: 4px;
                            padding: 4px;
                            border: 1px solid #ffc107;
                        """)
                    
                    # Start the timer
                    self.timer.start(interval)
                    
                    # After initializing NDI, return early (don't continue to camera initialization)
                    return
                except Exception as ndi_error:
                    logging.error(f"Failed to initialize from NDI manager: {str(ndi_error)}")
                    self.using_ndi = False
                    # Continue to try regular camera capture as fallback
                    
                    # Show camera controls since NDI failed, but only if we have cameras
                    if self.available_cameras:
                        # Repopulate camera list first to make sure we have valid entries
                        self.available_cameras = []  # Clear cache to force rescan
                        self.populate_camera_list()
                        self.camera_dropdown.setVisible(True)
            
            # Regular camera capture (if NDI is unavailable or failed)
            try:
                camera_id = int(self.camera_dropdown.currentText())
            except (ValueError, AttributeError):
                # Default to camera 0 if conversion fails
                camera_id = 0
                
            if self.cap is not None:
                self.cap.release()
            
            # Use our thread-safe VideoCapture class
            self.cap = VideoCapture(camera_id)
            if not self.cap.isOpened():
                self.status_label.setText("Status: Failed to open camera")
                self.status_label.setStyleSheet("""
                    font-weight: bold; 
                    font-size: 11pt;
                    color: white;
                    background-color: #f44336;
                    border-radius: 4px;
                    padding: 4px;
                    border: 1px solid #d32f2f;
                """)
                return
                
            # Show camera controls since we're using a camera
            self.camera_dropdown.setVisible(True)
                
            # Set viewport timer to 30 FPS (33ms interval) for smooth display
            # regardless of selected recording FPS
            viewport_fps = 30
            interval = 1000 // viewport_fps
            
            # Store selected recording FPS for later use when saving frames
            fps_text = self.fps_dropdown.currentText().split()[0]  # Get only the number
            try:
                self.recording_fps = int(fps_text)
            except ValueError:
                # Default to 5 FPS if there's an error
                self.recording_fps = 5
            
            # Update UI to show we're using the camera
            self.using_ndi = False
            
            self.timer.start(interval)
            self.status_label.setText(f"Status: Camera running at {viewport_fps} FPS (Recording at {self.recording_fps} FPS)")
            self.status_label.setStyleSheet("""
                font-weight: bold; 
                font-size: 11pt;
                color: #333333;
                background-color: #f8f8f8;
                border-radius: 4px;
                padding: 4px;
                border: 1px solid #e0e0e0;
            """)
        except Exception as e:
            self.status_label.setText(f"Status: Error starting video input - {str(e)}")
            logging.error(f"Error starting video input: {str(e)}")
    
    def stop_camera(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.status_label.setText("Status: Camera stopped")
            # Reset tracking and drawing states
            self.is_tracking = False
            self.is_recording = False
            self.draw_mode = False
            self.drawing = False
            self.start_point = None
            self.end_point = None
            if self.record_btn.isChecked():
                self.record_btn.setChecked(False)
    
    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return
            
        # Store the current frame for drawing or processing
        self.current_frame = frame.copy()
        
        # If we're tracking, handle the frame
        if self.is_tracking:
            processed_frame = frame.copy()
            
            # Process based on tracking checkbox state
            if self.track_checkbox.isChecked():          # auto‑track
                if self.anchor_pt is not None and self.current_box is not None:
                    cx, cy = self.dino_tracker.track(frame)
                    # convert to full‑res coords
                    cx = int(cx * frame.shape[1] / self.dino_tracker.side)
                    cy = int(cy * frame.shape[0] / self.dino_tracker.side)
                    self.anchor_pt = (cx, cy)
                    
                    # Update the box around the new center point
                    x, y, w, h = self.current_box
                    center_x_old = x + w // 2
                    center_y_old = y + h // 2
                    
                    # Calculate the shift from old center to new center
                    dx = cx - center_x_old
                    dy = cy - center_y_old
                    
                    # Create a new box centered on the tracked point
                    x = x + dx
                    y = y + dy
                    self.current_box = (x, y, w, h)
                    bbox = self.current_box
            else:                                        # manual drag only
                if self.current_box is not None:
                    self.anchor_pt = (self.current_box[0]+self.current_box[2]//2,
                                    self.current_box[1]+self.current_box[3]//2)
                bbox = self.current_box
            
            # Draw bounding box
            if self.current_box:
                x, y, w, h = self.current_box
                # Draw rectangle
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw the marker at the anchor point
            if self.anchor_pt:
                cx, cy = self.anchor_pt
                # Draw center marker (crosshair)
                marker_size = 20
                cv2.drawMarker(processed_frame, (cx, cy),
                              (0, 0, 255), cv2.MARKER_CROSS, marker_size, 2)
            
            # Add recording indicator if active
            if self.is_recording:
                # Add a red recording indicator in the top-left corner
                cv2.circle(processed_frame, (30, 30), 15, (0, 0, 255), -1)
                
                # Use the recording_fps value set in start_camera()
                # We're recording at self.recording_fps but displaying at viewport_fps (30)
                
                # Calculate how many display frames to skip based on ratio of viewport FPS to recording FPS
                # If viewport is 30 FPS and recording is 5 FPS, save every 6th frame
                viewport_fps = 30  # The fixed viewport FPS we set in start_camera()
                save_every_n_frames = max(1, round(viewport_fps / self.recording_fps))
                    
                # Save frame if it's the nth frame in sequence
                # Record frames even when no bounding box present as "no class"
                if self.frame_count % save_every_n_frames == 0:
                    try:
                        # Track frames with no bounding box as "no class" for YOLO training
                        if bbox is None:
                            print("Recording frame with no class detected")
                            # Record as is but mark it as "no class" - we'll create an empty label
                            self.save_frame_no_class(frame)
                        else:
                            # Normal frame with detection
                            self.save_frame(frame, bbox)
                    except Exception as e:
                        print(f"Error in recording: {str(e)}")
                else:
                    # Add small FPS text indicator to show frames being skipped
                    cv2.putText(processed_frame, f"REC {self.recording_fps} FPS", (50, 36), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                # Increment frame counter on every displayed frame, not just saved ones
                self.frame_count += 1
        else:
            processed_frame = frame.copy()
            
        # If we're in drawing mode and have a rectangle, draw it
        if self.draw_mode and self.drawing and self.start_point and self.end_point:
            cv2.rectangle(
                processed_frame,
                (self.start_point[0], self.start_point[1]),
                (self.end_point[0], self.end_point[1]),
                (0, 255, 0),
                2
            )
        
        # Convert to Qt format for display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale while maintaining aspect ratio
        self.camera_view.setPixmap(pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatio))
    
    def draw_bounding_box(self):
        """Redirect to toggle_draw_mode for compatibility"""
        self.toggle_draw_mode()
    
    def toggle_recording(self, checked):
        # This should never happen now because we disable the button,
        # but keep as a safeguard
        if not self.is_tracking:
            self.record_btn.setChecked(False)
            self.status_label.setText("Status: Please draw a bounding box before recording")
            return
            
        try:
            print(f"Recording toggled: {checked}")
            
            # Store current camera view size before toggling recording
            current_size = self.camera_view.size()
            
            # Update recording state
            self.is_recording = checked
            
            if checked:
                # Reset frame counters when starting a new recording
                self.frame_count = 0
                self.saved_frame_count = 0
                
                # Ensure directories exist
                os.makedirs(self.images_dir, exist_ok=True)
                os.makedirs(self.labels_dir, exist_ok=True)
                
                print(f"Images directory: {self.images_dir}")
                print(f"Labels directory: {self.labels_dir}")
                
                # Use the recording_fps value set in start_camera
                if self.track_checkbox.isChecked():
                    self.status_label.setText(f"Status: Recording {self.current_class} at {self.recording_fps} FPS with auto-tracking")
                else:
                    self.status_label.setText(f"Status: Recording {self.current_class} at {self.recording_fps} FPS - You can drag the box")
                
                # Restore camera view size to avoid scaling issues when recording starts
                if self.camera_view.pixmap() is not None:
                    self.camera_view.setFixedSize(current_size)
            else:
                # Get total frames recorded
                total_frames = self.saved_frame_count
                
                # Update status
                self.status_label.setText(f"Status: Tracking {self.current_class} (Saved {total_frames} frames)")
                
                # Save class list to file
                try:
                    classes_file = os.path.join(self.base_output_dir, "classes.txt")
                    
                    with open(classes_file, 'w') as f:
                        f.write('\n'.join(self.class_list))
                    print(f"Saved classes to: {classes_file}")
                except Exception as e:
                    print(f"Error saving classes file: {str(e)}")
                
                # Signal that new data is available
                self.data_updated.emit(self.base_output_dir)
                
                # Update status to confirm saved frames
                if total_frames > 0:
                    self.status_label.setText(f"Status: Saved {total_frames} frames to data directory")
                else:
                    self.status_label.setText("Status: No frames were recorded - check console for errors")
                
                # Restore camera view size to avoid scaling issues when recording stops
                if self.camera_view.pixmap() is not None:
                    self.camera_view.setFixedSize(current_size)
        
        except Exception as e:
            print(f"Error in toggle_recording: {str(e)}")
            self.status_label.setText(f"Status: Recording error - {str(e)}")
    
    def save_frame_no_class(self, frame):
        """Save frame with no bounding box as 'no_class' for YOLO training"""
        try:
            # Generate filenames - use saved_frames count for sequential numbering
            saved_frame_count = getattr(self, 'saved_frame_count', 0)
            
            # Special naming for no class frames
            frame_id = f"no_class_{saved_frame_count:06d}"
            img_path = os.path.join(self.images_dir, f"{frame_id}.jpg")
            label_path = os.path.join(self.labels_dir, f"{frame_id}.txt")
            
            # Debug information
            print(f"Saving no-class frame to: {img_path}")
            print(f"Creating empty label at: {label_path}")
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            # Check if directories exist
            if not os.path.exists(os.path.dirname(img_path)) or not os.path.exists(os.path.dirname(label_path)):
                print(f"ERROR: Failed to create directories for no-class frame")
                return
                
            # Save the image
            result = cv2.imwrite(img_path, frame)
            if not result:
                print(f"Standard imwrite failed for {img_path}, trying alternative")
                try:
                    from PIL import Image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pil_img.save(img_path, quality=95)
                    result = True
                except Exception as pil_error:
                    print(f"PIL save failed: {str(pil_error)}")
                    return
            
            # Create an empty label file - YOLO format requires a file but it can be empty for no detections
            open(label_path, 'w').close()
            
            # Increment saved frame count
            self.saved_frame_count = saved_frame_count + 1
            
            # Update status to show no-class frame was saved
            self.status_label.setText(f"Status: Saved frame with no class (Frame: {self.saved_frame_count})")
            
        except Exception as e:
            print(f"Error saving no-class frame: {str(e)}")
            
    def save_frame(self, frame, bbox):
        try:
            # Generate filenames - use saved_frames count for sequential numbering
            # self.frame_count is incremented in update_frame for every frame processed,
            # but we only want sequential numbering for frames actually saved
            saved_frame_count = getattr(self, 'saved_frame_count', 0)
            
            frame_id = f"{self.current_class}_{saved_frame_count:06d}"
            img_path = os.path.join(self.images_dir, f"{frame_id}.jpg")
            label_path = os.path.join(self.labels_dir, f"{frame_id}.txt")
            
            # Debug information
            print(f"Saving frame to: {img_path}")
            print(f"Saving label to: {label_path}")
            
            # Create directories if they don't exist (ensure they're created)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            # Check if directories actually exist
            if not os.path.exists(os.path.dirname(img_path)):
                print(f"ERROR: Failed to create directory {os.path.dirname(img_path)}")
                return
                
            if not os.path.exists(os.path.dirname(label_path)):
                print(f"ERROR: Failed to create directory {os.path.dirname(label_path)}")
                return
            
            # Save image (try different approaches if one fails)
            result = cv2.imwrite(img_path, frame)
            if not result:
                print(f"Standard imwrite failed for {img_path}, trying alternative approach")
                # Try PIL as alternative
                try:
                    from PIL import Image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pil_img.save(img_path, quality=95)
                    print(f"Saved image using PIL to {img_path}")
                    result = True
                except Exception as pil_error:
                    print(f"PIL save failed: {str(pil_error)}")
                    return
            
            # Verify image was saved
            if not os.path.exists(img_path):
                print(f"ERROR: Image file was not created at {img_path}")
                return
            else:
                print(f"SUCCESS: Image saved to {img_path} (size: {os.path.getsize(img_path)} bytes)")
                
            # Save label in YOLO format (class_id, x_center, y_center, width, height)
            # All values normalized to [0, 1]
            h, w = frame.shape[:2]
            x, y, box_w, box_h = bbox
            x_center = (x + box_w/2) / w
            y_center = (y + box_h/2) / h
            norm_width = box_w / w
            norm_height = box_h / h
            
            # Get class ID (index in class list)
            class_id = self.class_list.index(self.current_class)
            
            # Write YOLO format label
            try:
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
                # Verify label was saved
                if not os.path.exists(label_path):
                    print(f"ERROR: Label file was not created at {label_path}")
                    return
                else:
                    print(f"SUCCESS: Label saved to {label_path} (size: {os.path.getsize(label_path)} bytes)")
            except Exception as label_error:
                print(f"Error writing label file: {str(label_error)}")
                return
                
            # Increment saved frame count
            self.saved_frame_count = saved_frame_count + 1
            
            # Get FPS for status display
            fps_text = self.fps_dropdown.currentText().split()[0]  # Get only the number
            try:
                recording_fps = int(fps_text)
            except ValueError:
                recording_fps = 5  # Default to 5 FPS
            
            # Save class list to file each time a frame is saved (ensures it's always up to date)
            try:
                classes_file = os.path.join(self.base_output_dir, "classes.txt")
                with open(classes_file, 'w') as f:
                    f.write('\n'.join(self.class_list))
                print(f"Saved classes to: {classes_file}")
            except Exception as classes_error:
                print(f"Error saving classes file: {str(classes_error)}")
            
            # Update status with current count of saved frames and FPS
            self.status_label.setText(f"Status: Recording {self.current_class} at {recording_fps} FPS (Frames: {self.saved_frame_count})")
            
        except Exception as e:
            print(f"Error saving frame: {str(e)}")
    
    def map_to_image_coords(self, event_pos):
        """Map mouse position to original image coordinates"""
        pixmap = self.camera_view.pixmap()
        if pixmap is None or pixmap.isNull() or self.current_frame is None:
            return None
        
        # Calculate scaling and offset
        scaled_size = pixmap.size()
        widget_size = self.camera_view.size()
        offset_x = (widget_size.width() - scaled_size.width()) // 2
        offset_y = (widget_size.height() - scaled_size.height()) // 2
        
        # Calculate position in the scaled image
        pos_x = max(0, min(event_pos.x() - offset_x, scaled_size.width() - 1))
        pos_y = max(0, min(event_pos.y() - offset_y, scaled_size.height() - 1))
        
        # Check if click is within the image bounds
        if 0 <= pos_x < scaled_size.width() and 0 <= pos_y < scaled_size.height():
            # Convert to original image coordinates
            img_height, img_width = self.current_frame.shape[:2]
            scale_x = img_width / scaled_size.width()
            scale_y = img_height / scaled_size.height()
            
            original_x = int(pos_x * scale_x)
            original_y = int(pos_y * scale_y)
            
            return (original_x, original_y)
        
        return None
    
    def is_inside_box(self, point, box):
        """Check if point is inside the box"""
        if point and box:
            x, y = point
            box_x, box_y, box_w, box_h = box
            return (box_x <= x <= box_x + box_w) and (box_y <= y <= box_y + box_h)
        return False
    
    def mouse_press_event(self, event):
        """Handle mouse press events for bounding box drawing or dragging"""
        image_point = self.map_to_image_coords(event.pos())
        if not image_point:
            return
        
        # CASE 1: We're in draw mode - start drawing a new box
        if self.draw_mode and not self.is_tracking:
            self.start_point = image_point
            self.end_point = image_point
            self.drawing = True
            return
        
        # CASE 2: We're tracking and can drag the box
        if self.is_tracking:
            # Get current box
            if self.current_box:
                # Check if click is inside the box
                if self.is_inside_box(image_point, self.current_box):
                    self.dragging = True
                    x, y = image_point
                    box_x, box_y, _, _ = self.current_box
                    # Save offset for smooth dragging
                    self.drag_offset_x = x - box_x
                    self.drag_offset_y = y - box_y
    
    def mouse_move_event(self, event):
        """Handle mouse move events for drawing or dragging box"""
        image_point = self.map_to_image_coords(event.pos())
        if not image_point:
            return
        
        # CASE 1: Drawing a new box
        if self.draw_mode and self.drawing:
            self.end_point = image_point
            return
            
        # CASE 2: Dragging existing box
        if self.is_tracking and self.dragging:
            x, y = image_point
            _, _, w, h = self.current_box
            
            # Calculate new position with the offset
            new_x = x - self.drag_offset_x
            new_y = y - self.drag_offset_y
            
            # Ensure box stays within frame
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
                new_x = max(0, min(new_x, width - w))
                new_y = max(0, min(new_y, height - h))
            
            # Update box
            self.current_box = (int(new_x), int(new_y), w, h)
            
            # IMPORTANT: If recording is active, save the frame with the updated box 
            # but throttle the saves to avoid excessive files
            if self.is_recording and hasattr(self, 'current_frame') and self.current_frame is not None:
                try:
                    # Store timestamp of last drag update
                    current_time = time.time()
                    last_drag_save = getattr(self, 'last_drag_save', 0)
                    
                    # Only save if it's been at least 100ms since the last save during dragging
                    # This prevents thousands of frames while still making it responsive
                    if current_time - last_drag_save > 0.1:  # 100ms throttle
                        # Record a frame with the updated box position
                        self.save_frame(self.current_frame, self.current_box)
                        self.last_drag_save = current_time
                except Exception as e:
                    print(f"Error saving dragged box frame: {e}")
    
    def mouse_release_event(self, event):
        """Handle mouse release events for bounding box drawing or dragging"""
        # CASE 1: Finalizing a drawn box
        if self.draw_mode and self.drawing:
            self.drawing = False
            
            if self.start_point and self.end_point:
                # Calculate the bounding box (x, y, width, height)
                x = min(self.start_point[0], self.end_point[0])
                y = min(self.start_point[1], self.end_point[1])
                w = abs(self.end_point[0] - self.start_point[0])
                h = abs(self.end_point[1] - self.start_point[1])
                
                # Ensure the bounding box has a minimum size
                if w < 10 or h < 10:
                    self.status_label.setText("Status: Bounding box too small (min 10px width/height)")
                    return
                
                # Initialize tracking with the drawn box
                bbox = (x, y, w, h)
                self.current_box = bbox
                
                if self.current_frame is not None:
                    self.anchor_pt = (bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2)
                    
                    # Initialize DINO tracker if auto-tracking is enabled
                    if self.track_checkbox.isChecked():
                        self.dino_tracker.init_from_bbox(self.current_frame, bbox)
                    
                    self.is_tracking = True
                    self.draw_mode = False
                    
                    if self.track_checkbox.isChecked():
                        self.status_label.setText(f"Status: Auto-tracking {self.current_class}")
                    else:
                        self.status_label.setText(f"Status: Box created for {self.current_class} - you can drag it")
                        
                    # Enable the record button now that we have a box
                    self.record_btn.setEnabled(True)
            
        # CASE 2: Finishing dragging
        elif self.dragging:
            self.dragging = False
            
            # Force save a frame with the final box position when recording
            if self.is_recording and hasattr(self, 'current_frame') and self.current_frame is not None:
                try:
                    # Always save on mouse release to ensure final position is captured
                    self.save_frame(self.current_frame, self.current_box)
                    print(f"Saved final dragged position: {self.current_box}")
                except Exception as e:
                    print(f"Error saving final box position: {e}")
    
    def keyPressEvent(self, event):
        # Cancel drawing with Escape key
        if event.key() == Qt.Key_Escape and self.draw_mode:
            self.draw_mode = False
            self.drawing = False
            self.status_label.setText("Status: Drawing cancelled")
    
    def toggle_tracking(self, enabled: bool):
        """
        Checkbox handler:
          • when ON  → enable center‑point DINO tracker + smoothing slider
          • when OFF → stop automatic tracking; user can drag box manually
        """
        self.is_tracking = enabled and self.current_box is not None
        # enable / disable slider + style
        self.smooth_slider.setEnabled(enabled)
        if enabled:
            self.smooth_slider.setStyleSheet(self.slider_enabled_style)
            self.smooth_label.setStyleSheet("color:#555;")
            self.smooth_val.setStyleSheet("font-weight:bold; color:#333; min-width:40px;")
            # (re)start DINO tracker
            if self.current_box and self.current_frame is not None:
                self.dino_tracker.init_from_bbox(self.current_frame, self.current_box)
        else:
            self.smooth_slider.setStyleSheet(self.slider_disabled_style)
            self.smooth_label.setStyleSheet("color:#999;")
            self.smooth_val.setStyleSheet("font-weight:bold; color:#999; min-width:40px;")
        
        # Update status with context-aware message
        if self.current_box is not None:
            if enabled:
                self.status_label.setText("Status: Tracking enabled - automatic point tracking")
            else:
                self.status_label.setText("Status: Tracking disabled - you can drag the box")
        else:
            self.status_label.setText(f"Status: {'Auto' if enabled else 'Manual'} tracking selected")
            
    def update_smoothing(self, val):
        alpha = val / 100.0
        self.smooth_val.setText(f"{alpha:.2f}")
        self.dino_tracker.alpha = alpha
    
    # Old methods removed as they're no longer needed
    
    def toggle_draw_mode(self):
        """Toggle the drawing mode for bounding boxes"""
        if self.cap is None or not self.cap.isOpened():
            self.status_label.setText("Status: Start camera first")
            return
            
        # Special case: If we're in tracking mode and button says "Delete Bounding Box"
        if self.is_tracking and self.draw_box_btn.text() == "Delete Bounding Box":
            # This is a request to delete the current bounding box
            self.is_tracking = False  # Stop tracking
            self.current_box = None   # Remove box
            self.status_label.setText("Status: Bounding box deleted")
            self.draw_box_btn.setText("Draw Bounding Box")
            return
            
        # Normal drawing workflow
        if self.is_tracking:
            self.status_label.setText("Status: Cancel tracking first")
            return
            
        # Get current class if entering draw mode
        if not self.draw_mode:  # About to enable drawing
            self.current_class = self.class_dropdown.currentText()
            if not self.current_class:
                self.status_label.setText("Status: Please select or add a class first")
                return
                
        self.draw_mode = not self.draw_mode
        
        if self.draw_mode:
            # Start drawing new box
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.status_label.setText(f"Status: Draw a box around the {self.current_class}")
            self.draw_box_btn.setText("Delete Bounding Box")
        else:
            # Cancel drawing - simply reset the drawing state
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.status_label.setText("Status: Drawing cancelled")
            self.draw_box_btn.setText("Draw Bounding Box")
            
    def cleanup(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()  
        if self.timer.isActive():
            self.timer.stop()
