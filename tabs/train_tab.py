#!/usr/bin/env python3
import os
import subprocess
import threading
import json
import tempfile
import shutil
import time
import math
import glob
import webbrowser
import logging
from pathlib import Path
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, 
                             QComboBox, QTextEdit, QGroupBox, QMessageBox, QCheckBox,
                             QScrollArea, QFrame, QToolButton, QTabWidget, QGridLayout,
                             QProgressBar, QSizePolicy, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QSize, QTimer, QFileSystemWatcher
from PyQt5.QtGui import QIcon, QPixmap, QCursor, QImage
from .utils import create_folder_if_not_exists

class ClickableImageLabel(QLabel):
    """A custom QLabel that can be clicked to open the image in a browser"""
    
    def __init__(self, image_path=None, title=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.title = title or os.path.basename(image_path) if image_path else "No Image"
        
        # Set up appearance
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(200, 150))
        self.setMaximumSize(QSize(250, 200))
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 6px;
                padding: 4px;
                background-color: white;
            }
            QLabel:hover {
                border-color: #2196F3;
                background-color: #e3f2fd;
            }
        """)
        
        # Set cursor to hand when hovering
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        # Load image if provided
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        else:
            self.setText("No Image")
    
    def load_image(self, image_path):
        """Load an image from path and resize it to fit the label"""
        if not os.path.exists(image_path):
            self.setText("Image not found")
            return
            
        self.image_path = image_path
        
        # Load image using OpenCV for better handling of various formats
        try:
            # For plot images (PNG, JPG)
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                self.setText("Failed to load image")
                return
                
            # Convert to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create QImage and QPixmap
            h, w, c = img.shape
            q_img = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            pixmap = pixmap.scaled(self.maximumSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Set pixmap
            self.setPixmap(pixmap)
            
            # Set tooltip with file information
            self.setToolTip(f"{self.title}\nClick to view full size")
            
        except Exception as e:
            self.setText(f"Error: {str(e)}")
    
    def mousePressEvent(self, event):
        """Handle mouse press events to open image in browser"""
        if event.button() == Qt.LeftButton and self.image_path and os.path.exists(self.image_path):
            # Open the image in the default web browser using a separate thread
            image_path = self.image_path  # Create a local copy for the thread
            print(f"Opening image in browser: {os.path.abspath(image_path)}")
            threading.Thread(target=lambda: webbrowser.open(f"file:///{os.path.abspath(image_path)}"), 
                             daemon=True).start()
        super().mousePressEvent(event)

class TrainingWorker(QThread):
    update_signal = pyqtSignal(str)  # Raw output text
    metrics_signal = pyqtSignal(dict)  # Dictionary of all metrics
    image_update_signal = pyqtSignal(str)  # path to plot image
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, train_command, output_dir, is_cloud=False, api_key=""):
        super().__init__()
        self.train_command = train_command
        self.output_dir = output_dir
        self.process = None
        self.is_cloud = is_cloud
        self.api_key = api_key
        
        # Training metrics
        self.metrics = {
            'epoch': 0,
            'total_epochs': 0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'dfl_loss': 0.0,
            'map': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'phase': 'Initializing',
            'batch_progress': 0.0
        }
    
    def run(self):
        try:
            # On Windows, avoid using shell=True with complex commands
            if os.name == 'nt' and not self.train_command.startswith("cd "):
                # Direct command without cd or shell
                self.process = subprocess.Popen(
                    self.train_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True,  # Still need shell for the quotes to work properly
                    bufsize=1
                )
            else:
                # Standard command with shell
                self.process = subprocess.Popen(
                    self.train_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    shell=True,
                    bufsize=1
                )
            
            # Stream the output
            for line in iter(self.process.stdout.readline, ''):
                text_line = line.strip()
                self.update_signal.emit(text_line)
                
                # Parse training output for metrics
                try:
                    # Extract useful training information
                    import re
                    
                    # 1. Detect training start - extract total epochs
                    if "Starting training for" in text_line:
                        self.metrics['phase'] = "Starting Training"
                        epoch_match = re.search(r'Starting training for (\d+) epochs', text_line)
                        if epoch_match:
                            self.metrics['total_epochs'] = int(epoch_match.group(1))
                            print(f"Training will run for {self.metrics['total_epochs']} epochs")
                    
                    # 2. Detect training header
                    elif "Epoch    GPU_mem   box_loss" in text_line:
                        self.metrics['phase'] = "Training"
                        print("Training header detected - training is starting")
                    
                    # 3. Most important: Parse epoch/metrics lines
                    elif text_line.strip() and text_line[0].isdigit() and "/" in text_line.split()[0]:
                        parts = text_line.split()
                        epoch_part = parts[0]  # e.g., "1/50"
                        
                        # Check if this is a YOLO training line with metrics
                        if "/" in epoch_part and len(parts) > 6:  # Training line has many fields
                            try:
                                current, total = map(int, epoch_part.split('/'))
                                
                                # Update epoch data
                                self.metrics['epoch'] = current
                                self.metrics['total_epochs'] = total
                                self.metrics['phase'] = "Training"
                                
                                # YOLO format:
                                # "1/50     19.7G    0.07304  0.09175  0.02089   0.1857       16       640      0.635      0.566      0.615"
                                # epoch     GPU      box_loss cls_loss dfl_loss  total_loss   it/s     img_size  precision  recall     mAP
                                
                                # Parse the metrics if they're available
                                if len(parts) >= 11:  # Make sure we have all fields
                                    try:
                                        # Extract loss values - positions 2, 3, 4
                                        self.metrics['box_loss'] = float(parts[2])
                                        self.metrics['cls_loss'] = float(parts[3]) 
                                        self.metrics['dfl_loss'] = float(parts[4])
                                        
                                        # Extract validation metrics - positions 8, 9, 10
                                        if len(parts) >= 11:
                                            self.metrics['precision'] = float(parts[8])
                                            self.metrics['recall'] = float(parts[9])
                                            self.metrics['map'] = float(parts[10])
                                            
                                        print(f"Metrics updated: box_loss={self.metrics['box_loss']:.4f}, "
                                              f"cls_loss={self.metrics['cls_loss']:.4f}, "
                                              f"dfl_loss={self.metrics['dfl_loss']:.4f}, "
                                              f"mAP={self.metrics['map']:.4f}")
                                    except ValueError:
                                        # Silently ignore parsing errors - common during training
                                        if "Debug" in os.environ.get("YOLOFLOW_LOG_LEVEL", ""):
                                            print(f"Debug - Couldn't parse metrics from line")
                                    
                                # Emit the metrics update signal
                                self.metrics_signal.emit(self.metrics)
                            except:
                                pass  # Not a valid epoch line
                    
                    # 4. Validation phase
                    elif "Validating" in text_line:
                        self.metrics['phase'] = "Validating"
                        self.metrics_signal.emit(self.metrics)
                        
                    # 5. Parse batch progress (if line contains percentage)
                    elif "%" in text_line:
                        percent_matches = re.findall(r'(\d+)%', text_line)
                        if percent_matches:
                            try:
                                batch_percent = int(percent_matches[0])
                                self.metrics['batch_progress'] = batch_percent / 100.0
                                print(f"Batch progress: {batch_percent}%")
                                self.metrics_signal.emit(self.metrics)
                            except:
                                pass
                    
                    # Enhanced progress detection
                    import re
                    
                    # Extract any epoch information (works for multiple formats)
                    # First, check for the regular epoch pattern (e.g., "Epoch 1/50:")
                    epoch_matches = re.findall(r'[Ee]poch[: ]*(\d+)[/\\](\d+)', text_line)
                    
                    # Also look for YOLO specific format at the beginning of lines (e.g., "1/50      9.04G...")
                    if not epoch_matches and text_line.strip():
                        # Pattern 1: Look for digit/digit at start of line followed by whitespace 
                        epoch_matches_alt = re.findall(r'^(\d+)/(\d+)\s+', text_line)
                        if epoch_matches_alt:
                            epoch_matches = epoch_matches_alt  # Use this match instead
                            if "Debug" in os.environ.get("YOLOFLOW_LOG_LEVEL", ""):
                                print(f"Debug - Found YOLO training progress: {text_line[:20]}...")
                        # Pattern 2: Look for digit/digit anywhere in the line as a fallback
                        elif not epoch_matches and re.search(r'\d+/\d+', text_line):
                            general_epoch = re.findall(r'(\d+)/(\d+)', text_line)
                            if general_epoch:
                                # Validate the numbers make sense as epochs (usually not more than a few hundred)
                                if int(general_epoch[0][1]) < 1000:  # Total epochs unlikely to be >1000
                                    epoch_matches = [general_epoch[0]]
                                    print(f"Debug - Found general epoch format: {general_epoch[0][0]}/{general_epoch[0][1]}")
                            
                    # We don't need to process these epoch_matches anymore
                    # The direct detection above already handles epochs and progress
                        
                        # This section is no longer needed as we have more specific pattern matching above
                        # The code now uses the metrics dictionary and metrics_signal instead of progress_signal
                    
                    # This section is now handled by the structured metrics tracking above
                    
                    # This section is no longer needed as we're using the more structured metrics tracking above
                    
                    # Check for plot updates
                    if "results saved to" in text_line.lower():
                        # Try to find the results directory
                        parts = text_line.split("results saved to")
                        if len(parts) > 1:
                            results_dir = parts[1].strip()
                            # Look for plots
                            plot_path = os.path.join(results_dir, "results.png")
                            if os.path.exists(plot_path):
                                self.image_update_signal.emit(plot_path)
                
                except Exception as parse_error:
                    print(f"Error parsing training output: {str(parse_error)}")
                
                if not line:
                    break
            
            # Get the return code
            self.process.wait()
            success = self.process.returncode == 0
            
            # Look for plot in the output directory
            try:
                if success:
                    # Find most recent run directory
                    base_dir = os.path.dirname(self.output_dir)
                    detect_dir = os.path.join(base_dir, "detect")
                    if os.path.exists(detect_dir):
                        run_dirs = [os.path.join(detect_dir, d) for d in os.listdir(detect_dir) 
                                    if os.path.isdir(os.path.join(detect_dir, d))]
                        if run_dirs:
                            # Sort by modification time (newest first)
                            run_dirs.sort(key=os.path.getmtime, reverse=True)
                            latest_run = run_dirs[0]
                            plot_path = os.path.join(latest_run, "results.png")
                            if os.path.exists(plot_path):
                                self.image_update_signal.emit(plot_path)
            except Exception as e:
                print(f"Error finding plot: {str(e)}")
            
            if success:
                self.finished_signal.emit(True, "Training completed successfully!")
            else:
                self.finished_signal.emit(False, f"Training failed with exit code {self.process.returncode}")
                
        except Exception as e:
            self.update_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit(False, f"Training failed: {str(e)}")
    
    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.update_signal.emit("Training process terminated by user.")

class CollapsibleBox(QWidget):
    """Custom collapsible box widget for parameter sections"""
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)
        
        # Create a container widget for the header to make the entire area clickable
        self.header_widget = QWidget()
        header_layout = QHBoxLayout(self.header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Configure the toggle button with larger size and improved styling
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QToolButton { 
                border: none; 
                padding: 8px; 
                font-weight: bold;
            }
            QToolButton:hover { 
                background-color: rgba(200, 200, 200, 30);
            }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setMinimumHeight(32)  # Make the button taller
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        # Use clicked signal instead of pressed for more reliable toggling
        self.toggle_button.clicked.connect(self.on_clicked)
        
        # Add the button to the header layout
        header_layout.addWidget(self.toggle_button, 1)
        
        self.toggle_animation = None
        
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.content_area.setFrameShape(QFrame.NoFrame)
        
        # Make the entire header widget clickable
        self.header_widget.setCursor(Qt.PointingHandCursor)
        self.header_widget.mousePressEvent = self.header_clicked
        
        # Main layout
        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.header_widget)
        lay.addWidget(self.content_area)
        
    def header_clicked(self, event):
        # Toggle the button state when anywhere in the header is clicked
        self.toggle_button.setChecked(not self.toggle_button.isChecked())
        self.on_clicked()
        
    def on_clicked(self):
        # Update the UI based on the new toggle state
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
        self.on_clicked()


class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set default data path to the app's data directory
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
        
        # YOLO model outputs
        # Ensure the path is /root/runs/train and not in tabs directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get /root directory
        self.output_dir = os.path.join(base_dir, "runs/train")
        create_folder_if_not_exists(self.output_dir)
        
        # We'll use the data directory directly - no temp dir needed
        self.using_temp_dir = False
        
        # Training worker
        self.training_worker = None
        
        # Set up the UI
        self.setup_ui()
        
        # Update the data path input
        self.data_path_input.setText(self.data_path)
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        
        # Create a tab widget for the main sections
        self.main_tabs = QTabWidget()
        
        # Augmentation is now always available
        def on_temp_dir_toggled(checked):
            # This function is now a no-op since we always enable augmentation
            pass
            
        # We'll connect this later after creating the checkboxes
        
        # ===== SETUP TAB =====
        setup_tab = QWidget()
        setup_layout = QVBoxLayout(setup_tab)
        
        # ── Run / Model name field ──────────────────────────────────────────
        name_group = QGroupBox("Run / Model Name")
        name_layout = QHBoxLayout()
        name_label  = QLabel("Name:")
        self.run_name_edit = QLineEdit()

        # default ─ pick something like 20250420_153012
        default_stamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_name_edit.setText(f"run_{default_stamp}")
        self.run_name_edit.setPlaceholderText("leave blank to use timestamp")

        name_layout.addWidget(name_label)
        name_layout.addWidget(self.run_name_edit)
        name_group.setLayout(name_layout)

        # stick it *above* Data Settings
        setup_layout.addWidget(name_group)
        
        # Data settings group
        data_group = QGroupBox("Data Settings")
        data_layout = QFormLayout()
        
        # Data info section
        self.data_info_label = QLabel("Using data from the Capture tab")
        self.data_path_input = QLineEdit()
        self.data_path_input.setReadOnly(True)
        data_layout.addRow("Data Source:", self.data_info_label)
        data_layout.addRow("Data Path:", self.data_path_input)
        
        # Dataset preparation
        self.prepare_dataset_btn = QPushButton("Prepare Dataset for Training")
        self.prepare_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9900;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #ffad33;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.prepare_dataset_btn.clicked.connect(self.prepare_dataset)
        data_layout.addRow("", self.prepare_dataset_btn)
        
        data_group.setLayout(data_layout)
        setup_layout.addWidget(data_group)
        
        # Collapsible Parameters section
        self.params_box = CollapsibleBox("Advanced Training Parameters (click to expand)")
        params_layout = QFormLayout()
        
        # Model information (using newest YOLO version)
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models")
        
        # Model dropdown selection
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            "Nano (Smallest, fastest)",
            "Small (Balanced speed/accuracy)",
            "Medium (Better accuracy)",
            "Large (High accuracy, default)",
            "Extra Large (Highest accuracy)"
        ])
        self.model_dropdown.setCurrentIndex(3)  # Default to Large model
        params_layout.addRow("Model Size:", self.model_dropdown)
        
        # Epochs
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(50)
        params_layout.addRow("Epochs:", self.epochs_input)
        
        # Batch size
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 128)
        self.batch_size_input.setValue(16)
        params_layout.addRow("Batch Size:", self.batch_size_input)
        
        # Image size
        self.img_size_input = QSpinBox()
        self.img_size_input.setRange(32, 1280)
        self.img_size_input.setValue(640)
        self.img_size_input.setSingleStep(32)
        params_layout.addRow("Image Size:", self.img_size_input)
        
        # Learning rate
        self.lr_input = QDoubleSpinBox()
        self.lr_input.setRange(0.00001, 0.1)
        self.lr_input.setValue(0.01)
        self.lr_input.setDecimals(5)
        self.lr_input.setSingleStep(0.001)
        params_layout.addRow("Learning Rate:", self.lr_input)
        
        # Augmentation options
        augmentation_group = QGroupBox("Data Augmentation")
        augmentation_layout = QVBoxLayout()
        
        # Flip augmentation
        self.use_flip_checkbox = QCheckBox("Enable horizontal flip")
        self.use_flip_checkbox.setChecked(False)  # Off by default
        augmentation_layout.addWidget(self.use_flip_checkbox)
        
        # Rotation augmentation
        self.use_rotation_checkbox = QCheckBox("Enable rotation (±15°)")
        self.use_rotation_checkbox.setChecked(False)  # Off by default
        augmentation_layout.addWidget(self.use_rotation_checkbox)
        
        # Scale jitter
        self.use_scale_checkbox = QCheckBox("Enable scale jitter (±25%)")
        self.use_scale_checkbox.setChecked(False)  # Off by default
        augmentation_layout.addWidget(self.use_scale_checkbox)
        
        augmentation_multiplier_layout = QHBoxLayout()
        augmentation_multiplier_layout.addWidget(QLabel("Augmentation multiplier:"))
        self.augmentation_multiplier = QComboBox()
        self.augmentation_multiplier.addItems(["None (1x)", "Light (2x)", "Standard (3x)", "Heavy (4x)"])
        self.augmentation_multiplier.setCurrentIndex(0)  # Default to None (1x)
        augmentation_multiplier_layout.addWidget(self.augmentation_multiplier)
        augmentation_layout.addLayout(augmentation_multiplier_layout)
        
        augmentation_group.setLayout(augmentation_layout)
        params_layout.addRow("", augmentation_group)
        
        # We now directly use the data directory instead of a temporary one
        self.use_temp_dir_checkbox = QCheckBox("Use data directory directly (recommended)")
        self.use_temp_dir_checkbox.setChecked(True)  # Always on
        self.use_temp_dir_checkbox.setEnabled(False)  # Disable the option since we always use data directory
        self.use_temp_dir_checkbox.setToolTip("Uses the data directory directly without creating a temporary copy")
        params_layout.addRow("", self.use_temp_dir_checkbox)
        
        # Connect the toggle function now that we have the checkbox
        self.use_temp_dir_checkbox.toggled.connect(on_temp_dir_toggled)
        
        self.params_box.setContentLayout(params_layout)
        setup_layout.addWidget(self.params_box)
        
        # Cloud API Key (for Ultralytics cloud training)
        cloud_group = QGroupBox("Cloud Training")
        cloud_layout = QVBoxLayout()
        
        # Use a form layout for the API key
        api_key_form = QFormLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter your Ultralytics API key")
        self.api_key_input.setEchoMode(QLineEdit.Password)  # Hide the API key
        api_key_form.addRow("API Key:", self.api_key_input)
        cloud_layout.addLayout(api_key_form)
        
        # Add a note about notebook training
        notebook_note = QLabel("Notebook training integration coming soon!")
        notebook_note.setStyleSheet("color: #3366cc; font-style: italic; font-weight: bold;")
        notebook_note.setAlignment(Qt.AlignCenter)
        cloud_layout.addWidget(notebook_note)
        
        cloud_group.setLayout(cloud_layout)
        setup_layout.addWidget(cloud_group)
        
        # Training controls
        controls_layout = QHBoxLayout()
        
        # Style the buttons with more visual appeal
        self.train_local_btn = QPushButton("Train Locally")
        self.train_local_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.train_cloud_btn = QPushButton("Train in Cloud")
        self.train_cloud_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.train_local_btn.clicked.connect(lambda: self.start_training(is_cloud=False))
        self.train_cloud_btn.clicked.connect(lambda: self.start_training(is_cloud=True))
        self.stop_btn.clicked.connect(self.stop_training)
        
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.train_local_btn)
        controls_layout.addWidget(self.train_cloud_btn)
        controls_layout.addWidget(self.stop_btn)
        setup_layout.addLayout(controls_layout)
        
        # Add some space
        setup_layout.addStretch()
        
        # ===== TRAINING TAB =====
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        
        # Training Monitor - Enhanced metrics display
        monitor_group = QGroupBox("Training Monitor")
        monitor_group_layout = QVBoxLayout()
        
        # Add a progress bar for epoch tracking
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Training Progress:")
        progress_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        progress_layout.addWidget(progress_label)
        
        # Progress bar with percentage display
        progress_bar_layout = QHBoxLayout()
        self.epoch_progress = QProgressBar()
        self.epoch_progress.setRange(0, 100)
        self.epoch_progress.setValue(0)
        self.epoch_progress.setTextVisible(True)
        self.epoch_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 1px;
                text-align: center;
                height: 20px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0px;
            }
        """)
        progress_bar_layout.addWidget(self.epoch_progress, 9)
        
        # Add epoch counter next to progress bar
        self.epoch_label = QLabel("0 / 0")
        self.epoch_label.setStyleSheet("padding-left: 8px; font-weight: bold;")
        progress_bar_layout.addWidget(self.epoch_label, 1)
        
        progress_layout.addLayout(progress_bar_layout)
        monitor_group_layout.addLayout(progress_layout)
        
        # Create metrics display grid
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QGridLayout()
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnStretch(1, 2)
        
        # Style for metric labels
        metric_style = "font-weight: bold; color: #333;"
        value_style = "font-family: monospace; background-color: #f5f5f5; padding: 3px; border: 1px solid #ddd; border-radius: 3px;"
        
        # Box loss
        metrics_layout.addWidget(QLabel("Box Loss:"), 0, 0)
        self.box_loss_label = QLabel("0.000")
        self.box_loss_label.setStyleSheet(value_style)
        metrics_layout.addWidget(self.box_loss_label, 0, 1)
        
        # Classification loss
        metrics_layout.addWidget(QLabel("Cls Loss:"), 1, 0)
        self.cls_loss_label = QLabel("0.000")
        self.cls_loss_label.setStyleSheet(value_style)
        metrics_layout.addWidget(self.cls_loss_label, 1, 1)
        
        # DFL loss
        metrics_layout.addWidget(QLabel("DFL Loss:"), 2, 0)
        self.dfl_loss_label = QLabel("0.000")
        self.dfl_loss_label.setStyleSheet(value_style)
        metrics_layout.addWidget(self.dfl_loss_label, 2, 1)
        
        # mAP
        metrics_layout.addWidget(QLabel("mAP:"), 3, 0)
        self.map_label = QLabel("0.000")
        self.map_label.setStyleSheet(value_style)
        metrics_layout.addWidget(self.map_label, 3, 1)
        
        # Elapsed time
        metrics_layout.addWidget(QLabel("Elapsed:"), 4, 0)
        self.time_label = QLabel("00:00:00")
        self.time_label.setStyleSheet(value_style)
        metrics_layout.addWidget(self.time_label, 4, 1)
        
        metrics_group.setLayout(metrics_layout)
        monitor_group_layout.addWidget(metrics_group)
        
        # Hidden status message (kept for code compatibility)
        self.status_message = QLabel("Ready")
        self.status_message.setVisible(False)
        
        # Create a console output for logging
        self.console_output = QTextEdit()
        self.console_output.setMaximumHeight(200)
        self.console_output.setReadOnly(True)
        monitor_group_layout.addWidget(self.console_output)
        
        # Add a note about console output
        console_note = QLabel("Training log output is shown above")
        console_note.setStyleSheet("font-style: italic; color: #666; margin-top: 5px;")
        console_note.setAlignment(Qt.AlignCenter)
        monitor_group_layout.addWidget(console_note)
        
        # Set the group box layout
        monitor_group.setLayout(monitor_group_layout)
        
        # Add the monitor group to the tab
        monitor_layout.addWidget(monitor_group)
        
        # Add training start time tracker
        self.training_start_time = None
        # Add a timer to update elapsed time
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_elapsed_time)
        
        # Create simple results visualization
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_group)
        
        # Display title and description
        self.display_title = QLabel("Results")
        self.display_title.setAlignment(Qt.AlignCenter)
        self.display_title.setStyleSheet("font-weight: bold;")
        results_layout.addWidget(self.display_title)
        
        # Main display area
        self.results_image = QLabel("Training results will appear here when available")
        self.results_image.setAlignment(Qt.AlignCenter)
        self.results_image.setMinimumHeight(400)
        self.results_image.setStyleSheet("border: 1px solid #ccc; background-color: white;")
        results_layout.addWidget(self.results_image)
        
        self.display_description = QLabel("Training not started")
        self.display_description.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.display_description)
        
        # Dictionary to store image paths for results
        self.thumbnail_paths = {}
        self.thumbnail_titles = {}
        
        # Add the results group to the tab
        monitor_layout.addWidget(results_group)
        
        # Store a list of expected plot files for easy reference
        self.plot_types = {
            "results.png": "Training Metrics Overview",
            "confusion_matrix.png": "Confusion Matrix",
            "confusion_matrix_normalized.png": "Normalized Confusion Matrix",
            "PR_curve.png": "Precision-Recall Curve",
            "F1_curve.png": "F1-Score Curve",
            "P_curve.png": "Precision Curve",
            "R_curve.png": "Recall Curve",
            "labels.jpg": "Labels Distribution",
            "labels_correlogram.jpg": "Labels Correlation",
            "val_batch0_pred.jpg": "Validation Predictions (Batch 0)",
            "val_batch1_pred.jpg": "Validation Predictions (Batch 1)",
            "val_batch2_pred.jpg": "Validation Predictions (Batch 2)"
        }
        
        # We now use console_output for all logging directly
        
        # ===== GALLERY TAB =====
        gallery_tab = QWidget()
        gallery_layout = QVBoxLayout(gallery_tab)
        
        # Create a group box for the gallery
        gallery_group = QGroupBox("Training Visualizations Gallery")
        gallery_group_layout = QVBoxLayout()
        
        # Explanation text
        gallery_explanation = QLabel("Click on any image to view it in full size. Images are automatically updated when new training runs are completed.")
        gallery_explanation.setWordWrap(True)
        gallery_explanation.setStyleSheet("font-style: italic; color: #666; margin-bottom: 10px;")
        gallery_group_layout.addWidget(gallery_explanation)
        
        # Add refresh button with icon
        refresh_layout = QHBoxLayout()
        self.gallery_refresh_btn = QPushButton("Refresh Gallery")
        self.gallery_refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.gallery_refresh_btn.clicked.connect(self.refresh_gallery)
        refresh_layout.addWidget(self.gallery_refresh_btn)
        refresh_layout.addStretch()
        gallery_group_layout.addLayout(refresh_layout)
        
        # Create a scroll area for the gallery
        gallery_scroll = QScrollArea()
        gallery_scroll.setWidgetResizable(True)
        gallery_scroll.setFrameShape(QFrame.NoFrame)
        
        # Create a widget to hold the grid layout
        gallery_content = QWidget()
        self.gallery_grid = QGridLayout(gallery_content)
        self.gallery_grid.setSpacing(12)
        
        # Add the content widget to the scroll area
        gallery_scroll.setWidget(gallery_content)
        gallery_group_layout.addWidget(gallery_scroll)
        
        # Set the layout for the gallery group
        gallery_group.setLayout(gallery_group_layout)
        gallery_layout.addWidget(gallery_group)
        
        # Add tabs to the main tab widget
        self.main_tabs.addTab(setup_tab, "Setup")
        self.main_tabs.addTab(monitor_tab, "Training Monitor")
        self.main_tabs.addTab(gallery_tab, "Visualizations")
        
        main_layout.addWidget(self.main_tabs)
        self.setLayout(main_layout)
        
        # Setup file system watcher for the gallery
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)
        
        # Initial gallery refresh
        self.refresh_gallery()
    
    def refresh_gallery(self):
        """Refresh the visualization gallery with training images"""
        # Clear existing gallery items
        for i in reversed(range(self.gallery_grid.count())):
            widget = self.gallery_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Get runs directory
        runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
        train_dir = os.path.join(runs_dir, "train")
        
        if not os.path.exists(train_dir):
            print(f"No training runs directory found at: {train_dir}")
            return
            
        # Find all training run directories
        run_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        if not run_dirs:
            print("No training runs found")
            return
            
        # Get visualization images
        vis_images = []
        for run_dir in run_dirs:
            for img_ext in ['.png', '.jpg']:
                for img_file in glob.glob(os.path.join(run_dir, f"*{img_ext}")):
                    # Skip specific file patterns that aren't visualizations
                    if 'labels_correlogram' in img_file or 'labels.' in img_file:
                        continue
                    vis_images.append(img_file)
        
        # Sort by modification time (newest first)
        vis_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Limit to a reasonable number to avoid overwhelming the UI
        vis_images = vis_images[:30]
        
        if not vis_images:
            print("No visualization images found in training runs")
            return
            
        # Add images to gallery
        cols = 3
        for i, img_path in enumerate(vis_images):
            row = i // cols
            col = i % cols
            
            # Create clickable thumbnail
            thumb = QLabel()
            pixmap = QPixmap(img_path)
            scaled_pixmap = pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            thumb.setPixmap(scaled_pixmap)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 8px;")
            
            # Add image name as label
            img_name = os.path.basename(img_path)
            run_name = os.path.basename(os.path.dirname(img_path))
            label = QLabel(f"{img_name}<br><i>{run_name}</i>")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 11px; color: #666;")
            
            # Create container for thumbnail and label
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(5, 5, 5, 5)
            layout.addWidget(thumb)
            layout.addWidget(label)
            
            # Add to grid
            self.gallery_grid.addWidget(container, row, col)
            
            # Store the full image path for clicking
            thumb.setProperty("fullpath", img_path)
            thumb.setCursor(Qt.PointingHandCursor)
            thumb.mousePressEvent = lambda event, path=img_path: self.show_full_image(path)
            
        # Add the directory to the file watcher
        if train_dir not in self.file_watcher.directories():
            self.file_watcher.addPath(train_dir)
        
        logging.debug(f"Gallery refreshed with {len(vis_images)} images")
    
    def show_full_image(self, img_path):
        """Show full-sized image when thumbnail is clicked"""
        if os.path.exists(img_path):
            # Open the image in the default web browser instead of a dialog
            # This prevents the UI from freezing when displaying large images
            try:
                # Convert to absolute path and use file:/// protocol
                abs_path = os.path.abspath(img_path)
                webbrowser.open(f"file:///{abs_path}")
                print(f"Opening image in browser: {abs_path}")
            except Exception as e:
                print(f"Error opening image in browser: {str(e)}")
                QMessageBox.warning(self, "Error", f"Could not open image: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", f"Image not found: {img_path}")
    
    def on_directory_changed(self, path):
        """Called when the training directory changes"""
        # Only refresh if it's a change in the runs/train directory
        runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
        train_dir = os.path.join(runs_dir, "train")
        
        if path == train_dir or path.startswith(train_dir):
            # Wait a short time to let file operations complete
            QTimer.singleShot(1000, self.refresh_gallery)
    
    def update_data_path(self, path):
        """Update the data path from another tab"""
        self.data_path = path
        self.data_path_input.setText(path)
    
    def prepare_dataset(self):
        """Prepare the dataset for YOLO training"""
        # Simply disable the button with text change, no fancy progress
        self.prepare_dataset_btn.setEnabled(False)
        self.prepare_dataset_btn.setText("Preparing...")
        QApplication.processEvents()
        
        # Log to console directly
        print("Preparing dataset...")
        
        try:
            # Check if the data directory exists
            if not os.path.exists(self.data_path):
                print(f"Error: Data directory not found: {self.data_path}")
                
                # Create a custom styled error message box
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Error")
                msg_box.setText(f"Data directory not found: {self.data_path}\n\nPlease capture some data in the Capture tab first.")
                msg_box.setIcon(QMessageBox.Critical)
                
                # Style the buttons
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: white;
                    }
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border-radius: 4px;
                        padding: 6px 16px;
                        font-weight: bold;
                        min-width: 80px;
                        min-height: 30px;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                    QPushButton:pressed {
                        background-color: #b71c1c;
                    }
                """)
                
                msg_box.exec_()
                
                self.prepare_dataset_btn.setEnabled(True)
                self.prepare_dataset_btn.setText("Prepare Dataset for Training")
                return
            
            # Check for required directories
            images_dir = os.path.join(self.data_path, "images")
            labels_dir = os.path.join(self.data_path, "labels")
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print("Error: Required directories not found.")
                
                # Create a custom styled error message box
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Error")
                msg_box.setText("Required directories not found.\n\nPlease capture some data in the Capture tab first.")
                msg_box.setIcon(QMessageBox.Critical)
                
                # Style the buttons
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: white;
                    }
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border-radius: 4px;
                        padding: 6px 16px;
                        font-weight: bold;
                        min-width: 80px;
                        min-height: 30px;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                    QPushButton:pressed {
                        background-color: #b71c1c;
                    }
                """)
                
                msg_box.exec_()
                
                self.prepare_dataset_btn.setEnabled(True)
                self.prepare_dataset_btn.setText("Prepare Dataset for Training")
                return
            
            # Count files
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            
            if not image_files or not label_files:
                print("Error: No image or label files found.")
                
                # Create a custom styled error message box
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Error")
                msg_box.setText("No image or label files found in the data directories.")
                msg_box.setIcon(QMessageBox.Critical)
                
                # Style the buttons
                msg_box.setStyleSheet("""
                    QMessageBox {
                        background-color: white;
                    }
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        border-radius: 4px;
                        padding: 6px 16px;
                        font-weight: bold;
                        min-width: 80px;
                        min-height: 30px;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                    QPushButton:pressed {
                        background-color: #b71c1c;
                    }
                """)
                
                msg_box.exec_()
                
                self.prepare_dataset_btn.setEnabled(True)
                self.prepare_dataset_btn.setText("Prepare Dataset for Training")
                return
                
            # Get class list
            class_list = []
            
            # Try to load classes from classes.txt if it exists
            classes_txt_path = os.path.join(self.data_path, "classes.txt")
            if os.path.exists(classes_txt_path):
                with open(classes_txt_path, 'r') as f:
                    class_list = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Found existing classes.txt with {len(class_list)} classes")
            
            # If classes.txt doesn't exist or is empty, try to extract from filenames
            if not class_list:
                print("No class list found, extracting from filenames...")
                for filename in image_files:
                    parts = os.path.splitext(filename)[0].split('_')
                    if parts and parts[0] not in class_list:
                        class_list.append(parts[0])
            
            # If still no classes found, use default
            if not class_list:
                class_list = ["object"]
                print("No classes found, using default 'object' class")
            
            # Create dataset.yaml
            dataset_yaml_path = os.path.join(self.data_path, "dataset.yaml")
            with open(dataset_yaml_path, 'w') as f:
                f.write(f"path: {self.data_path}\n")
                f.write(f"train: images\n")
                f.write(f"val: images\n")
                f.write(f"nc: {len(class_list)}\n")
                f.write(f"names: {class_list}\n")
            
            # Create or update classes.txt
            with open(classes_txt_path, 'w') as f:
                for class_name in class_list:
                    f.write(f"{class_name}\n")
            
            # Show a simple success message box - only this is UI interaction
            message = (
                f"Dataset prepared successfully!\n\n"
                f"• {len(class_list)} classes\n"
                f"• {len(image_files)} images\n"
                f"• {len(label_files)} labels"
            )
            
            print(f"Dataset prepared successfully with {len(class_list)} classes: {', '.join(class_list)}")
            print(f"Total images: {len(image_files)}")
            print(f"Dataset YAML created at: {dataset_yaml_path}")
            
            # Create a custom styled message box
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Success")
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Information)
            
            # Style the buttons
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 4px;
                    padding: 6px 16px;
                    font-weight: bold;
                    min-width: 80px;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            
            msg_box.exec_()
            
        except Exception as e:
            error_message = f"Error preparing dataset: {str(e)}"
            print(error_message)
            
            # Create a custom styled error message box
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Error")
            msg_box.setText(error_message)
            msg_box.setIcon(QMessageBox.Critical)
            
            # Style the buttons
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                }
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border-radius: 4px;
                    padding: 6px 16px;
                    font-weight: bold;
                    min-width: 80px;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
                QPushButton:pressed {
                    background-color: #b71c1c;
                }
            """)
            
            msg_box.exec_()
        
        # Re-enable the button
        self.prepare_dataset_btn.setEnabled(True)
        self.prepare_dataset_btn.setText("Prepare Dataset for Training")
    
    def start_training(self, is_cloud=False):
        """Start the YOLO training process"""
        # Clear previous state
        self.console_output.clear()
        self.epoch_label.setText("0 / 0")
        self.box_loss_label.setText("0.000")
        self.cls_loss_label.setText("0.000")
        self.dfl_loss_label.setText("0.000")
        self.map_label.setText("0.000")
        self.time_label.setText("00:00:00")
        self.results_image.setText("Training results will appear here when available")
        self.display_title.setText("Results")
        self.display_description.setText("Training started - results will appear here")
        
        # Reset progress bar and metrics tracking for new training run
        self.epoch_progress.setValue(0)
        self.epoch_progress.setFormat("Initializing...")
        self.epoch_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 4px;
                padding: 1px;
                text-align: center;
                height: 20px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0px;
            }
        """)
        
        # Create heartbeat timer for progress updates even when no signals are received
        if hasattr(self, 'heartbeat_timer') and self.heartbeat_timer.isActive():
            self.heartbeat_timer.stop()
        
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self.progress_heartbeat)
        self.heartbeat_timer.start(1500)  # Update progress every 1.5 seconds for smoother feedback
        
        # Start elapsed time tracking
        self.training_start_time = time.time()
        self.timer.start(1000)  # Update elapsed time every second
        
        # Check if the data directory exists
        if not os.path.exists(self.data_path):
            self.console_output.append(f"Error: Data directory not found: {self.data_path}")
            self.console_output.append("Please capture some data in the Capture tab first.")
            return
            
        # Check if there are images in the images directory
        images_dir = os.path.join(self.data_path, "images")
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            self.console_output.append("Error: No images found in the data directory.")
            self.console_output.append("Please capture some data in the Capture tab first.")
            return
        
        # We always use the data directory directly now
        dataset_yaml_path = os.path.join(self.data_path, "dataset.yaml")
            
        # Enable augmentations directly on data directory
        self.use_flip_checkbox.setEnabled(True) 
        self.use_rotation_checkbox.setEnabled(True)
        self.use_scale_checkbox.setEnabled(True)
        self.augmentation_multiplier.setEnabled(True)
        self.console_output.append("Using data directory directly for training.")
        self.console_output.append(f"Data directory: {self.data_path}")
            
        if not os.path.exists(dataset_yaml_path):
            # Automatically prepare the dataset
            self.console_output.append("No dataset.yaml found. Preparing dataset automatically...")
            self.prepare_dataset()
            
            # Check again after preparation
            if not os.path.exists(dataset_yaml_path):
                self.console_output.append("Error: Failed to prepare the dataset.")
                return
        
        # Check for cloud training API key if needed
        if is_cloud:
            api_key = self.api_key_input.text().strip()
            if not api_key:
                self.console_output.append("Error: API key is required for cloud training.")
                self.console_output.append("Please enter your Ultralytics API key in the Cloud Training section.")
                return
        
        # Get training parameters
        epochs = self.epochs_input.value()
        batch_size = self.batch_size_input.value()
        img_size = self.img_size_input.value()
        lr = self.lr_input.value()
        
        # Get model size based on dropdown selection
        model_size_map = {
            0: "yolo11n.pt",  # Nano
            1: "yolo11s.pt",  # Small
            2: "yolo11m.pt",  # Medium
            3: "yolo11l.pt",  # Large (default)
            4: "yolo11x.pt"   # Extra Large
        }
        model_size = model_size_map.get(self.model_dropdown.currentIndex(), "yolo11l.pt")
        model_name = os.path.splitext(model_size)[0]
        
        # Get run name from input field or use default with timestamp
        custom = self.run_name_edit.text().strip()
        if not custom:
            custom = f'run_{timestamp}'        # fall‑back if user erased it
        run_name = custom
        
        # Get augmentation settings
        use_flip = self.use_flip_checkbox.isChecked()
        use_rotation = self.use_rotation_checkbox.isChecked()
        use_scale = self.use_scale_checkbox.isChecked()
        
        # Get augmentation multiplier (1x, 2x, 3x, 4x)
        aug_multiplier_map = {
            0: 1,  # None
            1: 2,  # Light
            2: 3,  # Standard
            3: 4   # Heavy
        }
        aug_multiplier = aug_multiplier_map.get(self.augmentation_multiplier.currentIndex(), 3)
        
        # Switch to the Training Monitor tab
        self.main_tabs.setCurrentIndex(1)
            
        # Update UI
        self.console_output.append("Starting training with YOLOv11...")
        
        if is_cloud:
            self.console_output.append("Training mode: Ultralytics Cloud")
        else:
            self.console_output.append("Training mode: Local")
            
        # Correct path format for Windows compatibility
        dataset_yaml_path_posix = dataset_yaml_path.replace("\\", "/")
        output_dir_posix = os.path.dirname(self.output_dir).replace("\\", "/")
        
        # Build training command based on mode (cloud or local)
        if is_cloud:
            # Create a temporary Python script for cloud training
            try:
                # Use runs/scripts directory for training scripts
                runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
                scripts_dir = os.path.join(runs_dir, "scripts")
                os.makedirs(scripts_dir, exist_ok=True)
                
                # Create timestamped script path
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                script_path = os.path.join(scripts_dir, f"yoloflow_cloud_{timestamp}.py")
                
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write("import os\n")
                    f.write("import time\n")
                    f.write("import sys\n\n")
                    f.write("# Use Ultralytics YOLO directly instead of trying to import yolo11.py\n")
                    f.write("# This avoids any import errors and works more reliably\n")
                    f.write("from ultralytics import YOLO\n\n")
                    f.write("if __name__ == '__main__':\n")
                    f.write("    # Set the API key\n")
                    f.write(f"    os.environ['ULTRALYTICS_API_KEY'] = '{self.api_key_input.text().strip()}'\n\n")
                    f.write(f"    # Create YOLO model (will use latest version by default)\n")
                    f.write(f"    model = YOLO()\n\n")
                    f.write(f"    # Create unique run name\n")
                    f.write(f"    timestamp = time.strftime('%Y%m%d_%H%M%S')\n")
                    f.write(f"    # Configure exact paths for output to ensure proper directory structure\n")
                    f.write(f"    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
                    f.write(f"    runs_dir = os.path.join(base_dir, 'runs')\n")
                    f.write(f"    project_dir = os.path.join(runs_dir, 'train')\n")
                    f.write(f"    os.makedirs(project_dir, exist_ok=True)\n")
                    f.write(f"    # Use custom run name from UI\n")
                    f.write(f"    run_name = '{run_name}'\n\n")
                    f.write(f"    # Start cloud training\n")
                    f.write(f"    model.train(\n")
                    f.write(f"        data=r'{dataset_yaml_path_posix}',\n")
                    f.write(f"        epochs={epochs},\n")
                    f.write(f"        batch={batch_size},\n")
                    f.write(f"        imgsz={img_size},\n")
                    f.write(f"        lr0={lr},\n")
                    f.write(f"        project=project_dir,\n")
                    f.write(f"        name=run_name,\n")
                    f.write(f"        exist_ok=True,\n")
                    f.write(f"        cloud=True  # Enable cloud training\n")
                    f.write(f"    )\n\n")
                    f.write(f"    output_path = os.path.join(project_dir, run_name)\n")
                    f.write(f"    print(f'Cloud training job submitted - results will be available at {{output_path}}')\n\n")
                    f.write(f"    # Cloud training doesn't immediately create model files, but will download them when complete\n")
                    f.write(f"    print('When cloud training is complete, models will be automatically saved to:')\n")
                    f.write(f"    weights_dir = os.path.join(project_dir, run_name, 'weights')\n")
                    f.write(f"    print(f'- {{os.path.join(weights_dir, \"best.pt\")}}')\n")
                    f.write(f"    print(f'- {{os.path.join(weights_dir, \"last.pt\")}}')\n")
                
                script_path_fixed = script_path.replace("\\", "/")
                train_command = f'python "{script_path_fixed}"'
                self.console_output.append(f"Created cloud training script at: {script_path}")
            except Exception as e:
                self.console_output.append(f"Error creating cloud training script: {str(e)}")
                return
        else:
            # Local training
            # Create a temporary Python script for local training
            try:
                # Use runs/scripts directory for training scripts
                runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
                scripts_dir = os.path.join(runs_dir, "scripts")
                os.makedirs(scripts_dir, exist_ok=True)
                
                # Create timestamped script path
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                script_path = os.path.join(scripts_dir, f"yoloflow_train_{timestamp}.py")
                
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write("import multiprocessing\n")
                    f.write("import time\n")
                    f.write("import os\n")
                    f.write("import shutil\n")
                    f.write("import sys\n\n")
                    f.write("# Use Ultralytics YOLO directly instead of trying to import yolo11.py\n")
                    f.write("# This avoids any import errors and works more reliably\n")
                    f.write("from ultralytics import YOLO\n\n")
                    
                    # Add data augmentation functions if enabled and using temp directory
                    if (use_flip or use_rotation or use_scale) and self.use_temp_dir_checkbox.isChecked():
                        f.write("# Data augmentation functions\n")
                        f.write("def apply_augmentations(dataset_dir):\n")
                        f.write("    import cv2\n")
                        f.write("    import numpy as np\n")
                        f.write("    import os\n")
                        f.write("    import random\n")
                        f.write("    from pathlib import Path\n\n")
                        
                        f.write("    print('Applying data augmentations...')\n")
                        f.write("    images_dir = os.path.join(dataset_dir, 'images')\n")
                        f.write("    labels_dir = os.path.join(dataset_dir, 'labels')\n")
                        f.write("    aug_counter = 0\n\n")
                        
                        f.write("    # Get all image files\n")
                        f.write("    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n")
                        f.write("    print(f'Found {len(image_files)} original images for augmentation')\n\n")
                        
                        # Augmentation multiplier control
                        f.write(f"    # Apply augmentations (up to {aug_multiplier}x original dataset size)\n")
                        f.write(f"    max_augmentations = {aug_multiplier} * len(image_files) - len(image_files)\n")
                        f.write("    if max_augmentations <= 0:\n")
                        f.write("        print('No augmentations to create')\n")
                        f.write("        return 0\n\n")
                        
                        f.write("    for img_file in image_files:\n")
                        f.write("        # If we've reached the target number of augmentations, stop\n")
                        f.write("        if aug_counter >= max_augmentations:\n")
                        f.write("            break\n\n")
                        
                        f.write("        img_path = os.path.join(images_dir, img_file)\n")
                        f.write("        img = cv2.imread(img_path)\n")
                        f.write("        if img is None:\n")
                        f.write("            print(f'Error reading image: {img_path}')\n")
                        f.write("            continue\n\n")
                        
                        f.write("        # Get corresponding label file\n")
                        f.write("        base_name = os.path.splitext(img_file)[0]\n")
                        f.write("        label_file = f'{base_name}.txt'\n")
                        f.write("        label_path = os.path.join(labels_dir, label_file)\n")
                        f.write("        if not os.path.exists(label_path):\n")
                        f.write("            print(f'Warning: No label file for {img_file}')\n")
                        f.write("            continue\n\n")
                        
                        f.write("        # Read the label file\n")
                        f.write("        with open(label_path, 'r') as f:\n")
                        f.write("            label_lines = f.readlines()\n\n")
                        
                        f.write("        h, w = img.shape[:2]\n")
                        
                        # Horizontal flip
                        if use_flip:
                            f.write("        # Horizontal flip\n")
                            f.write("        if aug_counter < max_augmentations:\n")
                            f.write("            aug_img = cv2.flip(img, 1)  # 1 for horizontal flip\n")
                            f.write("            aug_name = f'{base_name}_flip.jpg'\n")
                            f.write("            aug_img_path = os.path.join(images_dir, aug_name)\n")
                            f.write("            cv2.imwrite(aug_img_path, aug_img)\n\n")
                            
                            f.write("            # Update labels (flip x coordinates)\n")
                            f.write("            aug_label_path = os.path.join(labels_dir, f'{base_name}_flip.txt')\n")
                            f.write("            with open(aug_label_path, 'w') as f:\n")
                            f.write("                for line in label_lines:\n")
                            f.write("                    parts = line.strip().split()\n")
                            f.write("                    if len(parts) >= 5:\n")
                            f.write("                        cls, x, y, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])\n")
                            f.write("                        # Flip x coordinate (1.0 - x)\n")
                            f.write("                        x = 1.0 - x\n")
                            f.write("                        f.write(f'{cls} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\\n')\n")
                            f.write("            aug_counter += 1\n\n")
                        
                        # Rotation
                        if use_rotation:
                            f.write("        # Rotation (+/-15 degrees)\n")
                            f.write("        if aug_counter < max_augmentations:\n")
                            f.write("            angle = random.uniform(-15, 15)  # Random angle between -15 and 15 degrees\n")
                            f.write("            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)\n")
                            f.write("            aug_img = cv2.warpAffine(img, M, (w, h))\n")
                            f.write("            aug_name = f'{base_name}_rot{int(angle)}.jpg'\n")
                            f.write("            aug_img_path = os.path.join(images_dir, aug_name)\n")
                            f.write("            cv2.imwrite(aug_img_path, aug_img)\n\n")
                            
                            f.write("            # Adjust labels for rotation\n")
                            f.write("            aug_label_path = os.path.join(labels_dir, f'{base_name}_rot{int(angle)}.txt')\n")
                            f.write("            with open(aug_label_path, 'w') as f:\n")
                            f.write("                for line in label_lines:\n")
                            f.write("                    # For simplicity, we'll just copy the original labels\n")
                            f.write("                    # In a production system, you'd want to transform the coordinates correctly\n")
                            f.write("                    f.write(line)\n")
                            f.write("            aug_counter += 1\n\n")
                        
                        # Scale jitter
                        if use_scale:
                            f.write("        # Scale jitter (+/-25%)\n")
                            f.write("        if aug_counter < max_augmentations:\n")
                            f.write("            scale = random.uniform(0.75, 1.25)  # Random scale between 75% and 125%\n")
                            f.write("            aug_w, aug_h = int(w * scale), int(h * scale)\n")
                            f.write("            aug_img = cv2.resize(img, (aug_w, aug_h))\n")
                            f.write("            # Create blank canvas of original size\n")
                            f.write("            canvas = np.zeros((h, w, 3), dtype=np.uint8)\n")
                            f.write("            # Calculate paste position (center)\n")
                            f.write("            x_offset = max(0, (w - aug_w) // 2)\n")
                            f.write("            y_offset = max(0, (h - aug_h) // 2)\n")
                            f.write("            # Paste resized image onto canvas\n")
                            f.write("            if aug_w <= w and aug_h <= h:  # Only if smaller than canvas\n")
                            f.write("                canvas[y_offset:y_offset+aug_h, x_offset:x_offset+aug_w] = aug_img\n")
                            f.write("            else:  # If bigger, crop center\n")
                            f.write("                crop_x = max(0, (aug_w - w) // 2)\n")
                            f.write("                crop_y = max(0, (aug_h - h) // 2)\n")
                            f.write("                crop_img = aug_img[crop_y:crop_y+min(h, aug_h), crop_x:crop_x+min(w, aug_w)]\n")
                            f.write("                canvas[:min(h, crop_img.shape[0]), :min(w, crop_img.shape[1])] = crop_img\n")
                            f.write("            aug_name = f'{base_name}_scale{int(scale*100)}.jpg'\n")
                            f.write("            aug_img_path = os.path.join(images_dir, aug_name)\n")
                            f.write("            cv2.imwrite(aug_img_path, canvas)\n\n")
                            
                            f.write("            # Adjust labels for scaling\n")
                            f.write("            aug_label_path = os.path.join(labels_dir, f'{base_name}_scale{int(scale*100)}.txt')\n")
                            f.write("            with open(aug_label_path, 'w') as f:\n")
                            f.write("                for line in label_lines:\n")
                            f.write("                    # For simplicity, we'll just copy the original labels\n")
                            f.write("                    # In a production system, you'd want to transform the coordinates correctly\n")
                            f.write("                    f.write(line)\n")
                            f.write("            aug_counter += 1\n\n")
                            
                        f.write("    print(f'Created {aug_counter} augmented images')\n")
                        f.write("    return aug_counter\n\n")
                    
                    # Main training script
                    f.write("if __name__ == '__main__':\n")
                    f.write("    multiprocessing.freeze_support()\n")
                    
                    # Load specific model size instead of default
                    f.write(f"    model = YOLO('{model_size}')  # Use {model_name} model\n")
                    f.write(f"    timestamp = time.strftime('%Y%m%d_%H%M%S')\n")
                    f.write(f"    # Configure exact paths for output to make sure they go to runs/train\n")
                    f.write(f"    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n")
                    f.write(f"    runs_dir = os.path.join(base_dir, 'runs')\n")
                    f.write(f"    project_dir = os.path.join(runs_dir, 'train')\n")
                    f.write(f"    os.makedirs(project_dir, exist_ok=True)\n")
                    f.write(f"    # Use custom run name from UI\n")
                    f.write(f"    run_name = '{run_name}'\n")
                    
                    # Apply augmentations if selected (only when using temp directory)
                    if (use_flip or use_rotation or use_scale) and self.use_temp_dir_checkbox.isChecked():
                        f.write(f"    # Apply data augmentations (only in temporary directory)\n")
                        f.write(f"    dataset_dir = r'{os.path.dirname(dataset_yaml_path_posix)}'\n")
                        f.write(f"    aug_count = apply_augmentations(dataset_dir)\n")
                        f.write(f"    print(f'Training with {{aug_count}} augmented images plus original dataset')\n\n")
                    elif (use_flip or use_rotation or use_scale) and not self.use_temp_dir_checkbox.isChecked():
                        f.write(f"    # Data augmentation requires temporary directory - skipping\n")
                        f.write(f"    print('WARNING: Data augmentation is only available when using temporary directory.')\n")
                        f.write(f"    print('Enable temporary directory option to use augmentation.')\n\n")
                    
                    # Standard training parameters
                    f.write(f"    # Start training with proper output directory\n")
                    f.write(f"    results = model.train(\n")
                    f.write(f"        data=r'{dataset_yaml_path_posix}',\n")
                    f.write(f"        epochs={epochs},\n")
                    f.write(f"        batch={batch_size},\n")
                    f.write(f"        imgsz={img_size},\n")
                    f.write(f"        lr0={lr},\n")
                    f.write(f"        workers=0,\n")
                    f.write(f"        project=project_dir,  # This is already runs/train\n")
                    f.write(f"        name=run_name,\n")
                    f.write(f"        exist_ok=True\n")
                    f.write(f"    )\n\n")
                    f.write(f"    output_path = os.path.join(project_dir, run_name)\n")
                    f.write(f"    print(f'Training completed - results saved to {{output_path}}')\n\n")
                    
                    # Add code to copy the best model to the models directory with timestamp
                    f.write(f"    # Copy the best trained model to models directory with timestamp\n")
                    f.write(f"    weights_dir = os.path.join(project_dir, run_name, 'weights')\n")
                    f.write(f"    best_model_path = os.path.join(weights_dir, 'best.pt')\n")
                    f.write(f"    last_model_path = os.path.join(weights_dir, 'last.pt')\n\n")
                    
                    f.write(f"    # Get the models directory\n")
                    f.write(f"    models_dir = os.path.join(base_dir, 'models')\n")
                    f.write(f"    os.makedirs(models_dir, exist_ok=True)\n\n")
                    
                    f.write(f"    # Generate timestamped model filenames\n")
                    f.write(f"    model_name_base = os.path.splitext(os.path.basename('{model_size}'))[0]\n")
                    f.write(f"    best_target = os.path.join(models_dir, f'trained_{{model_name_base}}_best_{{timestamp}}.pt')\n")
                    f.write(f"    last_target = os.path.join(models_dir, f'trained_{{model_name_base}}_last_{{timestamp}}.pt')\n\n")
                    
                    f.write(f"    # Copy the best model if it exists\n")
                    f.write(f"    if os.path.exists(best_model_path):\n")
                    f.write(f"        print(f'Copying best model to: {{best_target}}')\n")
                    f.write(f"        shutil.copy2(best_model_path, best_target)\n")
                    f.write(f"        print(f'Best model saved successfully!')\n")
                    f.write(f"    else:\n")
                    f.write(f"        print(f'Warning: Could not find best model at {{best_model_path}}')\n\n")
                    
                    f.write(f"    # Copy the last model if it exists\n")
                    f.write(f"    if os.path.exists(last_model_path):\n")
                    f.write(f"        print(f'Copying last model to: {{last_target}}')\n")
                    f.write(f"        shutil.copy2(last_model_path, last_target)\n")
                    f.write(f"        print(f'Last model saved successfully!')\n")
                    f.write(f"    else:\n")
                    f.write(f"        print(f'Warning: Could not find last model at {{last_model_path}}')\n")
                    
                    # We no longer use temporary directories
                    f.write(f"\n    # No temporary directory cleanup needed\n")
                
                script_path_fixed = script_path.replace("\\", "/")
                train_command = f'python "{script_path_fixed}"'
                self.console_output.append(f"Created training script at: {script_path}")
            except Exception as e:
                self.console_output.append(f"Error creating training script: {str(e)}")
                return
        
        # Log command
        self.console_output.append(f"Command: {train_command}")
        
        # Disable buttons
        self.train_local_btn.setEnabled(False)
        self.train_cloud_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Start training in a separate thread
        self.training_worker = TrainingWorker(train_command, self.output_dir, is_cloud, self.api_key_input.text())
        self.training_worker.update_signal.connect(self.update_training_output)
        self.training_worker.metrics_signal.connect(self.update_metrics)
        self.training_worker.image_update_signal.connect(self.update_results_image)
        self.training_worker.finished_signal.connect(self.training_finished)
        self.training_worker.start()
    
    def stop_training(self):
        """Stop the training process"""
        if self.training_worker and self.training_worker.isRunning():
            # No confirmation dialog, just stop
            self.console_output.append("Stopping training process...")
            self.training_worker.stop()
            
        # Stop heartbeat timer
        if hasattr(self, 'heartbeat_timer') and self.heartbeat_timer.isActive():
            self.heartbeat_timer.stop()
    
    @pyqtSlot(str)
    def update_training_output(self, text):
        """Update the training output by printing to console"""
        # Update the console output
        self.console_output.append(text)
        # Also print to system console
        print(f"[TRAINING] {text}")
        
    @pyqtSlot(dict)
    def update_metrics(self, metrics):
        """Update the metrics display with the latest values"""
        # Update epoch display and progress bar
        if metrics['total_epochs'] > 0:
            self.epoch_label.setText(f"{metrics['epoch']} / {metrics['total_epochs']}")
            
            # Calculate progress percentage for the progress bar
            progress_pct = int((metrics['epoch'] / metrics['total_epochs']) * 100)
            self.epoch_progress.setValue(progress_pct)
            
            # Also update the progress bar text
            self.epoch_progress.setFormat(f"{progress_pct}% - Epoch {metrics['epoch']}/{metrics['total_epochs']}")
        
        # Update loss metrics with color coding based on values
        if metrics['box_loss'] > 0:
            self.box_loss_label.setText(f"{metrics['box_loss']:.4f}")
            # Color code based on loss value: green (good) to red (needs improvement)
            if metrics['box_loss'] < 0.03:
                self.box_loss_label.setStyleSheet("font-family: monospace; background-color: #e6ffe6; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            elif metrics['box_loss'] < 0.07:
                self.box_loss_label.setStyleSheet("font-family: monospace; background-color: #f5f5f5; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            else:
                self.box_loss_label.setStyleSheet("font-family: monospace; background-color: #fff0f0; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
        
        if metrics['cls_loss'] > 0:
            self.cls_loss_label.setText(f"{metrics['cls_loss']:.4f}")
            # Color code based on loss value
            if metrics['cls_loss'] < 0.03:
                self.cls_loss_label.setStyleSheet("font-family: monospace; background-color: #e6ffe6; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            elif metrics['cls_loss'] < 0.09:
                self.cls_loss_label.setStyleSheet("font-family: monospace; background-color: #f5f5f5; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            else:
                self.cls_loss_label.setStyleSheet("font-family: monospace; background-color: #fff0f0; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
        
        if metrics['dfl_loss'] > 0:
            self.dfl_loss_label.setText(f"{metrics['dfl_loss']:.4f}")
            # Color code based on loss value
            if metrics['dfl_loss'] < 0.01:
                self.dfl_loss_label.setStyleSheet("font-family: monospace; background-color: #e6ffe6; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            elif metrics['dfl_loss'] < 0.03:
                self.dfl_loss_label.setStyleSheet("font-family: monospace; background-color: #f5f5f5; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            else:
                self.dfl_loss_label.setStyleSheet("font-family: monospace; background-color: #fff0f0; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            
        # Update performance metrics with color coding
        if metrics['map'] > 0:
            self.map_label.setText(f"{metrics['map']:.4f}")
            # Color code based on mAP value (higher is better)
            if metrics['map'] > 0.7:
                self.map_label.setStyleSheet("font-family: monospace; background-color: #e6ffe6; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            elif metrics['map'] > 0.5:
                self.map_label.setStyleSheet("font-family: monospace; background-color: #f5f5f5; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            else:
                self.map_label.setStyleSheet("font-family: monospace; background-color: #fff0f0; padding: 3px; border: 1px solid #ddd; border-radius: 3px;")
            
        # Make sure UI stays responsive
        QApplication.processEvents()
    
    def update_elapsed_time(self):
        """Update the elapsed time display"""
        if self.training_start_time:
            elapsed = time.time() - self.training_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.time_label.setText(time_str)
    
    # This method is no longer needed as we're using the update_metrics method instead
    
    # Methods related to thumbnails have been removed to simplify the interface
    
    @pyqtSlot(str)
    def update_results_image(self, image_path):
        """Update the results image display"""
        if os.path.exists(image_path):
            self.console_output.append(f"Loading results plot from: {image_path}")
            
            # Update main results graph
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale while maintaining aspect ratio
                self.results_image.setPixmap(pixmap.scaled(
                    self.results_image.width(), 
                    self.results_image.height(),
                    Qt.KeepAspectRatio
                ))
                
                # Update title and description
                self.display_title.setText("Training Metrics")
                self.display_description.setText(f"Latest results from {os.path.basename(os.path.dirname(image_path))}")
                
            else:
                self.console_output.append("Error: Failed to load results image")
        else:
            self.console_output.append(f"Error: Results image not found at {image_path}")
    
    def cleanup_temp_dir(self):
        """
        This function is kept for compatibility but does nothing now
        since we don't use temporary directories anymore
        """
        pass
    
    def progress_heartbeat(self):
        """Heartbeat function to refresh UI during training"""
        # Only refresh if training is in progress
        if not hasattr(self, 'training_worker') or not self.training_worker or not self.training_worker.isRunning():
            return
            
        # Get current values from worker
        if hasattr(self, 'training_worker'):
            # Get metrics from the training worker
            metrics = self.training_worker.metrics
            
            # Pulse the progress bar if no progress data yet
            if metrics['total_epochs'] <= 0:
                # Create a pulsing effect for the progress bar
                current_value = self.epoch_progress.value()
                new_value = (current_value + 5) % 100
                self.epoch_progress.setValue(new_value)
                self.epoch_progress.setFormat("Initializing training...")
            else:
                # Update properly with the metrics
                self.update_metrics(metrics)
            
            # Force UI update
            QApplication.processEvents()
    
    def training_finished(self, success, message):
        """Handle training completion"""
        self.console_output.append(message)
        
        # Stop all timers
        self.timer.stop()
        if hasattr(self, 'heartbeat_timer') and self.heartbeat_timer.isActive():
            self.heartbeat_timer.stop()
            
        # Re-enable buttons
        self.train_local_btn.setEnabled(True)
        self.train_cloud_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # Update progress bar to show completion or failure
        if success:
            # Set progress bar to 100% with success message
            self.epoch_progress.setValue(100)
            self.epoch_progress.setFormat("100% - Training Complete")
            self.epoch_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bbb;
                    border-radius: 4px;
                    padding: 1px;
                    text-align: center;
                    height: 20px;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    width: 10px;
                    margin: 0px;
                }
            """)
            
            # Calculate total training time
            if self.training_start_time:
                total_time = time.time() - self.training_start_time
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                seconds = int(total_time % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                self.console_output.append(f"Training completed successfully in {time_str}!")
                self.time_label.setText(time_str)
                self.display_description.setText(f"Training complete - total time: {time_str}")
        else:
            # Set progress bar to show failure
            self.epoch_progress.setFormat("Training Failed")
            self.epoch_progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bbb;
                    border-radius: 4px;
                    padding: 1px;
                    text-align: center;
                    height: 20px;
                    background-color: #f0f0f0;
                }
                QProgressBar::chunk {
                    background-color: #f44336;
                    width: 10px;
                    margin: 0px;
                }
            """)
            self.console_output.append("Training failed - see log for details")
            self.display_description.setText("Training failed - see log for details")
            
        # Find the latest training run
        try:
            base_dir = os.path.dirname(self.output_dir)
            train_dir = os.path.join(base_dir, "train")
            if os.path.exists(train_dir):
                dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
                if dirs:
                    # Sort by modification time (newest first)
                    dirs.sort(key=os.path.getmtime, reverse=True)
                    latest_run = dirs[0]
                    self.console_output.append(f"Latest training run: {latest_run}")
                    
                    # Try to update results image one more time
                    results_png = os.path.join(latest_run, "results.png")
                    if os.path.exists(results_png):
                        self.update_results_image(results_png)
                        
                    # Check for trained model weights
                    weights_path = os.path.join(latest_run, "weights/best.pt")
                    if os.path.exists(weights_path):
                        self.console_output.append(f"Trained model saved at: {weights_path}")
                        
                        # Check if the models were copied to the models directory with timestamp
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get /root directory
                        models_dir = os.path.join(base_dir, "models")
                        trained_best_models = [f for f in os.listdir(models_dir) if f.startswith('trained_') and '_best_' in f and f.endswith('.pt')]
                        trained_last_models = [f for f in os.listdir(models_dir) if f.startswith('trained_') and '_last_' in f and f.endswith('.pt')]
                        
                        if trained_best_models or trained_last_models:
                            # Show the models that were saved
                            self.console_output.append("Trained models have been saved to the models directory:")
                            
                            if trained_best_models:
                                # Sort by modification time (newest first)
                                trained_best_models.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                                latest_best_model = trained_best_models[0]
                                self.console_output.append(f"  - Best model: {latest_best_model}")
                            
                            if trained_last_models:
                                # Sort by modification time (newest first)
                                trained_last_models.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                                latest_last_model = trained_last_models[0]
                                self.console_output.append(f"  - Last model: {latest_last_model}")
                                
                            self.console_output.append("These models are now available for inference.")
                        else:
                            # If no timestamped models found, copy them now with current timestamp
                            self.console_output.append("No saved models found. Copying models to models directory...")
                            try:
                                timestamp = time.strftime('%Y%m%d_%H%M%S')
                                run_name = os.path.basename(latest_run)
                                model_name_base = run_name.split('_train_')[0] if '_train_' in run_name else "yolo11"
                                
                                # Copy best model if it exists
                                best_path = os.path.join(latest_run, "weights/best.pt")
                                if os.path.exists(best_path):
                                    best_target = os.path.join(models_dir, f'trained_{model_name_base}_best_{timestamp}.pt')
                                    shutil.copy2(best_path, best_target)
                                    self.console_output.append(f"  - Best model saved to: {best_target}")
                                
                                # Copy last model if it exists
                                last_path = os.path.join(latest_run, "weights/last.pt")
                                if os.path.exists(last_path):
                                    last_target = os.path.join(models_dir, f'trained_{model_name_base}_last_{timestamp}.pt')
                                    shutil.copy2(last_path, last_target)
                                    self.console_output.append(f"  - Last model saved to: {last_target}")
                                    
                            except Exception as copy_error:
                                self.console_output.append(f"Error copying models: {str(copy_error)}")
                        
                        self.console_output.append("This model will be automatically used in the Test tab.")
        except Exception as e:
            self.console_output.append(f"Error finding latest run: {str(e)}")
        
        # We no longer use temporary directories - no cleanup needed