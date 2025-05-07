import os
import cv2
import numpy as np
import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QListWidget, QSplitter, QMessageBox, QMenu, QApplication,
                             QProgressDialog, QComboBox, QGroupBox, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

class ReviewTab(QWidget):
    data_updated = pyqtSignal(str)  # Signal emitted when data is updated
    
    def __init__(self):
        super().__init__()
        
        # Data paths
        self.data_dir = ""
        self.images_dir = ""
        self.labels_dir = ""
        
        # Current item being viewed
        self.current_image_path = ""
        self.current_label_path = ""
        self.current_bbox = None       # Single bounding box (x,y,w,h)
        self.current_class_id = 0      # Class ID for the current bbox
        
        # List of class names
        self.class_list = []
        
        # Editing mode - always enabled by default
        self.edit_mode = True
        self.edit_start_pos = None
        self.resizing = False
        self.resize_handle = None  # Which handle is being dragged (tl, tr, bl, br)
        self.handle_size = 14      # Size of resize handles in pixels
        self.original_bbox = None  # Store original bbox during resize
        
        # Selection management
        self.selected_items = []
        self.last_selected_item = None
        
        # Caching
        self.cached_image = None
        
        # Status label for messages
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #555;")
        self.status_label.setAlignment(Qt.AlignLeft)
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_file_list)
        self.refresh_timer.setInterval(2000)  # Check for new files every 2 seconds
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        main_layout = QVBoxLayout()
        
        # Create a splitter for image view and data list
        splitter = QSplitter(Qt.Horizontal)
        
        # Image view panel
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # Image display
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setMinimumSize(640, 800)  # 80% taller (480 -> 800)
        self.image_view.setStyleSheet("border: 1px solid #ccc;")
        image_layout.addWidget(self.image_view)
        
        # Image controls with styled buttons
        image_controls = QHBoxLayout()
        
        # Standard button style
        button_style = """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 12px 24px;
                font-weight: bold;
                font-size: 16px;
                min-height: 50px;
                min-width: 180px;
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
        """
        
        # Navigation buttons
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.clear_all_btn = QPushButton("Clear All")
        
        # Apply styles for large navigation buttons only
        self.prev_btn.setStyleSheet(button_style)
        self.next_btn.setStyleSheet(button_style)
        
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        
        # Add spacer to center the navigation buttons
        image_controls.addStretch(1)
        image_controls.addWidget(self.prev_btn)
        image_controls.addWidget(self.next_btn)
        image_controls.addStretch(1)
        image_layout.addLayout(image_controls)
        
        # Data panel
        data_panel = QWidget()
        data_layout = QVBoxLayout(data_panel)
        
        # File list with multi-selection support
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)  # Enable multi-selection
        self.file_list.itemClicked.connect(self.load_selected_item)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)
        selection_hint = QLabel("Captured Images: (Shift-click for range, Ctrl-click for multi-select)")
        selection_hint.setStyleSheet("color: #666; font-size: 9pt;")  # Smaller, gray hint text
        data_layout.addWidget(QLabel("Captured Images:"))
        data_layout.addWidget(selection_hint)
        data_layout.addWidget(self.file_list)
        
        # Action buttons with same styling as image controls
        action_buttons = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected")
        self.save_changes_btn = QPushButton("Save Changes")
        
        # Re-use same button style from above, just with color changes
        self.delete_btn.setStyleSheet(button_style.replace("#2196F3", "#f44336"))  # Red for delete
        self.save_changes_btn.setStyleSheet(button_style.replace("#2196F3", "#4CAF50"))  # Green for save
        
        self.delete_btn.clicked.connect(self.delete_selected)
        self.save_changes_btn.clicked.connect(lambda: self.save_changes(show_message=True))
        
        action_buttons.addWidget(self.delete_btn)
        action_buttons.addWidget(self.save_changes_btn)
        data_layout.addLayout(action_buttons)
        
        # Add panels to splitter
        splitter.addWidget(image_panel)
        splitter.addWidget(data_panel)
        splitter.setSizes([700, 300])  # Initial size allocation
        
        main_layout.addWidget(splitter)
        
        # Add status bar at the bottom
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.status_label)
        main_layout.addLayout(bottom_layout)
        
        self.setLayout(main_layout)
        
        # Connect mouse events for editing
        self.image_view.setMouseTracking(True)
        self.image_view.mousePressEvent = self.mouse_press_event
        self.image_view.mouseMoveEvent = self.mouse_move_event
        self.image_view.mouseReleaseEvent = self.mouse_release_event
        self.image_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_view.customContextMenuRequested.connect(self.show_box_context_menu)
    
    def load_data(self, data_dir):
        """Load data from the specified directory"""
        if not data_dir:
            print("Error: No data directory provided")
            return
            
        print(f"Loading data from: {data_dir}")
        self.data_dir = data_dir
        
        # Use data directory directly
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        print(f"Using data directory for images and labels: {data_dir}")
        
        # Show loading dialog
        self.status_label.setText("Initializing...")
        QApplication.processEvents()
        
        # Use a timer to show loading dialog before heavy operations start
        QTimer.singleShot(100, lambda: self._continue_loading_data())
    
    def _continue_loading_data(self):
        """Continue loading data after UI has updated"""
        # Loading indicator in the status bar
        dots = 0
        timer = QTimer()
        
        def update_dots():
            nonlocal dots
            dots = (dots + 1) % 4
            dot_str = "." * dots + " " * (3 - dots)
            self.status_label.setText(f"Loading{dot_str}")
        
        # Start loading indicator
        timer.timeout.connect(update_dots)
        timer.start(200)
        QApplication.processEvents()
        
        # Check if directories exist
        if not os.path.exists(self.images_dir):
            print(f"Warning: Images directory does not exist: {self.images_dir}")
            os.makedirs(self.images_dir, exist_ok=True)
            
        if not os.path.exists(self.labels_dir):
            print(f"Warning: Labels directory does not exist: {self.labels_dir}")
            os.makedirs(self.labels_dir, exist_ok=True)
        
        # Load classes from classes.txt if exists
        classes_file = os.path.join(self.data_dir, "classes.txt")
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r') as f:
                    self.class_list = [line.strip() for line in f.readlines()]
                logging.debug(f"Loaded {len(self.class_list)} classes: {', '.join(self.class_list)}")
                
            except Exception as e:
                print(f"Error loading classes: {str(e)}")
                self.class_list = ["unknown"]
        else:
            print(f"Warning: Classes file not found: {classes_file}")
            self.class_list = ["unknown"]
        
        # Stop the loading animation
        timer.stop()
        
        # This now happens in a separate method to allow UI to update
        self.refresh_file_list()
        
        # Start the auto-refresh timer
        self.refresh_timer.start()
    
    def refresh_file_list(self):
        """Refresh the list of files in the images directory"""
        # Check if refreshing is needed by comparing file counts
        if not hasattr(self, 'last_file_count'):
            self.last_file_count = -1  # Initialize to force first refresh
            
        # Always ensure directories exist
        if not os.path.exists(self.images_dir):
            print(f"Cannot refresh file list: Images directory does not exist: {self.images_dir}")
            # Create the directory if it doesn't exist
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.labels_dir, exist_ok=True)
            print(f"Created missing directories: {self.images_dir} and {self.labels_dir}")
            
        # Check if files have changed before doing expensive UI refresh
        try:
            # Quick check - compare file count
            current_files = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            current_file_count = len(current_files)
            
            # If file count hasn't changed, skip refresh unless forced
            if current_file_count == self.last_file_count and self.last_file_count >= 0 and self.file_list.count() > 0:
                # Only update the last refresh time, no UI update needed
                return
                
            # Store current selection if any
            current_selection = None
            if self.file_list.currentItem():
                current_selection = self.file_list.currentItem().text()
            
            # Now do the full refresh
            self.file_list.clear()
            QApplication.processEvents()  # Process UI events to show cleared list
            
            # Update the last file count
            self.last_file_count = current_file_count
            
        except Exception as e:
            print(f"Error checking for file changes: {str(e)}")
            # If there's an error, do a full refresh
            self.file_list.clear()
            QApplication.processEvents()
            
        try:
            # Animated loading indicator in status bar
            dots = 0
            timer = QTimer()
            
            def update_dots():
                nonlocal dots
                dots = (dots + 1) % 4
                dot_str = "." * dots + " " * (3 - dots)
                self.status_label.setText(f"Loading images{dot_str}")
            
            # Start loading animation
            timer.timeout.connect(update_dots)
            timer.start(200)
            QApplication.processEvents()
            
            files = os.listdir(self.images_dir)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_files.sort()
            
            print(f"Found {len(image_files)} images in {self.images_dir}")
            
            # Disable sorting to improve performance during batch add
            self.file_list.setSortingEnabled(False)
            
            # Add images in batches for better UI responsiveness
            batch_size = 20
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i+batch_size]
                for img_file in batch:
                    self.file_list.addItem(img_file)
                QApplication.processEvents()  # Process UI events after each batch
                # Don't update status here - let the timer handle it for animation
            
            # Check if corresponding label files exist
            missing_labels = []
            for img_file in image_files:
                label_file = os.path.splitext(img_file)[0] + ".txt"
                label_path = os.path.join(self.labels_dir, label_file)
                if not os.path.exists(label_path):
                    missing_labels.append(img_file)
                    
            if missing_labels:
                print(f"Warning: Missing {len(missing_labels)} label files")
            
            # Stop the loading animation
            timer.stop()
            
            if self.file_list.count() > 0:
                # Try to restore previous selection if it still exists
                if current_selection:
                    # Find the item in the list
                    for i in range(self.file_list.count()):
                        if self.file_list.item(i).text() == current_selection:
                            self.file_list.setCurrentRow(i)
                            self.load_selected_item(self.file_list.currentItem())
                            break
                    else:
                        # If not found, select the first item
                        self.file_list.setCurrentRow(0)
                        self.load_selected_item(self.file_list.currentItem())
                else:
                    # No previous selection, select the first item
                    self.file_list.setCurrentRow(0)
                    self.load_selected_item(self.file_list.currentItem())
                
                status_text = f"Loaded {self.file_list.count()} images"
                if missing_labels:
                    status_text += f" ({len(missing_labels)} missing labels)"
                self.status_label.setText(status_text)
            else:
                print("No images found in directory")
                self.status_label.setText("No images found")
                self.image_view.clear()
        except Exception as e:
            print(f"Error refreshing file list: {str(e)}")
            self.status_label.setText(f"Error loading files: {str(e)}")
    
    def load_selected_item(self, item):
        """Load the selected image and its corresponding label"""
        if item is None:
            return
        
        # Handle multi-selection with modifier keys
        modifiers = QApplication.keyboardModifiers()
        
        # If Ctrl is pressed, toggle selection of this item
        if modifiers & Qt.ControlModifier:
            if item in self.selected_items:
                self.selected_items.remove(item)
                item.setSelected(False)
            else:
                self.selected_items.append(item)
                self.last_selected_item = item
        # If Shift is pressed, select range from last selected to current
        elif modifiers & Qt.ShiftModifier and self.last_selected_item:
            start_idx = self.file_list.row(self.last_selected_item)
            end_idx = self.file_list.row(item)
            min_idx = min(start_idx, end_idx)
            max_idx = max(start_idx, end_idx)
            
            self.selected_items = []
            for idx in range(min_idx, max_idx + 1):
                item_at_idx = self.file_list.item(idx)
                self.selected_items.append(item_at_idx)
                item_at_idx.setSelected(True)
                
            self.last_selected_item = item
        # Process QListWidget's selection (supports Shift/Ctrl selection)
        else:
            # Get all selected items from QListWidget
            selected_items = self.file_list.selectedItems()
            
            # If this item isn't in the current selection, clear and select just this one
            if item not in selected_items:
                self.file_list.clearSelection()
                item.setSelected(True)
                self.selected_items = [item]
            else:
                # Otherwise, keep the existing multiple selection
                self.selected_items = selected_items
                
            self.last_selected_item = item
        
        # Process selected item for display
        image_filename = item.text()
        self.current_image_path = os.path.join(self.images_dir, image_filename)
        
        # Determine label path (same filename but with .txt extension)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        self.current_label_path = os.path.join(self.labels_dir, label_filename)
        
        # Clear the cached image when loading a new item
        self.cached_image = None
        
        # Load image
        self.display_image(load_from_file=True)
    
    def display_image(self, load_from_file=True):
        """Display the current image with bounding box"""
        if not os.path.exists(self.current_image_path):
            self.image_view.clear()
            return
            
        # Load image (or use cached version during editing)
        if self.cached_image is None:
            image = cv2.imread(self.current_image_path)
            self.cached_image = image.copy() if image is not None else None
        else:
            image = self.cached_image.copy()
            
        if image is None:
            self.image_view.clear()
            return
            
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Only load bounding box from file if requested (and if we're not in the middle of editing)
        # This prevents the bounding box from being reset during dragging
        if load_from_file and self.edit_start_pos is None:
            self.current_bbox = None
            self.current_class_id = 0
            
            if os.path.exists(self.current_label_path):
                with open(self.current_label_path) as f:
                    lines = f.readlines()
                    if lines and len(lines) > 0:  # Take only the first line/box
                        parts = lines[0].split()
                        if len(parts) >= 5:
                            cls, xc, yc, w, h = map(float, parts)
                            x = int((xc - w/2) * img_width)
                            y = int((yc - h/2) * img_height)
                            self.current_bbox = (x, y, int(w*img_width), int(h*img_height))
                            self.current_class_id = int(cls)
        
        # Create a copy of the image for drawing
        display_img = image.copy()
        
        # Draw the bounding box if it exists
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            color = (0, 255, 0)  # Green for the bounding box
            cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
            
            # Add class label if available
            if 0 <= self.current_class_id < len(self.class_list):
                class_name = self.class_list[self.current_class_id]
                cv2.putText(display_img, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw corner handles as filled squares (top-left, top-right, bottom-left, bottom-right)
            for cx, cy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
                cv2.rectangle(display_img,
                            (cx-self.handle_size, cy-self.handle_size),
                            (cx+self.handle_size, cy+self.handle_size),
                            (0, 0, 255), -1)
        
        # Convert image for display
        rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale while maintaining aspect ratio
        self.image_view.setPixmap(pixmap.scaled(self.image_view.size(), Qt.KeepAspectRatio))
    
    def show_previous(self):
        """Show the previous image in the list"""
        current_row = self.file_list.currentRow()
        if current_row > 0:
            self.file_list.setCurrentRow(current_row - 1)
            self.load_selected_item(self.file_list.currentItem())
    
    def show_next(self):
        """Show the next image in the list"""
        current_row = self.file_list.currentRow()
        if current_row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(current_row + 1)
            self.load_selected_item(self.file_list.currentItem())
    
    def clear_all_data(self):
        """Clear all images and labels after confirmation"""
        if self.file_list.count() == 0:
            self.status_label.setText("Status: No data to clear")
            return
            
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Clear All Data",
            "Are you sure you want to delete ALL images and labels? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Delete all files in images and labels directories
                for dir_path in [self.images_dir, self.labels_dir]:
                    if os.path.exists(dir_path):
                        for filename in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, filename)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                
                # Clear file list and displayed image
                self.file_list.clear()
                self.image_view.clear()
                self.current_image_path = ""
                self.current_label_path = ""
                self.current_bbox = None
                
                # Signal that data has been updated
                self.data_updated.emit(self.data_dir)
                
                self.status_label.setText("Status: All data cleared successfully")
            except Exception as e:
                self.status_label.setText(f"Error: Failed to clear data - {str(e)}")
                print(f"Error clearing data: {str(e)}")
        else:
            self.status_label.setText("Status: Clear operation canceled")
    
    # Edit mode is always enabled, so this function not needed
    
    def detect_handle(self, x, y):
        """Detect if the given position is on a resize handle, returns handle name"""
        if not self.current_bbox:
            return None
            
        threshold = self.handle_size + 8  # Increased threshold for easier selection
        
        bbox_x, bbox_y, bbox_w, bbox_h = self.current_bbox
        
        # Check corner handles
        if abs(x - bbox_x) < threshold and abs(y - bbox_y) < threshold:
            return "tl"  # top-left
        if abs(x - (bbox_x + bbox_w)) < threshold and abs(y - bbox_y) < threshold:
            return "tr"  # top-right
        if abs(x - bbox_x) < threshold and abs(y - (bbox_y + bbox_h)) < threshold:
            return "bl"  # bottom-left
        if abs(x - (bbox_x + bbox_w)) < threshold and abs(y - (bbox_y + bbox_h)) < threshold:
            return "br"  # bottom-right
        
        # Check if mouse is inside the box
        if (bbox_x <= x <= bbox_x + bbox_w) and (bbox_y <= y <= bbox_y + bbox_h):
            return "inside"  # inside the box
                
        return None
        
    def mouse_press_event(self, event):
        """Handle mouse press events for bounding box editing"""
        if not self.edit_mode or not hasattr(self, 'image_view'):
            return
            
        # Get click position relative to the image (accounting for scaling)
        pixmap = self.image_view.pixmap()
        if pixmap is None or pixmap.isNull():
            return
            
        # Calculate scaling and offset
        scaled_size = pixmap.size()
        widget_size = self.image_view.size()
        offset_x = (widget_size.width() - scaled_size.width()) // 2
        offset_y = (widget_size.height() - scaled_size.height()) // 2
        
        # Calculate position in the scaled image
        pos_x = event.pos().x() - offset_x
        pos_y = event.pos().y() - offset_y
        
        # Check if click is within the image bounds
        if 0 <= pos_x < scaled_size.width() and 0 <= pos_y < scaled_size.height():
            # Convert to original image coordinates
            if self.cached_image is None:
                image = cv2.imread(self.current_image_path)
                if image is None:
                    return
                self.cached_image = image.copy()
            
            img_height, img_width = self.cached_image.shape[:2]
            scale_x = img_width / scaled_size.width()
            scale_y = img_height / scaled_size.height()
            
            original_x = int(pos_x * scale_x)
            original_y = int(pos_y * scale_y)
            
            print(f"Mouse pressed at ({original_x}, {original_y})")
            
            # Check if we're clicking on a handle
            if self.current_bbox:
                handle = self.detect_handle(original_x, original_y)
                
                if handle and handle != "inside":
                    # Clicked on a handle - start resizing
                    self.resizing = True
                    self.resize_handle = handle
                    self.edit_start_pos = (original_x, original_y)
                    self.original_bbox = self.current_bbox  # Store original bbox
                    return
                elif handle == "inside":
                    # Clicked inside the box - just select it
                    self.resizing = False
                    self.resize_handle = None
                    self.edit_start_pos = None  # Don't start drawing
                    self.display_image(load_from_file=False)  # Just redraw to show selection
                    return
            
            # If not on a handle or no box exists, start drawing a new box
            self.current_bbox = (original_x, original_y, 0, 0)
            self.resizing = True
            self.resize_handle = "br"  # Start resizing from bottom-right
            self.edit_start_pos = (original_x, original_y)
            
            # Redraw the image to show the new box
            self.display_image(load_from_file=False)
    
    def mouse_move_event(self, event):
        """Handle mouse move events for bounding box editing"""
        if not self.edit_mode or self.edit_start_pos is None:
            return
            
        # Get current position relative to the image
        pixmap = self.image_view.pixmap()
        if pixmap is None or pixmap.isNull():
            return
            
        # Calculate scaling and offset
        scaled_size = pixmap.size()
        widget_size = self.image_view.size()
        offset_x = (widget_size.width() - scaled_size.width()) // 2
        offset_y = (widget_size.height() - scaled_size.height()) // 2
        
        # Calculate position in the scaled image
        pos_x = max(0, min(event.pos().x() - offset_x, scaled_size.width() - 1))
        pos_y = max(0, min(event.pos().y() - offset_y, scaled_size.height() - 1))
        
        # Convert to original image coordinates
        if self.cached_image is None:
            self.cached_image = cv2.imread(self.current_image_path)
        
        if self.cached_image is None:
            return
            
        img_height, img_width = self.cached_image.shape[:2]
        scale_x = img_width / scaled_size.width()
        scale_y = img_height / scaled_size.height()
        
        current_x = int(pos_x * scale_x)
        current_y = int(pos_y * scale_y)
        
        # Handle resizing if a handle is being dragged
        if self.resizing and self.resize_handle and self.edit_start_pos and self.current_bbox:
            print(f"Resizing with handle: {self.resize_handle} to position ({current_x}, {current_y})")
            
            # Get original bounds
            x, y, w, h = self.current_bbox
            right = x + w
            bottom = y + h
            
            # Update based on handle being dragged
            if self.resize_handle == "tl":  # Top-left
                new_x = current_x
                new_y = current_y
                new_w = right - new_x
                new_h = bottom - new_y
            elif self.resize_handle == "tr":  # Top-right
                new_x = x
                new_y = current_y
                new_w = current_x - x
                new_h = bottom - new_y
            elif self.resize_handle == "bl":  # Bottom-left
                new_x = current_x
                new_y = y
                new_w = right - new_x
                new_h = current_y - y
            elif self.resize_handle == "br":  # Bottom-right
                new_x = x
                new_y = y
                new_w = current_x - x
                new_h = current_y - y
            else:
                return
            
            # Ensure minimum size
            if new_w > 10 and new_h > 10:
                self.current_bbox = (new_x, new_y, new_w, new_h)
                print(f"Updated bbox: {self.current_bbox}")
            
        # No else clause needed - removed drawing new box on drag since we use New Box button
        
        self.display_image(load_from_file=False)  # Redraw without reloading from file
    
    def mouse_release_event(self, event):
        """Handle mouse release events for bounding box editing"""
        if self.edit_start_pos is None and not self.resizing:
            return
            
        # Finalize the bounding box
        prev_handle = self.resize_handle
        self.edit_start_pos = None
        self.resizing = False
        self.resize_handle = None
        
        # Ensure the current bounding box has a minimum size
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            if w < 10 or h < 10:  # Minimum size threshold
                self.status_label.setText("Status: Bounding box too small (min 10px)")
                self.current_bbox = None  # Remove the too-small box
                self.display_image(load_from_file=False)  # Redraw without reloading
                return
            
            # Auto-save all changes
            success = self.save_changes(show_message=False)  # Silent save without popup message
            if success:
                self.status_label.setText("Status: Bounding box updated and saved")
            else:
                self.status_label.setText("Status: Error saving bounding box data")
    
    def save_changes(self, show_message=True):
        """Save the bounding box to the label file"""
        if not self.current_bbox or not os.path.exists(self.current_image_path):
            return False
            
        # Get image dimensions for normalization
        if self.cached_image is None:
            image = cv2.imread(self.current_image_path)
        else:
            image = self.cached_image.copy()
            
        if image is None:
            return False
            
        img_height, img_width = image.shape[:2]
        
        try:
            # Write to label file
            with open(self.current_label_path, "w") as f:
                x, y, w, h = self.current_bbox
                cls = self.current_class_id
                
                # Convert to normalized center format
                xc = (x + w/2) / img_width
                yc = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                f.write(f"{cls} {xc:.6f} {yc:.6f} {norm_width:.6f} {norm_height:.6f}\n")
            
            # Verify file was written
            if not os.path.exists(self.current_label_path):
                print(f"Warning: Label file was not created: {self.current_label_path}")
                return False
                
            # Only update status if requested (not when auto-saving)
            if show_message:
                self.status_label.setText("Status: Bounding box saved successfully")
            
            # Signal that data has been updated
            self.data_updated.emit(self.data_dir)
            return True
            
        except Exception as e:
            print(f"Error saving label file: {str(e)}")
            if show_message:
                self.status_label.setText(f"Error: Failed to save bounding boxes - {str(e)}")
            return False
    
    def delete_selected(self):
        """Delete the selected images and their labels"""
        if not self.selected_items:
            self.status_label.setText("Status: No items selected for deletion")
            return
            
        # Show confirmation dialog if multiple items selected
        if len(self.selected_items) > 1:
            reply = QMessageBox.question(
                self,
                "Delete Multiple Items",
                f"Are you sure you want to delete {len(self.selected_items)} selected items?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self.status_label.setText("Status: Deletion cancelled")
                return
            
        # Use selected_items list instead of just current item
        try:
            # Keep track of rows to delete
            rows_to_delete = sorted([self.file_list.row(item) for item in self.selected_items], reverse=True)
            
            # Delete all selected items
            for item in self.selected_items:
                # Get file paths
                image_filename = item.text()
                image_path = os.path.join(self.images_dir, image_filename)
                label_filename = os.path.splitext(image_filename)[0] + ".txt"
                label_path = os.path.join(self.labels_dir, label_filename)
                
                # Log file paths before deletion
                print(f"Deleting image: {image_path}")
                print(f"Deleting label: {label_path}")
                
                # Delete files
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"Image deleted: {image_path}")
                else:
                    print(f"Image not found: {image_path}")
                    
                if os.path.exists(label_path):
                    os.remove(label_path)
                    print(f"Label deleted: {label_path}")
                else:
                    print(f"Label not found: {label_path}")
                    
                # Remove from list
                row = self.file_list.row(item)
                self.file_list.takeItem(row)
            
            # Clear selection list
            self.selected_items = []
            self.last_selected_item = None
            
            # Show next available item
            if self.file_list.count() > 0:
                next_row = min(rows_to_delete[-1], self.file_list.count() - 1)
                self.file_list.setCurrentRow(next_row)
                self.load_selected_item(self.file_list.currentItem())
            else:
                self.image_view.clear()
                self.current_image_path = ""
                self.current_label_path = ""
                self.current_bbox = None
            
            # Signal that data has been updated
            self.data_updated.emit(self.data_dir)
            
            self.status_label.setText(f"Status: Deleted {len(rows_to_delete)} items")
            
        except Exception as e:
            self.status_label.setText(f"Error: Failed to delete files - {str(e)}")
            print(f"Error deleting files: {str(e)}")
    
    def show_context_menu(self, position):
        """Show context menu for the file list"""
        context_menu = QMenu(self)
        
        # Add actions
        view_action = context_menu.addAction("View")
        delete_action = context_menu.addAction("Delete")
        
        # Get selected action
        action = context_menu.exec_(self.file_list.mapToGlobal(position))
        
        # Handle action
        if action == view_action:
            self.load_selected_item(self.file_list.currentItem())
        elif action == delete_action:
            self.delete_selected()
    
    def show_box_context_menu(self, event):
        """Show context menu for active bounding box"""
        if not self.current_bbox:
            return
            
        # Create context menu
        context_menu = QMenu(self)
        delete_box_action = context_menu.addAction("Delete Box")
        
        # Get global position for the menu
        pos = self.image_view.mapToGlobal(event.pos())
        
        # Show menu and get selected action
        action = context_menu.exec_(pos)
        
        # Handle action
        if action == delete_box_action:
            # Remove the box
            self.current_bbox = None
            
            # Delete the label file if it exists
            if os.path.exists(self.current_label_path):
                try:
                    os.remove(self.current_label_path)
                    self.status_label.setText("Status: Bounding box deleted")
                except Exception as e:
                    print(f"Error deleting label file: {str(e)}")
                    
            # Update display
            self.display_image(load_from_file=False)