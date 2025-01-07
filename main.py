import numpy as np
import cv2
import os
import random
from datetime import datetime
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import sys
from colorsys import hsv_to_rgb

from absl import app, flags
from absl.flags import FLAGS
import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QComboBox, QGroupBox,
                            QListWidget, QListWidgetItem, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSlot, QPoint, QPointF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QPushButton

flags.DEFINE_string('config_file', "/home/huzaifa/Kopernikus/ml-envs-config/0012-avm5/box3d/box3d.yaml", 'path to config file')
flags.DEFINE_string('map', "/home/huzaifa/Kopernikus/ml-envs-config/0012-avm5/map/blended.png", "path to the map")


#helpers


def generate_random_roi_polygon(center_x: float, center_y: float) -> List[List[List[float]]]:
    """Generate a fixed ROI polygon translated near the given center_x and center_y."""

    # Fixed polygon points
    base_points = [[[573, 400], [524, 226], [941, 320], [925, 393], [979, 452]]]
    
    # Calculate the original center of the fixed polygon
    original_center_x = sum(point[0] for point in base_points[0]) / len(base_points[0])
    original_center_y = sum(point[1] for point in base_points[0]) / len(base_points[0])
    
    # Calculate the offsets to move the polygon near the given center
    offset_x = center_x - original_center_x
    offset_y = center_y - original_center_y
    
    # Move each point by the calculated offsets
    translated_points = [
        [[point[0] + offset_x, point[1] + offset_y] for point in polygon]
        for polygon in base_points
    ]
    
    return translated_points


def update_roi_position(roi: List[List[List[float]]], dx: float, dy: float) -> List[List[List[float]]]:
    """Update ROI polygon position by moving it by dx, dy."""
    return [[[point[0] + dx, point[1] + dy] for point in zone] 
            for zone in roi]

@dataclass
class CameraParams:
    """Store camera parameters including orientation."""
    cam_id: str
    matrix: List[np.ndarray]
    ego_roi_poly: List[List[List[float]]]
    original_ego_roi_poly: List[List[List[float]]]
    global_angle: float
    cam_x: float
    cam_y: float
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0  # Added yaw parameter
    color: Tuple[int, int, int] = (0, 0, 255)
    has_roi: bool = False 

    def __post_init__(self):
        """Initialize additional attributes after creation."""
        self.original_ego_roi_poly = [
            [list(point) for point in zone] for zone in self.ego_roi_poly
        ]
        self.original_cam_x = self.cam_x
        self.original_cam_y = self.cam_y
        self.transformed_points = self.calculate_transformed_points()

    def calculate_transformed_points(self) -> List[List[Tuple[int, int]]]:
        """Pre-calculate transformed points for all zones using proper RPY order."""
        transformed = []
        for zone_id in range(len(self.ego_roi_poly)):
            zone_points = []
            for point in self.original_ego_roi_poly[zone_id]:
                # Get the point relative to the original camera position
                rel_x = point[0] - self.original_cam_x
                rel_y = point[1] - self.original_cam_y
                
                # Convert angles to radians
                roll_rad = math.radians(self.roll)
                pitch_rad = math.radians(self.pitch)
                yaw_rad = math.radians(self.yaw)
                
                # Apply roll first (around X-axis)
                y_roll = rel_y * math.cos(roll_rad) - rel_y * math.sin(roll_rad)
                z_roll = rel_y * math.sin(roll_rad) + rel_y * math.cos(roll_rad)
                
                # Apply pitch second (around Y-axis)
                x_pitch = rel_x * math.cos(pitch_rad) + z_roll * math.sin(pitch_rad)
                z_pitch = -rel_x * math.sin(pitch_rad) + z_roll * math.cos(pitch_rad)
                
                # Apply yaw last (around Z-axis)
                x_final = x_pitch * math.cos(yaw_rad) - y_roll * math.sin(yaw_rad)
                y_final = x_pitch * math.sin(yaw_rad) + y_roll * math.cos(yaw_rad)
                
                # Project back to 2D plane and translate to current camera position
                final_x = x_final + self.cam_x
                final_y = y_final + self.cam_y
                
                zone_points.append((round(final_x), round(final_y)))
            transformed.append(zone_points)
        return transformed


    def update_orientation(self, pitch: float, roll: float, yaw: float):
        """Update camera orientation and recalculate ROI points."""
        if self.pitch == pitch and self.roll == roll and self.yaw == yaw:
            return False
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.transformed_points = self.calculate_transformed_points()
        return True

    def draw_axes(self, image: np.ndarray, scale_factor: float = 1.0, offset_x: int = 0, offset_y: int = 0):
        """Draw XYZ axes to visualize camera orientation."""
        axis_length = 50  # Length of axis arrows
        
        # Calculate rotated axis endpoints
        def rotate_point(x, y, z):
            # Apply yaw
            yaw_rad = math.radians(self.yaw)
            x_yaw = x * math.cos(yaw_rad) - y * math.sin(yaw_rad)
            y_yaw = x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
            
            # Apply pitch and roll
            pitch_rad = math.radians(self.pitch)
            roll_rad = math.radians(self.roll)
            
            x_final = (x_yaw * math.cos(pitch_rad) + 
                    z * math.sin(pitch_rad))
            y_final = (y_yaw * math.cos(roll_rad) - 
                    z * math.sin(roll_rad))
            
            return (round(x_final), round(y_final))

        # Draw axes
        origin = (int(self.cam_x * scale_factor + offset_x), 
                int(self.cam_y * scale_factor + offset_y))
        
        # X-axis (Red)
        x_end = rotate_point(axis_length, 0, 0)
        cv2.line(image, origin, 
                (origin[0] + x_end[0], origin[1] + x_end[1]), 
                (0, 0, 255), 2)
        
        # Y-axis (Green)
        y_end = rotate_point(0, axis_length, 0)
        cv2.line(image, origin,
                (origin[0] + y_end[0], origin[1] + y_end[1]),
                (0, 255, 0), 2)
        
        # Z-axis (Blue)
        z_end = rotate_point(0, 0, axis_length)
        cv2.line(image, origin,
                (origin[0] + z_end[0], origin[1] + z_end[1]),
                (255, 0, 0), 2)

class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
    def mousePressEvent(self, event):
        self.parent.mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        self.parent.mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        self.parent.mouseReleaseEvent(event)
        
    def wheelEvent(self, event):
        self.parent.wheel_event(event)

class CoverageViewer(QMainWindow):
    def __init__(self, analyzer: 'CoverageAnalyzer'):
        super().__init__()
        self.analyzer = analyzer
        self.selected_cameras: Set[str] = set()
        # Add zoom-related attributes
        self.zoom_scale = 1.0
        self.zoom_center = QPoint(0, 0)
        self.last_mouse_pos = QPoint(0, 0)
        
        # Get map dimensions for X,Y sliders
        self.map_width = self.analyzer.width
        self.map_height = self.analyzer.height
        
        # Add drag-related attributes
        self.is_dragging = False
        self.drag_start_pos = QPoint(0, 0)
        self.drag_start_center = QPoint(0, 0)
        self.cameras_needing_roi = set()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Camera Coverage Viewer')
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # Image display area with custom label class
        self.image_label = DraggableLabel(self)
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label)

        # Control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(250)
        layout.addWidget(control_panel)

        # Camera selection list
        selector_group = QGroupBox("Camera Selection")
        selector_layout = QVBoxLayout()
        
        # Add "All Cameras" checkbox
        self.all_cameras_checkbox = QCheckBox("All Cameras")
        self.all_cameras_checkbox.stateChanged.connect(self.on_all_cameras_toggled)
        selector_layout.addWidget(self.all_cameras_checkbox)
        
        # Camera list
        self.camera_list = QListWidget()
        self.camera_list.itemChanged.connect(self.on_camera_selection_changed)
        for cam_id in sorted(self.analyzer.cameras.keys()):
            item = QListWidgetItem(f"Camera {cam_id}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.camera_list.addItem(item)
        selector_layout.addWidget(self.camera_list)
        
        selector_group.setLayout(selector_layout)
        control_layout.addWidget(selector_group)

        # Active camera controls
        self.slider_group = QGroupBox("Active Camera Controls")
        slider_layout = QVBoxLayout()

        # Active camera selector
        self.active_camera_selector = QComboBox()
        self.active_camera_selector.currentTextChanged.connect(self.on_active_camera_changed)
        slider_layout.addWidget(self.active_camera_selector)

        # Pitch slider
        self.pitch_label = QLabel('Pitch: 0°')
        slider_layout.addWidget(self.pitch_label)
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setRange(-90, 90)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickInterval(10)
        self.pitch_slider.setTickPosition(QSlider.TicksBelow)
        self.pitch_slider.valueChanged.connect(lambda v: self.on_slider_change('pitch', v))
        slider_layout.addWidget(self.pitch_slider)

        # Roll slider
        self.roll_label = QLabel('Roll: 0°')
        slider_layout.addWidget(self.roll_label)
        self.roll_slider = QSlider(Qt.Horizontal)
        self.roll_slider.setRange(-90, 90)
        self.roll_slider.setValue(0)
        self.roll_slider.setTickInterval(10)
        self.roll_slider.setTickPosition(QSlider.TicksBelow)
        self.roll_slider.valueChanged.connect(lambda v: self.on_slider_change('roll', v))
        slider_layout.addWidget(self.roll_slider)

        # Yaw slider
        self.yaw_label = QLabel('Yaw: 0°')
        slider_layout.addWidget(self.yaw_label)
        self.yaw_slider = QSlider(Qt.Horizontal)
        self.yaw_slider.setRange(-180, 180)  # Yaw range is -180 to +180 degrees
        self.yaw_slider.setValue(0)
        self.yaw_slider.setTickInterval(30)
        self.yaw_slider.setTickPosition(QSlider.TicksBelow)
        self.yaw_slider.valueChanged.connect(lambda v: self.on_slider_change('yaw', v))
        slider_layout.addWidget(self.yaw_slider)
        
        # X position slider
        self.x_label = QLabel('X: 0')
        slider_layout.addWidget(self.x_label)
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setRange(0, self.map_width)
        self.x_slider.setValue(0)
        self.x_slider.setTickInterval(100)
        self.x_slider.setTickPosition(QSlider.TicksBelow)
        self.x_slider.valueChanged.connect(lambda v: self.on_slider_change('x', v))
        slider_layout.addWidget(self.x_slider)

        # Y position slider
        self.y_label = QLabel('Y: 0')
        slider_layout.addWidget(self.y_label)
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(0, self.map_height)
        self.y_slider.setValue(0)
        self.y_slider.setTickInterval(100)
        self.y_slider.setTickPosition(QSlider.TicksBelow)
        self.y_slider.valueChanged.connect(lambda v: self.on_slider_change('y', v))
        slider_layout.addWidget(self.y_slider)

        # Add New Camera button
        self.add_camera_button = QPushButton("Add New Camera")
        self.add_camera_button.clicked.connect(self.add_new_camera)
        control_layout.addWidget(self.add_camera_button)

        # # Add Generate ROI button
        # self.generate_roi_button = QPushButton("Generate ROI for Camera")
        # self.generate_roi_button.clicked.connect(self.generate_roi_for_camera)
        # self.generate_roi_button.setEnabled(False)  # Disabled by default
        # control_layout.addWidget(self.generate_roi_button)

        # Add Delete Camera button
        self.delete_camera_button = QPushButton("Delete Selected Camera")
        self.delete_camera_button.clicked.connect(self.delete_selected_camera)
        control_layout.addWidget(self.delete_camera_button)

        # Add Save Heat Map button
        self.save_heatmap_button = QPushButton("Save Heat Map Images")
        self.save_heatmap_button.clicked.connect(self.save_heat_map_images)
        control_layout.addWidget(self.save_heatmap_button)
    
        self.slider_group.setLayout(slider_layout)
        control_layout.addWidget(self.slider_group)
        self.slider_group.setEnabled(False)
        
        control_layout.addStretch()

        # Set window size
        map_height, map_width = self.analyzer.base_map.shape[:2]
        scaling_factor = min(1800 / map_width, 1600 / map_height)
        self.setFixedSize(int(map_width * scaling_factor) + 270, int(map_height * scaling_factor))

        # Initial update
        self.update_coverage_map()

    def on_all_cameras_toggled(self, state):
        """Handle 'All Cameras' checkbox toggle."""
        for i in range(self.camera_list.count()):
            item = self.camera_list.item(i)
            item.setCheckState(Qt.Checked if state else Qt.Unchecked)

    def on_camera_selection_changed(self, item):
        """Handle camera selection changes."""
        self.selected_cameras.clear()
        if self.all_cameras_checkbox.isChecked():
            self.selected_cameras = set(str(cam_id) for cam_id in self.analyzer.cameras.keys())
        else:
            for i in range(self.camera_list.count()):
                item = self.camera_list.item(i)
                if item.checkState() == Qt.Checked:
                    cam_id = str(sorted(self.analyzer.cameras.keys())[i])
                    self.selected_cameras.add(cam_id)


        active_cam_text = self.active_camera_selector.currentText()
        if active_cam_text:
            cam_id = int(active_cam_text.split()[-1])
            camera = self.analyzer.cameras[cam_id]
            self.generate_roi_button.setEnabled(not camera.has_roi)
        else:
            self.generate_roi_button.setEnabled(False)

        # Update active camera selector
        self.active_camera_selector.clear()
        self.active_camera_selector.addItems([f"Camera {cam_id}" for cam_id in self.selected_cameras])
        self.slider_group.setEnabled(len(self.selected_cameras) > 0)
        
        self.update_coverage_map()

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = event.pos()
            self.drag_start_center = QPoint(self.zoom_center)

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.is_dragging:
            # Calculate the movement delta
            delta = event.pos() - self.drag_start_pos
            
            # Update the zoom center based on the drag
            self.zoom_center = QPoint(
                self.drag_start_center.x() - delta.x(),
                self.drag_start_center.y() - delta.y()
            )
            
            # Update the display
            self.update_coverage_map()

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def wheel_event(self, event):
        """Handle mouse wheel events for zooming."""
        # Get the position of the mouse relative to the label
        mouse_pos = event.pos()
        
        # Store the mouse position relative to the scene before zoom
        old_pos = self.map_to_scene(mouse_pos)
        
        # Calculate zoom factor
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:  # Zoom out
            zoom_factor = 1.0 / zoom_factor
            
        # Update zoom scale
        new_scale = self.zoom_scale * zoom_factor
        
        # Limit zoom range (adjust these values as needed)
        if 0.1 <= new_scale <= 10.0:
            self.zoom_scale = new_scale
            
            # Calculate new position after zoom to keep mouse point fixed
            new_pos = self.map_to_scene(mouse_pos)
            
            # Convert QPoints to QPointF for floating-point arithmetic
            old_pos_f = QPointF(old_pos)
            new_pos_f = QPointF(new_pos)
            delta = new_pos_f - old_pos_f
            
            # Update zoom center (convert back to integers for QPoint)
            self.zoom_center = QPoint(
                int(self.zoom_center.x() - delta.x()),
                int(self.zoom_center.y() - delta.y())
            )
            
            # Update the display
            self.update_coverage_map()
            
    def map_to_scene(self, pos):
        """Convert mouse position to scene coordinates."""
        if not self.image_label.pixmap():
            return QPoint(0, 0)
            
        # Get the size of the displayed image
        viewport_size = self.image_label.size()
        image_size = self.image_label.pixmap().size()
        
        # Calculate scaling factors
        scale_x = image_size.width() / viewport_size.width()
        scale_y = image_size.height() / viewport_size.height()
        
        # Calculate coordinates and convert to integers
        x = int((pos.x() * scale_x) / self.zoom_scale - self.zoom_center.x())
        y = int((pos.y() * scale_y) / self.zoom_scale - self.zoom_center.y())
        
        return QPoint(x, y)


    def on_active_camera_changed(self, cam_text):
        """Handle active camera change for controls."""
        if not cam_text:
            return
            
        cam_id = cam_text.split()[-1]
        camera = self.analyzer.cameras[int(cam_id)]
        
        self.pitch_slider.blockSignals(True)
        self.roll_slider.blockSignals(True)
        self.yaw_slider.blockSignals(True)
        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        
        self.pitch_slider.setValue(int(camera.pitch))
        self.roll_slider.setValue(int(camera.roll))
        self.yaw_slider.setValue(int(camera.yaw))
        self.x_slider.setValue(int(camera.cam_x))
        self.y_slider.setValue(int(camera.cam_y))
        
        self.pitch_label.setText(f'Pitch: {int(camera.pitch)}°')
        self.roll_label.setText(f'Roll: {int(camera.roll)}°')
        self.yaw_label.setText(f'Yaw: {int(camera.yaw)}°')
        self.x_label.setText(f'X: {int(camera.cam_x)}')
        self.y_label.setText(f'Y: {int(camera.cam_y)}')
        
        self.pitch_slider.blockSignals(False)
        self.roll_slider.blockSignals(False)
        self.yaw_slider.blockSignals(False)
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)


    def on_slider_change(self, param: str, value: int):
        """Handle slider value changes."""
        active_cam_text = self.active_camera_selector.currentText()
        if not active_cam_text:
            return
            
        cam_id = int(self.active_camera_selector.currentText().split()[-1])
        camera = self.analyzer.cameras[cam_id]
        
        if param == 'pitch':
            self.pitch_label.setText(f'Pitch: {value}°')
            camera.update_orientation(value, camera.roll, camera.yaw)
        elif param == 'roll':
            self.roll_label.setText(f'Roll: {value}°')
            camera.update_orientation(camera.pitch, value, camera.yaw)
        elif param == 'yaw':
            self.yaw_label.setText(f'Yaw: {value}°')
            camera.update_orientation(camera.pitch, camera.roll, value)
        elif param == 'x':
            # Calculate movement delta
            dx = value - camera.cam_x
            self.x_label.setText(f'X: {value}')
            camera.cam_x = value
            # Update ROI polygon position
            camera.ego_roi_poly = update_roi_position(camera.ego_roi_poly, dx, 0)
            camera.original_ego_roi_poly = camera.ego_roi_poly
            camera.transformed_points = camera.calculate_transformed_points()
        elif param == 'y':
            # Calculate movement delta
            dy = value - camera.cam_y
            self.y_label.setText(f'Y: {value}')
            camera.cam_y = value
            # Update ROI polygon position
            camera.ego_roi_poly = update_roi_position(camera.ego_roi_poly, 0, dy)
            camera.original_ego_roi_poly = camera.ego_roi_poly
            camera.transformed_points = camera.calculate_transformed_points()
        
        self.update_coverage_map()

    def update_coverage_map(self):
        """Update the coverage map visualization."""
        coverage_map = self.analyzer.create_coverage_map(self.selected_cameras)
        
        height, width = coverage_map.shape[:2]
        max_height = 1600
        max_width = 1800
        
        # Calculate base scale
        base_scale = min(max_width / width, max_height / height)
        
        # Apply zoom scale
        scale = base_scale * self.zoom_scale
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Create a larger canvas for dragging
        canvas_width = int(new_width * 1.5)
        canvas_height = int(new_height * 1.5)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Resize the coverage map
        coverage_map = cv2.resize(coverage_map, (new_width, new_height))
        
        # Calculate the offset for centered placement
        offset_x = (canvas_width - new_width) // 2 - self.zoom_center.x()
        offset_y = (canvas_height - new_height) // 2 - self.zoom_center.y()
        
        # Place the coverage map on the canvas
        y_start = max(0, offset_y)
        y_end = min(canvas_height, offset_y + new_height)
        x_start = max(0, offset_x)
        x_end = min(canvas_width, offset_x + new_width)
        
        canvas_region = canvas[y_start:y_end, x_start:x_end]
        map_y_start = max(0, -offset_y)
        map_x_start = max(0, -offset_x)
        map_y_end = map_y_start + (y_end - y_start)
        map_x_end = map_x_start + (x_end - x_start)
        
        canvas_region[:] = coverage_map[map_y_start:map_y_end, map_x_start:map_x_end]
        
        # Add camera markers and axes for selected cameras
        adjusted_scale = scale
        for cam_id in self.selected_cameras:
            cam_params = self.analyzer.cameras[int(cam_id)]
            position = (
                int(cam_params.cam_x * adjusted_scale + offset_x),
                int(cam_params.cam_y * adjusted_scale + offset_y)
            )
            
            # Only draw if the camera position is within the canvas
            if (0 <= position[0] < canvas_width and 
                0 <= position[1] < canvas_height):
                cam_params.draw_axes(canvas, adjusted_scale, offset_x, offset_y)
                cv2.putText(canvas, f'Cam {cam_id}', position,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.zoom_scale, 
                           cam_params.color, 2)

        # Convert to Qt image
        bytes_per_line = 3 * canvas_width
        qt_image = QImage(canvas.data, canvas_width, canvas_height, 
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Create pixmap and set it
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)


    def add_new_camera(self):
        """Add a new camera with automatically generated ROI polygon."""
        new_id = max(self.analyzer.cameras.keys()) + 1 if self.analyzer.cameras else 0
        default_x = self.map_width // 2
        default_y = self.map_height // 2
        
        # Create base polygon at the center
        base_polygon = [[[0, -100], [-150, 100], [150, 100]]]  # Triangle shape
        
        # Move polygon points relative to camera position
        moved_polygon = [[[point[0] + default_x, point[1] + default_y] for point in zone] 
                        for zone in base_polygon]
        
        template_camera = next(iter(self.analyzer.cameras.values())) if self.analyzer.cameras else None
        
        new_camera = CameraParams(
            cam_id=str(new_id),
            matrix=[np.eye(3)] if template_camera is None else template_camera.matrix.copy(),
            ego_roi_poly=moved_polygon,
            original_ego_roi_poly=moved_polygon.copy(),
            global_angle=0,
            cam_x=default_x,
            cam_y=default_y,
            pitch=0,
            roll=0,
            yaw=0,
            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            has_roi=True  # ROI is automatically created
        )
        
        self.analyzer.cameras[new_id] = new_camera
        
        # Add to camera list in UI
        item = QListWidgetItem(f"Camera {new_id}")
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        self.camera_list.addItem(item)
        
        # Select the new camera
        item.setCheckState(Qt.Checked)
        self.selected_cameras.add(str(new_id))
        
        # Update active camera selector
        self.active_camera_selector.clear()
        self.active_camera_selector.addItems([f"Camera {cam_id}" for cam_id in self.selected_cameras])
        self.active_camera_selector.setCurrentText(f"Camera {new_id}")
        
        # Enable controls
        self.slider_group.setEnabled(True)
        
        self.update_coverage_map()


    def generate_roi_for_camera(self):
        """Generate ROI for the currently selected camera."""
        active_cam_text = self.active_camera_selector.currentText()
        if not active_cam_text:
            return
            
        cam_id = int(self.active_camera_selector.currentText().split()[-1])
        camera = self.analyzer.cameras[cam_id]
        
        # Generate ROI based on current camera position
        new_roi = generate_random_roi_polygon(camera.cam_x, camera.cam_y)
        
        # Update camera with new ROI
        camera.ego_roi_poly = new_roi
        camera.original_ego_roi_poly = [
            [list(point) for point in zone] for zone in new_roi
        ]
        camera.has_roi = True
        camera.transformed_points = camera.calculate_transformed_points()
        
        self.update_coverage_map()

    def delete_selected_camera(self):
        """Delete the currently selected camera."""
        active_cam_text = self.active_camera_selector.currentText()
        if not active_cam_text:
            return
            
        cam_id = int(active_cam_text.split()[-1])
        
        # Remove from analyzer's cameras
        if cam_id in self.analyzer.cameras:
            del self.analyzer.cameras[cam_id]
        
        # Remove from selected cameras
        self.selected_cameras.discard(str(cam_id))
        
        # Remove from camera list
        for i in range(self.camera_list.count()):
            if self.camera_list.item(i).text() == f"Camera {cam_id}":
                self.camera_list.takeItem(i)
                break
        
        # Update active camera selector
        self.active_camera_selector.clear()
        self.active_camera_selector.addItems([f"Camera {cam_id}" for cam_id in self.selected_cameras])
        
        # Disable slider controls if no cameras left
        self.slider_group.setEnabled(len(self.selected_cameras) > 0)
        
        # Update display
        self.update_coverage_map()

    def save_heat_map_images(self):
        """Save heat map images showing camera coverage overlaps."""
        if not self.selected_cameras:
            return

        # Calculate overlap map
        overlap_map = self.analyzer.calculate_coverage_overlap(self.selected_cameras)
        
        # Create base map for visualization
        base_map = self.analyzer.base_map.copy()
        
        # Define color map similar to the script
        color_map = {
            0: (0, 0, 0),    # Black
            1: (0, 0, 255),  # Red
            2: (0, 255, 0),  # Green
            3: (255, 0, 0),  # Blue
            4: (128, 0, 128),# Purple
            5: (255, 255, 0) # Cyan
        }
        
        # Initialize image maps
        height, width = base_map.shape[:2]
        image_map = np.zeros((height, width, 3), dtype=np.uint8)
        image_map_ones = np.zeros((height, width, 3), dtype=np.uint8)
        image_map_coverages = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Populate image maps based on overlap
        for i in range(height):
            for j in range(width):
                overlap_value = min(overlap_map[i, j], 5)  # Cap at 5
                image_map[i, j] = color_map.get(overlap_value, color_map[5])
                
                if overlap_value == 1:
                    image_map_ones[i, j] = color_map.get(1)
        
        # Create overlayed maps
        overlayed_map = cv2.addWeighted(base_map, 0.7, image_map, 0.3, 0)
        
        # Add legend to images
        def add_legend(img):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            rectangle_height = 20
            rectangle_width = 40
            text_offset_x = -70
            text_offset_y = 0
            padding = 5
            start_x = img.shape[1] - rectangle_width - padding
            start_y = img.shape[0] - padding

            for key, color in reversed(list(color_map.items())):
                if key == 0:
                    continue
                
                end_x = start_x + rectangle_width
                end_y = start_y - rectangle_height
                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, -1)

                text = str(key)
                if text == "5":
                    text = "5+"
                cv2.putText(img, text, (end_x + text_offset_x, start_y - text_offset_y), 
                            font, font_scale, (255, 255, 255), font_thickness)

                start_y -= rectangle_height + padding

            return img
        
        # Add legend to images
        image_map = add_legend(image_map)
        overlayed_map = add_legend(overlayed_map)
        
        # Save images
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "coverage_heatmaps"
        os.makedirs(save_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(save_dir, f"coverage_heatmap_{timestamp}.png"), image_map)
        cv2.imwrite(os.path.join(save_dir, f"overlayed_coverage_heatmap_{timestamp}.png"), overlayed_map)
        cv2.imwrite(os.path.join(save_dir, f"single_camera_coverage_{timestamp}.png"), image_map_ones)

class CoverageAnalyzer:
    def __init__(self, config_file: str, map_path: str):
        self.config_file = config_file
        self.map_path = map_path
        self.base_map = cv2.imread(map_path)
        self.height, self.width = self.base_map.shape[:2]
        self.cameras: Dict[str, CameraParams] = {}
        self.load_cameras()

    def load_cameras(self):
        """Load camera parameters and assign unique colors."""
        from mlcommon.config import get_available_cameras, get_camera_params

        cam_ids = get_available_cameras(self.config_file)
        
        num_cameras = len([cid for cid in cam_ids 
                         if get_camera_params(cid, self.config_file)["camera_enabled"]])
        
        # Generate unique colors using HSV color space
        colors = []
        for i in range(num_cameras):
            hue = i * (360 / num_cameras)
            rgb = hsv_to_rgb(hue/360, 1.0, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb))
        
        color_idx = 0
        for cam_id in cam_ids:
            params = get_camera_params(cam_id, self.config_file)
            if params["camera_enabled"]:
                self.cameras[cam_id] = CameraParams(
                    cam_id=cam_id,
                    matrix=params["matrix"],
                    ego_roi_poly=params["ego_roi_poly"],
                    original_ego_roi_poly=params["ego_roi_poly"],
                    global_angle=params["global_angle"],
                    cam_x=params["cam_x"],
                    cam_y=params["cam_y"],
                    color=colors[color_idx]
                )
                color_idx += 1

    @staticmethod
    def reproject_to_global_map_coords(x: float, y: float, matrix: np.ndarray) -> Tuple[float, float]:
        """Convert local coordinates to global map coordinates."""
        res = np.dot(matrix, [x, y, 1])
        return round(res[0] / res[2], 4), round(res[1] / res[2], 4)

    def create_coverage_map(self, selected_cameras: Set[str]) -> np.ndarray:
        """Create the coverage visualization map."""
        result = self.base_map.copy()
        
        if not selected_cameras:
            return result

        overlay = np.zeros_like(self.base_map)
        
        for cam_id in selected_cameras:
            camera = self.cameras[int(cam_id)]
            local_overlay = np.zeros_like(self.base_map)
            
            for zone_id, points in enumerate(camera.transformed_points):
                projected_points = []
                for x, y in points:
                    map_x, map_y = self.reproject_to_global_map_coords(
                        x, y, camera.matrix[zone_id])
                    if 0 <= map_x < self.width and 0 <= map_y < self.height:
                        projected_points.append((int(map_x), int(map_y)))

                if len(projected_points) > 2:
                    cv2.fillPoly(local_overlay, [np.array(projected_points)], 
                               camera.color)

            overlay = cv2.addWeighted(overlay, 1, local_overlay, 0.6, 0)

        result = cv2.addWeighted(result, 0.7, overlay, 0.6, 0)
        return result



    def calculate_coverage_overlap(self, selected_cameras: Set[str]) -> np.ndarray:
        """
        Calculate the number of cameras covering each pixel.
        
        Returns:
            np.ndarray: 2D array with the number of cameras covering each pixel
        """
        overlap_map = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for cam_id in selected_cameras:
            camera = self.cameras[int(cam_id)]
            local_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            
            for zone_id, points in enumerate(camera.transformed_points):
                projected_points = []
                for x, y in points:
                    map_x, map_y = self.reproject_to_global_map_coords(
                        x, y, camera.matrix[zone_id])
                    if 0 <= map_x < self.width and 0 <= map_y < self.height:
                        projected_points.append((int(map_x), int(map_y)))

                if len(projected_points) > 2:
                    cv2.fillPoly(local_mask, [np.array(projected_points)], 255)
            
            overlap_map += local_mask // 255
        
        return overlap_map

def main(_argv):
    analyzer = CoverageAnalyzer(FLAGS.config_file, FLAGS.map)
    app = QApplication(sys.argv)
    viewer = CoverageViewer(analyzer)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
