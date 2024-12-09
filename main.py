import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sys

from absl import app, flags
from absl.flags import FLAGS
import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QComboBox, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
flags.DEFINE_string('config_file', "/home/huzaifa/Kopernikus/ml-envs-config/0012-avm5/box3d/box3d.yaml", 'path to config file')
flags.DEFINE_string('map', "/home/huzaifa/Kopernikus/ml-envs-config/0012-avm5/map/blended.png", "path to the map")

@dataclass
class CameraParams:
    """Store camera parameters including orientation."""
    cam_id: str
    matrix: List[np.ndarray]
    ego_roi_poly: List[List[List[float]]]
    original_ego_roi_poly: List[List[List[float]]]  # Store original ROI
    global_angle: float
    cam_x: float
    cam_y: float
    pitch: float = 0.0
    roll: float = 0.0

    def __post_init__(self):
        # Deep copy of original ROI to preserve initial state
        self.original_ego_roi_poly = [
            [list(point) for point in zone] for zone in self.ego_roi_poly
        ]

    def update_orientation(self, pitch: float, roll: float):
        """Update camera orientation and recalculate ROI points."""
        self.pitch = pitch
        self.roll = roll
        
        # Reset to original ROI first
        self.ego_roi_poly = [
            [list(point) for point in zone] for zone in self.original_ego_roi_poly
        ]
        
        # Apply transformation if not at 0,0
        if pitch != 0 or roll != 0:
            for zone_id in range(len(self.ego_roi_poly)):
                for point_idx in range(len(self.ego_roi_poly[zone_id])):
                    x = self.ego_roi_poly[zone_id][point_idx][0]
                    y = self.ego_roi_poly[zone_id][point_idx][1]
                    
                    # More complex rotation matrix for pitch and roll
                    # This is a simplified version and might need refinement
                    pitch_rad = math.radians(pitch)
                    roll_rad = math.radians(roll)
                    
                    # 3D rotation matrix (simplified)
                    x_rot = (x * math.cos(pitch_rad) + 
                             y * math.sin(roll_rad) * math.sin(pitch_rad))
                    y_rot = (y * math.cos(roll_rad) - 
                             x * math.sin(pitch_rad))
                    
                    self.ego_roi_poly[zone_id][point_idx][0] = round(x_rot)
                    self.ego_roi_poly[zone_id][point_idx][1] = round(y_rot)

class CoverageViewer(QMainWindow):
    """Main window for interactive coverage visualization."""
    def __init__(self, analyzer: 'CoverageAnalyzer'):
        super().__init__()
        self.analyzer = analyzer
        self.current_camera_id = None
        self.debounce_timer = QTimer()
        self.debounce_timer.setInterval(100)  # 100 ms debounce interval
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_coverage_map)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('Camera Coverage Viewer')
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # Create image display area
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(250)  # Reduced width
        layout.addWidget(control_panel)

        # Add camera selector
        selector_group = QGroupBox("Camera Selection")
        selector_layout = QVBoxLayout()
        self.camera_selector = QComboBox()
        #self.camera_selector.addItems(sorted(map(str, self.analyzer.cameras.keys())))
        self.camera_selector.addItems([str(cam_id) for cam_id in sorted(self.analyzer.cameras.keys())])
        self.camera_selector.currentTextChanged.connect(self.on_camera_selected)
        selector_layout.addWidget(self.camera_selector)
        selector_group.setLayout(selector_layout)
        control_layout.addWidget(selector_group)

        # Add slider controls group
        self.slider_group = QGroupBox("Camera Controls")
        slider_layout = QVBoxLayout()

        # Pitch slider
        pitch_label = QLabel('Pitch: 0°')
        self.pitch_label = pitch_label
        slider_layout.addWidget(pitch_label)
        pitch_slider = QSlider(Qt.Horizontal)
        pitch_slider.setRange(-90, 90)
        pitch_slider.setValue(0)
        pitch_slider.setTickInterval(10)
        pitch_slider.setTickPosition(QSlider.TicksBelow)
        pitch_slider.valueChanged.connect(lambda v: self.on_slider_change('pitch', v))
        self.pitch_slider = pitch_slider
        slider_layout.addWidget(pitch_slider)

        # Roll slider
        roll_label = QLabel('Roll: 0°')
        self.roll_label = roll_label
        slider_layout.addWidget(roll_label)
        roll_slider = QSlider(Qt.Horizontal)
        roll_slider.setRange(-90, 90)
        roll_slider.setValue(0)
        roll_slider.setTickInterval(10)
        roll_slider.setTickPosition(QSlider.TicksBelow)
        roll_slider.valueChanged.connect(lambda v: self.on_slider_change('roll', v))
        self.roll_slider = roll_slider
        slider_layout.addWidget(roll_slider)

        self.slider_group.setLayout(slider_layout)
        control_layout.addWidget(self.slider_group)
        
        # Add stretch to push controls to the top
        control_layout.addStretch()

        # Select first camera by default
        if self.camera_selector.count() > 0:
            self.current_camera_id = self.camera_selector.currentText()
            
        # Initial update
        self.update_coverage_map()
        
        # Set window size based on map dimensions
        map_height, map_width = self.analyzer.base_map.shape[:2]
        scaling_factor = min(800 / map_width, 600 / map_height)  # Max dimensions 800x600
        self.setFixedSize(int(map_width * scaling_factor) + 270, int(map_height * scaling_factor))

    @pyqtSlot(str)
    def on_camera_selected(self, cam_id: str):
        """Handle camera selection change."""
        try:
            # Convert cam_id to integer
            cam_id = int(cam_id)
            
            # Retrieve camera
            camera = self.analyzer.cameras.get(cam_id)
            
            if camera is None:
                print(f"Camera {cam_id} not found in available cameras.")
                return
            
            # Set current camera ID in the analyzer
            self.analyzer.current_camera_id = cam_id
            self.current_camera_id = cam_id
            
            # Update slider values without triggering updates
            self.pitch_slider.blockSignals(True)
            self.roll_slider.blockSignals(True)
            
            self.pitch_slider.setValue(int(camera.pitch))
            self.roll_slider.setValue(int(camera.roll))
            self.pitch_label.setText(f'Pitch: {int(camera.pitch)}°')
            self.roll_label.setText(f'Roll: {int(camera.roll)}°')
            
            self.pitch_slider.blockSignals(False)
            self.roll_slider.blockSignals(False)
            
            self.update_coverage_map()
        
        except (ValueError, KeyError) as e:
            print(f"Error selecting camera {cam_id}: {e}")


    @pyqtSlot(str, int)
    def on_slider_change(self, param: str, value: int):
        """Handle slider value changes with debounce."""
        if not self.current_camera_id:
            return

        # Update label
        if param == 'pitch':
            self.pitch_label.setText(f'Pitch: {value}°')
        else:
            self.roll_label.setText(f'Roll: {value}°')

        # Update camera parameters
        camera = self.analyzer.cameras[self.current_camera_id]
        camera.update_orientation(
            self.pitch_slider.value(),
            self.roll_slider.value()
        )

        # Trigger debounce timer to update the map
        self.debounce_timer.start()

    def update_coverage_map(self):
        """Update the coverage map visualization."""
        coverage_map = self.analyzer.create_coverage_map()
        
        # Scale the image to fit the window while maintaining aspect ratio
        height, width = coverage_map.shape[:2]
        max_height = self.height() - 50  # Leave room for controls
        max_width = self.width() - 270  # Leave room for the sidebar

        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the map image
        scaled_map = cv2.resize(coverage_map, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Add camera IDs to the visualization
        for cam_id, cam_params in self.analyzer.cameras.items():
            position = (
                int(cam_params.cam_x * scale), 
                int(cam_params.cam_y * scale)
            )
            color = (0, 255, 0) if str(cam_id) == str(self.current_camera_id) else (255, 255, 255)
            cv2.putText(
                scaled_map, f'Cam {cam_id}', position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Convert to Qt format and display
        bytes_per_line = 3 * new_width
        qt_image = QImage(
            scaled_map.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))


class CoverageAnalyzer:
    """Analyzes camera coverage and generates coverage maps."""
    def __init__(self, config_file: str, map_path: str):
        self.config_file = config_file
        self.map_path = map_path
        self.base_map = cv2.imread(map_path)
        self.height, self.width = self.base_map.shape[:2]
        self.cameras: Dict[str, CameraParams] = {}
        self.current_camera_id = None  # Add this line
        self.color_map = {
            0: (0, 0, 0),      # Black
            1: (0, 0, 255),    # Red
            2: (0, 255, 0),    # Green
            3: (255, 0, 0),    # Blue
            4: (128, 0, 128),  # Purple
            5: (255, 255, 0)   # Cyan
        }
        self.load_cameras()

    def load_cameras(self):
        """Load camera parameters from config file."""
        from mlcommon.config import get_available_cameras, get_camera_params
        
        cam_ids = get_available_cameras(self.config_file)
        for cam_id in cam_ids:
            params = get_camera_params(cam_id, self.config_file)
            if params["camera_enabled"]:
                self.cameras[cam_id] = CameraParams(
                    cam_id=cam_id,
                    matrix=params["matrix"],
                    ego_roi_poly=params["ego_roi_poly"],
                    original_ego_roi_poly=params["ego_roi_poly"],  # Store original
                    global_angle=params["global_angle"],
                    cam_x=params["cam_x"],
                    cam_y=params["cam_y"]
                )

    @staticmethod
    def reproject_to_global_map_coords(x: float, y: float, matrix: np.ndarray) -> Tuple[float, float]:
        """Convert local coordinates to global map coordinates."""
        res = np.dot(matrix, [x, y, 1])
        return round(res[0] / res[2], 4), round(res[1] / res[2], 4)

    def create_coverage_map(self) -> np.ndarray:
        """Create the coverage visualization map."""
        # Use the base map as a starting point
        coverage_map = self.base_map.copy()

        # Draw all camera ROIs
        for cam_id, camera in self.cameras.items():
            for zone_id, roi in enumerate(camera.ego_roi_poly):
                projected_points = []
                for point in roi:
                    x, y = self.reproject_to_global_map_coords(
                        point[0], point[1], camera.matrix[zone_id]
                    )
                    if 0 <= x < self.width and 0 <= y < self.height:
                        projected_points.append((int(x), int(y)))

                if len(projected_points) > 2:
                    if self.current_camera_id is not None and str(cam_id) == str(self.current_camera_id):
                        # Bright green for the selected camera
                        cv2.fillPoly(coverage_map, [np.array(projected_points)], (0, 255, 0))
                    else:
                        # Red for other cameras
                        cv2.fillPoly(coverage_map, [np.array(projected_points)], (0, 0, 255))

        return coverage_map


def main(_argv):
    # Create analyzer
    analyzer = CoverageAnalyzer(FLAGS.config_file, FLAGS.map)
    
    # Create and show GUI
    app = QApplication(sys.argv)
    viewer = CoverageViewer(analyzer)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass