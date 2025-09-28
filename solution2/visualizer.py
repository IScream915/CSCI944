import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
from typing import List, Optional, Tuple

class Visualizer:
    def __init__(self, config: dict):
        self.config = config
        self.vis_config = config.get('visualization', {})
        
        # Window settings
        self.window_width = self.vis_config.get('window_width', 800)
        self.window_height = self.vis_config.get('window_height', 600)
        
        # Trajectory data
        self.trajectory_points = []
        self.current_pose = np.eye(4)
        self.current_image = None
        
        # Matplotlib setup for 3D visualization
        self.fig = None
        self.ax_3d = None
        self.ax_image = None
        
        # Animation and threading
        self.animation = None
        self.is_running = False
        self.update_event = threading.Event()
        
        # Plot elements
        self.trajectory_line = None
        self.camera_scatter = None
        self.image_display = None
        
        self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup matplotlib visualization"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 8))
        
        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_3d.set_title('Camera Trajectory')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # Current image display
        self.ax_image = self.fig.add_subplot(122)
        self.ax_image.set_title('Current Frame')
        self.ax_image.axis('off')
        
        # Initialize empty plots
        self.trajectory_line, = self.ax_3d.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
        self.camera_scatter = self.ax_3d.scatter([], [], [], c='red', s=50, label='Current Position')
        
        self.ax_3d.legend()
        
        # Set initial limits
        self.ax_3d.set_xlim([-2, 2])
        self.ax_3d.set_ylim([-2, 2])
        self.ax_3d.set_zlim([-1, 3])
        
        plt.tight_layout()
    
    def update_pose(self, pose: np.ndarray):
        """Update current camera pose"""
        self.current_pose = pose.copy()
        
        # Extract camera position
        R = pose[:3, :3]
        t = pose[:3, 3]
        camera_center = -R.T @ t
        
        self.trajectory_points.append(camera_center)
        self.update_event.set()
    
    def update_image(self, image: np.ndarray):
        """Update current image display"""
        self.current_image = image.copy()
        self.update_event.set()
    
    def _update_plot(self, frame):
        """Update the visualization plots"""
        if not self.update_event.is_set():
            return self.trajectory_line, self.camera_scatter
        
        # Update trajectory
        if len(self.trajectory_points) > 0:
            trajectory_array = np.array(self.trajectory_points)
            self.trajectory_line.set_data(trajectory_array[:, 0], trajectory_array[:, 1])
            self.trajectory_line.set_3d_properties(trajectory_array[:, 2])
            
            # Update current camera position
            current_pos = trajectory_array[-1]
            self.camera_scatter._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
            
            # Adjust plot limits if necessary
            if len(trajectory_array) > 1:
                margin = 1.0
                x_min, x_max = trajectory_array[:, 0].min() - margin, trajectory_array[:, 0].max() + margin
                y_min, y_max = trajectory_array[:, 1].min() - margin, trajectory_array[:, 1].max() + margin
                z_min, z_max = trajectory_array[:, 2].min() - margin, trajectory_array[:, 2].max() + margin
                
                self.ax_3d.set_xlim([x_min, x_max])
                self.ax_3d.set_ylim([y_min, y_max])
                self.ax_3d.set_zlim([z_min, z_max])
        
        # Update current image
        if self.current_image is not None:
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            if self.image_display is None:
                self.image_display = self.ax_image.imshow(image_rgb)
            else:
                self.image_display.set_data(image_rgb)
        
        self.update_event.clear()
        return self.trajectory_line, self.camera_scatter
    
    def start_visualization(self):
        """Start the real-time visualization"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start animation
        self.animation = FuncAnimation(
            self.fig, self._update_plot, 
            interval=50, blit=False, cache_frame_data=False
        )
        
        plt.show(block=False)
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)
    
    def save_trajectory(self, filename: str):
        """Save trajectory to file"""
        if len(self.trajectory_points) > 0:
            trajectory_array = np.array(self.trajectory_points)
            np.savetxt(filename, trajectory_array, fmt='%.6f', 
                      header='x y z', comments='# ')
            print(f"Trajectory saved to {filename}")


class OpenCVVisualizer:
    """Alternative OpenCV-based visualizer for simpler display"""
    
    def __init__(self, config: dict):
        self.config = config
        self.vis_config = config.get('visualization', {})
        
        self.window_name_image = "Current Frame"
        self.window_name_trajectory = "Camera Trajectory (Top View)"
        
        # Trajectory visualization parameters
        self.trajectory_points = []
        self.trajectory_image = np.ones((600, 600, 3), dtype=np.uint8) * 255
        self.scale = 100  # pixels per meter
        self.center = (300, 300)  # image center
        
        self.current_image = None
        
        # Initialize windows
        cv2.namedWindow(self.window_name_image, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.window_name_trajectory, cv2.WINDOW_NORMAL)
    
    def update_pose(self, pose: np.ndarray):
        """Update current camera pose"""
        # Extract camera position
        R = pose[:3, :3]
        t = pose[:3, 3]
        camera_center = -R.T @ t
        
        self.trajectory_points.append(camera_center)
        self._draw_trajectory()
    
    def update_image(self, image: np.ndarray):
        """Update current image display"""
        self.current_image = image.copy()
        
        # Add some text overlay
        if self.current_image is not None:
            # Add frame info
            cv2.putText(self.current_image, f"Frame: {len(self.trajectory_points)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name_image, self.current_image)
    
    def _draw_trajectory(self):
        """Draw trajectory on the top-view image"""
        if len(self.trajectory_points) < 2:
            return
        
        # Clear trajectory image
        self.trajectory_image.fill(255)
        
        # Draw coordinate axes
        cv2.line(self.trajectory_image, (self.center[0], 0), 
                (self.center[0], 600), (200, 200, 200), 1)
        cv2.line(self.trajectory_image, (0, self.center[1]), 
                (600, self.center[1]), (200, 200, 200), 1)
        
        # Convert 3D points to 2D image coordinates (top view: X-Z plane)
        image_points = []
        for point in self.trajectory_points:
            x = int(point[0] * self.scale + self.center[0])
            z = int(-point[2] * self.scale + self.center[1])  # Negative Z for correct orientation
            image_points.append((x, z))
        
        # Draw trajectory
        for i in range(1, len(image_points)):
            cv2.line(self.trajectory_image, image_points[i-1], image_points[i], 
                    (255, 0, 0), 2)
        
        # Draw current position
        if len(image_points) > 0:
            cv2.circle(self.trajectory_image, image_points[-1], 5, (0, 0, 255), -1)
        
        # Add scale info
        cv2.putText(self.trajectory_image, f"Scale: {self.scale} pixels/meter", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(self.trajectory_image, f"Points: {len(self.trajectory_points)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow(self.window_name_trajectory, self.trajectory_image)
    
    def start_visualization(self):
        """Start visualization (OpenCV is already showing windows)"""
        pass
    
    def stop_visualization(self):
        """Stop visualization and close windows"""
        cv2.destroyAllWindows()
    
    def save_trajectory(self, filename: str):
        """Save trajectory to file"""
        if len(self.trajectory_points) > 0:
            trajectory_array = np.array(self.trajectory_points)
            np.savetxt(filename, trajectory_array, fmt='%.6f', 
                      header='x y z', comments='# ')
            print(f"Trajectory saved to {filename}")