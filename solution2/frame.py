import numpy as np
import cv2
from typing import Optional, Tuple, List

class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, depth_scale: float = 5000.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        
        # Camera intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Distortion coefficients (assuming no distortion for TUM dataset)
        self.dist_coeffs = np.zeros(4, dtype=np.float32)
    
    def pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        """Convert pixel coordinates to camera coordinates"""
        if depth <= 0:
            return None
        
        z = depth / self.depth_scale
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return np.array([x, y, z], dtype=np.float32)
    
    def camera_to_pixel(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """Convert camera coordinates to pixel coordinates"""
        if point_3d[2] <= 0:
            return None
        
        u = int(self.fx * point_3d[0] / point_3d[2] + self.cx)
        v = int(self.fy * point_3d[1] / point_3d[2] + self.cy)
        
        return u, v


class Frame:
    def __init__(self, timestamp: float, rgb_image: np.ndarray, depth_image: np.ndarray, 
                 camera: Camera, frame_id: int = 0):
        self.timestamp = timestamp
        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.camera = camera
        self.frame_id = frame_id
        
        # Convert to grayscale for feature extraction
        if len(rgb_image.shape) == 3:
            self.gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = rgb_image
        
        # Feature extraction results
        self.keypoints = None
        self.descriptors = None
        
        # 3D points in camera coordinate
        self.points_3d = []
        self.points_2d = []
        
        # Pose relative to world coordinate (4x4 transformation matrix)
        self.pose = np.eye(4, dtype=np.float32)
    
    def extract_features(self, detector):
        """Extract keypoints and descriptors using given detector"""
        self.keypoints, self.descriptors = detector.detectAndCompute(self.gray_image, None)
        
        # Convert keypoints to 2D points and compute corresponding 3D points
        self.points_2d = []
        self.points_3d = []
        
        if self.keypoints is not None:
            for kp in self.keypoints:
                u, v = int(kp.pt[0]), int(kp.pt[1])
                
                # Check if the pixel coordinates are within image bounds
                if (0 <= u < self.depth_image.shape[1] and 
                    0 <= v < self.depth_image.shape[0]):
                    
                    depth = self.depth_image[v, u]
                    if depth > 0:
                        point_3d = self.camera.pixel_to_camera(u, v, depth)
                        if point_3d is not None:
                            self.points_2d.append([u, v])
                            self.points_3d.append(point_3d)
        
        self.points_2d = np.array(self.points_2d, dtype=np.float32)
        self.points_3d = np.array(self.points_3d, dtype=np.float32)
    
    def get_valid_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get valid 2D and 3D point correspondences"""
        return self.points_2d, self.points_3d
    
    def set_pose(self, pose: np.ndarray):
        """Set the pose of this frame"""
        self.pose = pose.copy()
    
    def get_pose(self) -> np.ndarray:
        """Get the pose of this frame"""
        return self.pose.copy()
    
    def get_camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates"""
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        return -R.T @ t