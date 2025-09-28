import numpy as np
import cv2
import time
from enum import Enum
from typing import Optional, List, Tuple
from frame import Frame, Camera
from feature_matching import FeatureExtractor, FeatureMatcher, MotionEstimator, match_frames

class VOState(Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"

class VisualOdometry:
    def __init__(self, camera: Camera, config: dict):
        self.camera = camera
        self.config = config
        
        # Initialize feature extractor and matcher
        orb_config = config.get('orb', {})
        self.feature_extractor = FeatureExtractor("ORB", **orb_config)
        self.feature_matcher = FeatureMatcher("BF", distance_threshold=0.7)
        
        # Initialize motion estimator
        vo_config = config.get('vo', {})
        self.motion_estimator = MotionEstimator(
            camera_matrix=camera.K,
            dist_coeffs=camera.dist_coeffs,
            ransac_threshold=vo_config.get('ransac_threshold', 1.0),
            max_iterations=vo_config.get('max_iterations', 1000),
            min_inliers=vo_config.get('min_inliers', 20)
        )
        
        # VO state
        self.state = VOState.INITIALIZING
        self.current_frame = None
        self.previous_frame = None
        
        # Camera trajectory
        self.trajectory = []
        self.poses = []
        
        # Statistics
        self.num_inliers = 0
        self.num_matches = 0
        self.processing_time = 0.0
        
        # Initialization parameters
        self.min_initialization_matches = 50
        
    def process_frame(self, frame: Frame) -> bool:
        """
        Process a new frame and update the camera pose
        
        Returns:
            True if processing successful, False if VO is lost
        """
        start_time = time.time()
        
        # Extract features for the current frame
        frame.extract_features(self.feature_extractor.detector)
        
        if self.state == VOState.INITIALIZING:
            success = self._initialize(frame)
        elif self.state == VOState.TRACKING:
            success = self._track(frame)
        else:  # LOST
            return False
        
        self.processing_time = time.time() - start_time
        
        if success:
            self.previous_frame = self.current_frame
            self.current_frame = frame
            
            # Add current pose to trajectory
            camera_center = frame.get_camera_center()
            self.trajectory.append(camera_center)
            self.poses.append(frame.get_pose())
            
        return success
    
    def _initialize(self, frame: Frame) -> bool:
        """Initialize the VO system with the first frame"""
        if self.current_frame is None:
            # First frame - set as reference
            self.current_frame = frame
            # Set initial pose as identity (world origin)
            frame.set_pose(np.eye(4))
            
            print(f"Initialized with frame {frame.frame_id}")
            print(f"Features detected: {len(frame.keypoints) if frame.keypoints else 0}")
            
            return True
        else:
            # Try to initialize with second frame
            if len(frame.keypoints) == 0 or len(self.current_frame.keypoints) == 0:
                print("No features detected, cannot initialize")
                return False
            
            # Match features between frames
            points_3d_1, points_2d_2, points_3d_2, matches = match_frames(
                self.current_frame, frame, 
                self.feature_extractor, self.feature_matcher
            )
            
            self.num_matches = len(matches)
            
            if len(matches) < self.min_initialization_matches:
                print(f"Not enough matches for initialization: {len(matches)} < {self.min_initialization_matches}")
                return False
            
            # Estimate pose using PnP
            success, rvec, tvec, inliers = self.motion_estimator.estimate_pose_3d_2d(
                points_3d_1, points_2d_2
            )
            
            if success and inliers is not None:
                self.num_inliers = len(inliers)
                
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()
                
                # Set frame pose (current frame pose relative to previous frame)
                frame.set_pose(self.current_frame.get_pose() @ T)
                
                # Switch to tracking state
                self.state = VOState.TRACKING
                
                print(f"VO initialized successfully!")
                print(f"Matches: {self.num_matches}, Inliers: {self.num_inliers}")
                
                return True
            else:
                print("Failed to estimate initial pose")
                return False
    
    def _track(self, frame: Frame) -> bool:
        """Track camera motion using the new frame"""
        if self.current_frame is None:
            self.state = VOState.LOST
            return False
        
        if len(frame.keypoints) == 0 or len(self.current_frame.keypoints) == 0:
            print("No features detected, tracking lost")
            self.state = VOState.LOST
            return False
        
        # Match features with previous frame
        points_3d_1, points_2d_2, points_3d_2, matches = match_frames(
            self.current_frame, frame,
            self.feature_extractor, self.feature_matcher
        )
        
        self.num_matches = len(matches)
        
        if len(matches) < 10:  # Minimum matches for tracking
            print(f"Not enough matches for tracking: {len(matches)}")
            self.state = VOState.LOST
            return False
        
        # Estimate pose using PnP (3D points from previous frame, 2D points from current frame)
        success, rvec, tvec, inliers = self.motion_estimator.estimate_pose_3d_2d(
            points_3d_1, points_2d_2
        )
        
        if success and inliers is not None:
            self.num_inliers = len(inliers)
            
            # Check if we have enough inliers
            if self.num_inliers < self.motion_estimator.min_inliers:
                print(f"Not enough inliers: {self.num_inliers}")
                self.state = VOState.LOST
                return False
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            
            # Set frame pose
            frame.set_pose(self.current_frame.get_pose() @ T)
            
            return True
        else:
            print("Failed to estimate pose")
            self.state = VOState.LOST
            return False
    
    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get current camera pose"""
        if self.current_frame is not None:
            return self.current_frame.get_pose()
        return None
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get camera trajectory as list of 3D points"""
        return self.trajectory.copy()
    
    def get_poses(self) -> List[np.ndarray]:
        """Get all camera poses"""
        return self.poses.copy()
    
    def get_state(self) -> VOState:
        """Get current VO state"""
        return self.state
    
    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        return {
            'state': self.state.value,
            'num_matches': self.num_matches,
            'num_inliers': self.num_inliers,
            'processing_time': self.processing_time,
            'trajectory_length': len(self.trajectory)
        }
    
    def reset(self):
        """Reset the VO system"""
        self.state = VOState.INITIALIZING
        self.current_frame = None
        self.previous_frame = None
        self.trajectory = []
        self.poses = []
        self.num_inliers = 0
        self.num_matches = 0
        self.processing_time = 0.0