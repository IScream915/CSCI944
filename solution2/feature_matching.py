import cv2
import numpy as np
from typing import Tuple, List, Optional

class FeatureExtractor:
    def __init__(self, detector_type: str = "ORB", **kwargs):
        self.detector_type = detector_type
        
        if detector_type == "ORB":
            n_features = kwargs.get('n_features', 1000)
            scale_factor = kwargs.get('scale_factor', 1.2)
            n_levels = kwargs.get('n_levels', 8)
            self.detector = cv2.ORB_create(
                nfeatures=n_features,
                scaleFactor=scale_factor,
                nlevels=n_levels
            )
        elif detector_type == "SIFT":
            n_features = kwargs.get('n_features', 1000)
            self.detector = cv2.SIFT_create(nfeatures=n_features)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect keypoints and compute descriptors"""
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors


class FeatureMatcher:
    def __init__(self, matcher_type: str = "BF", distance_threshold: float = 0.7):
        self.matcher_type = matcher_type
        self.distance_threshold = distance_threshold
        
        if matcher_type == "BF":
            # Use Hamming distance for ORB, L2 distance for SIFT
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif matcher_type == "FLANN":
            # FLANN matcher for SIFT features
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unsupported matcher type: {matcher_type}")
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """Match features between two descriptor sets"""
        if desc1 is None or desc2 is None:
            return []
        
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        # Use knnMatch to get the best 2 matches for each descriptor
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.distance_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def filter_matches_by_distance(self, matches: List[cv2.DMatch], 
                                  max_distance: float = 50.0) -> List[cv2.DMatch]:
        """Filter matches by descriptor distance"""
        return [m for m in matches if m.distance < max_distance]


class MotionEstimator:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 ransac_threshold: float = 1.0, max_iterations: int = 1000,
                 min_inliers: int = 20):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.ransac_threshold = ransac_threshold
        self.max_iterations = max_iterations
        self.min_inliers = min_inliers
    
    def estimate_pose_3d_2d(self, points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate camera pose using 3D-2D point correspondences (PnP)
        
        Returns:
            success: Whether pose estimation succeeded
            rvec: Rotation vector
            tvec: Translation vector
            inliers: Inlier indices
        """
        if len(points_3d) < 4 or len(points_2d) < 4:
            return False, None, None, None
        
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, self.camera_matrix, self.dist_coeffs,
                iterationsCount=self.max_iterations,
                reprojectionError=self.ransac_threshold,
                confidence=0.99
            )
            
            if success and inliers is not None and len(inliers) >= self.min_inliers:
                return True, rvec, tvec, inliers
            else:
                return False, None, None, None
                
        except cv2.error as e:
            print(f"PnP solver error: {e}")
            return False, None, None, None
    
    def estimate_pose_3d_3d(self, points_3d_1: np.ndarray, points_3d_2: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Estimate relative pose using 3D-3D point correspondences (ICP)
        
        Returns:
            success: Whether pose estimation succeeded
            R: Rotation matrix
            t: Translation vector
        """
        if len(points_3d_1) < 3 or len(points_3d_2) < 3:
            return False, None, None
        
        # Compute centroids
        centroid_1 = np.mean(points_3d_1, axis=0)
        centroid_2 = np.mean(points_3d_2, axis=0)
        
        # Center the points
        centered_1 = points_3d_1 - centroid_1
        centered_2 = points_3d_2 - centroid_2
        
        # Compute the cross-covariance matrix
        H = centered_1.T @ centered_2
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_2 - R @ centroid_1
        
        return True, R, t


def match_frames(frame1, frame2, feature_extractor: FeatureExtractor, 
                feature_matcher: FeatureMatcher) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Match features between two frames and return corresponding 3D and 2D points
    
    Returns:
        points_3d_1: 3D points from frame1
        points_2d_2: 2D points from frame2
        points_3d_2: 3D points from frame2
        matches: Good matches
    """
    # Extract features if not already done
    if frame1.keypoints is None:
        frame1.extract_features(feature_extractor.detector)
    if frame2.keypoints is None:
        frame2.extract_features(feature_extractor.detector)
    
    # Match features
    matches = feature_matcher.match_features(frame1.descriptors, frame2.descriptors)
    
    if len(matches) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Extract corresponding points
    points_3d_1 = []
    points_2d_2 = []
    points_3d_2 = []
    good_matches = []
    
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        
        # First check if we can get the 2D keypoint coordinates
        if (idx1 < len(frame1.keypoints) and idx2 < len(frame2.keypoints)):
            # Get keypoint coordinates
            kp1 = frame1.keypoints[idx1]
            kp2 = frame2.keypoints[idx2]
            u1, v1 = int(kp1.pt[0]), int(kp1.pt[1])
            u2, v2 = int(kp2.pt[0]), int(kp2.pt[1])
            
            # Check if coordinates are within bounds and get depth
            if (0 <= u1 < frame1.depth_image.shape[1] and 0 <= v1 < frame1.depth_image.shape[0] and
                0 <= u2 < frame2.depth_image.shape[1] and 0 <= v2 < frame2.depth_image.shape[0]):
                
                depth1 = frame1.depth_image[v1, u1]
                depth2 = frame2.depth_image[v2, u2]
                
                if depth1 > 0 and depth2 > 0:
                    # Convert to 3D points
                    point_3d_1 = frame1.camera.pixel_to_camera(u1, v1, depth1)
                    point_3d_2 = frame2.camera.pixel_to_camera(u2, v2, depth2)
                    
                    if (point_3d_1 is not None and point_3d_2 is not None and
                        point_3d_1[2] > 0.1 and point_3d_2[2] > 0.1 and
                        point_3d_1[2] < 10.0 and point_3d_2[2] < 10.0):
                        
                        points_3d_1.append(point_3d_1)
                        points_2d_2.append([u2, v2])
                        points_3d_2.append(point_3d_2)
                        good_matches.append(match)
    
    return (np.array(points_3d_1, dtype=np.float32),
            np.array(points_2d_2, dtype=np.float32),
            np.array(points_3d_2, dtype=np.float32),
            good_matches)