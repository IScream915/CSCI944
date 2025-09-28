import os
import numpy as np
from typing import List, Tuple
import cv2

class TUMDataLoader:
    def __init__(self, dataset_path: str, rgb_list_file: str = "rgb.txt", 
                 depth_list_file: str = "depth.txt"):
        self.dataset_path = dataset_path
        self.rgb_list_file = rgb_list_file
        self.depth_list_file = depth_list_file
        
        self.rgb_list = []
        self.depth_list = []
        
        self._load_file_lists()
        self._associate_rgb_depth()
    
    def _load_file_lists(self):
        """Load RGB and depth file lists with timestamps"""
        # Load RGB files
        rgb_file_path = os.path.join(self.dataset_path, self.rgb_list_file)
        if not os.path.exists(rgb_file_path):
            raise FileNotFoundError(f"RGB list file not found: {rgb_file_path}")
        
        with open(rgb_file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    filename = parts[1]
                    self.rgb_list.append((timestamp, filename))
        
        # Load depth files
        depth_file_path = os.path.join(self.dataset_path, self.depth_list_file)
        if not os.path.exists(depth_file_path):
            raise FileNotFoundError(f"Depth list file not found: {depth_file_path}")
        
        with open(depth_file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    filename = parts[1]
                    self.depth_list.append((timestamp, filename))
    
    def _associate_rgb_depth(self, max_time_diff: float = 0.02):
        """Associate RGB and depth images based on timestamps"""
        self.associated_pairs = []
        
        for rgb_timestamp, rgb_file in self.rgb_list:
            best_match = None
            best_time_diff = float('inf')
            
            for depth_timestamp, depth_file in self.depth_list:
                time_diff = abs(rgb_timestamp - depth_timestamp)
                if time_diff < best_time_diff and time_diff <= max_time_diff:
                    best_time_diff = time_diff
                    best_match = (depth_timestamp, depth_file)
            
            if best_match is not None:
                self.associated_pairs.append({
                    'timestamp': rgb_timestamp,
                    'rgb_file': rgb_file,
                    'depth_file': best_match[1],
                    'time_diff': best_time_diff
                })
        
        print(f"Associated {len(self.associated_pairs)} RGB-D pairs")
    
    def get_frame_count(self) -> int:
        """Get total number of associated frames"""
        return len(self.associated_pairs)
    
    def get_frame_data(self, index: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Get RGB and depth images for a specific frame index
        
        Returns:
            timestamp: Frame timestamp
            rgb_image: RGB image as numpy array
            depth_image: Depth image as numpy array
        """
        if index >= len(self.associated_pairs):
            raise IndexError(f"Frame index {index} out of range")
        
        frame_data = self.associated_pairs[index]
        
        # Load RGB image
        rgb_path = os.path.join(self.dataset_path, frame_data['rgb_file'])
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        rgb_image = cv2.imread(rgb_path)
        
        # Load depth image
        depth_path = os.path.join(self.dataset_path, frame_data['depth_file'])
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        return frame_data['timestamp'], rgb_image, depth_image
    
    def get_all_timestamps(self) -> List[float]:
        """Get all frame timestamps"""
        return [pair['timestamp'] for pair in self.associated_pairs]