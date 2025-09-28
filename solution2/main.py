#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import cv2
import time
import numpy as np
from typing import Dict, Any

# Import our modules
from data_loader import TUMDataLoader
from frame import Frame, Camera
from visual_odometry import VisualOdometry, VOState
from visualizer import OpenCVVisualizer

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def create_camera_from_config(config: Dict[str, Any]) -> Camera:
    """Create camera object from configuration"""
    camera_config = config['camera']
    return Camera(
        fx=camera_config['fx'],
        fy=camera_config['fy'],
        cx=camera_config['cx'],
        cy=camera_config['cy'],
        depth_scale=camera_config['depth_scale']
    )

def print_statistics(frame_id: int, vo_stats: Dict[str, Any], processing_time: float):
    """Print frame processing statistics"""
    print(f"Frame {frame_id:4d}: "
          f"State={vo_stats['state']:12s} "
          f"Matches={vo_stats['num_matches']:3d} "
          f"Inliers={vo_stats['num_inliers']:3d} "
          f"Time={processing_time:.3f}s")

def main():
    parser = argparse.ArgumentParser(description='Visual Odometry System')
    parser.add_argument('config_file', help='Path to configuration file')
    parser.add_argument('--max_frames', type=int, default=None, 
                       help='Maximum number of frames to process')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Starting frame index')
    parser.add_argument('--save_trajectory', type=str, default=None,
                       help='Save trajectory to file')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--step_mode', action='store_true',
                       help='Step through frames manually (press any key to continue)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config_file}")
    config = load_config(args.config_file)
    
    # Create data loader
    dataset_path = config['dataset_path']
    if not os.path.isabs(dataset_path):
        # Make path relative to config file directory
        config_dir = os.path.dirname(os.path.abspath(args.config_file))
        dataset_path = os.path.join(config_dir, dataset_path)
    
    print(f"Loading dataset from: {dataset_path}")
    try:
        data_loader = TUMDataLoader(
            dataset_path=dataset_path,
            rgb_list_file=config['rgb_list_file'],
            depth_list_file=config['depth_list_file']
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Create camera
    camera = create_camera_from_config(config)
    print(f"Camera created: fx={camera.fx}, fy={camera.fy}, cx={camera.cx}, cy={camera.cy}")
    
    # Create visual odometry system
    vo = VisualOdometry(camera, config)
    print("Visual Odometry system initialized")
    
    # Create visualizer
    visualizer = None
    if not args.no_visualization:
        visualizer = OpenCVVisualizer(config)
        visualizer.start_visualization()
        print("Visualization started")
    
    # Get frame count and set processing range
    total_frames = data_loader.get_frame_count()
    start_frame = max(0, args.start_frame)
    end_frame = min(total_frames, start_frame + args.max_frames) if args.max_frames else total_frames
    
    print(f"Processing frames {start_frame} to {end_frame-1} ({end_frame - start_frame} frames)")
    print("-" * 70)
    
    # Main processing loop
    frame_count = 0
    successful_frames = 0
    total_processing_time = 0.0
    
    try:
        for frame_idx in range(start_frame, end_frame):
            start_time = time.time()
            
            # Load frame data
            try:
                timestamp, rgb_image, depth_image = data_loader.get_frame_data(frame_idx)
            except Exception as e:
                print(f"Error loading frame {frame_idx}: {e}")
                continue
            
            # Create frame object
            frame = Frame(timestamp, rgb_image, depth_image, camera, frame_idx)
            
            # Process frame with VO
            vo_start_time = time.time()
            success = vo.process_frame(frame)
            vo_processing_time = time.time() - vo_start_time
            
            # Get statistics
            vo_stats = vo.get_statistics()
            total_processing_time += vo_processing_time
            
            # Print statistics
            print_statistics(frame_idx, vo_stats, vo_processing_time)
            
            if success:
                successful_frames += 1
                
                # Update visualization
                if visualizer:
                    current_pose = vo.get_current_pose()
                    if current_pose is not None:
                        visualizer.update_pose(current_pose)
                    visualizer.update_image(rgb_image)
            else:
                print(f"Visual Odometry LOST at frame {frame_idx}")
                break
            
            frame_count += 1
            
            # Handle visualization and user input
            if not args.no_visualization:
                key = cv2.waitKey(1 if not args.step_mode else 0) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC to quit
                    print("User requested exit")
                    break
                elif key == ord('s') and args.save_trajectory:  # 's' to save trajectory
                    if visualizer:
                        visualizer.save_trajectory(args.save_trajectory)
            
            # Check VO state
            if vo.get_state() == VOState.LOST:
                print("Visual Odometry system lost tracking")
                break
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Print final statistics
        print("-" * 70)
        print(f"Processing completed:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Successful frames: {successful_frames}")
        print(f"  Success rate: {successful_frames/frame_count*100:.1f}%" if frame_count > 0 else "N/A")
        print(f"  Average processing time: {total_processing_time/frame_count:.3f}s" if frame_count > 0 else "N/A")
        print(f"  Total processing time: {total_processing_time:.3f}s")
        
        # Get final trajectory
        trajectory = vo.get_trajectory()
        if len(trajectory) > 0:
            print(f"  Trajectory length: {len(trajectory)} points")
            trajectory_array = np.array(trajectory)
            distance = np.sum(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))
            print(f"  Total distance traveled: {distance:.2f}m")
        
        # Save trajectory if requested
        if args.save_trajectory and visualizer and len(trajectory) > 0:
            visualizer.save_trajectory(args.save_trajectory)
        
        # Clean up visualization
        if visualizer:
            print("Press any key to close visualization windows...")
            cv2.waitKey(0)
            visualizer.stop_visualization()
        
        print("Program finished")

if __name__ == "__main__":
    main()