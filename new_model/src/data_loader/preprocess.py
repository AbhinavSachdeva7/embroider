import torch
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

# --- Configuration ---
# Root directory for all processed data
OUTPUT_ROOT = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed"
# Source data paths
SOURCE_VIDEO_ROOT = "/scratch/avs7793/PSU100/Modality_wise/Video"
SOURCE_POSE_ROOT = "/scratch/avs7793/soma/work/SOMA/amass_npzs/gt"

# Define which subjects and takes to process
# Format: {"subject_id_string": range(start_take, end_take)}
SUBJECT_TAKE_MAPPING = {
    # "1": range(1, 10),  # Process Subject 1, takes 1 through 9
    "2": range(1, 7),   # Process Subject 5, takes 1 through 8
    # "3": range(1, 10),
    "4": range(1, 12),
    "5": range(1, 9),
    "6": range(1, 11),
    "7": range(1, 10),
    "8": range(1, 11),
    "9": range(1, 14),
    "10": range(1, 11),

    # 2,7 3,10  4,12  5,9  6,11   7,10  8,11  9,14  10, 11
    # Add other subjects and their take ranges here
}

IMAGE_SIZE = (192, 256)  # (Width, Height) for cv2.resize

# Setup basic logging to see progress and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_and_save_frames(video_path: str, image_output_dir: str):
    """Extracts frames from a video, resizes them, and saves as PNG files."""
    if not os.path.exists(video_path):
        logging.warning(f"Video not found, skipping: {video_path}")
        return
    
    os.makedirs(image_output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame and convert BGR (cv2) to RGB for standard saving
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, IMAGE_SIZE)
        
        # Save as a PNG image
        img = Image.fromarray(resized_frame)
        output_filename = os.path.join(image_output_dir, f"{frame_count:05d}.png")
        img.save(output_filename)
        
        frame_count += 1
            
    cap.release()
    logging.info(f"Saved {frame_count} frames from {os.path.basename(video_path)}")

def process_and_save_poses(pose_path: str, pose_output_path: str):
    """Loads poses from an .npz file, processes, and saves as a .pt file."""
    if not os.path.exists(pose_path):
        logging.warning(f"Pose file not found, skipping: {pose_path}")
        return

    os.makedirs(os.path.dirname(pose_output_path), exist_ok=True)

    poses_data = np.load(pose_path)
    poses_np = poses_data['poses']
    print(f"Loaded poses with shape: {poses_np.shape}")
    
    # Convert to tensor and ensure correct shape
    # Assuming input is (n_frames, 22, 3) - adjust as needed for your data
    poses_tensor = torch.from_numpy(poses_np).float()
    
    # Ensure poses are flattened to the first 66 features, matching original logic
    if poses_tensor.shape[1] >= 66:
        poses_tensor = poses_tensor[:, :66]
    else:
        # Pad with zeros if a sample has fewer than 66 features
        logging.warning(f"Pose tensor in {pose_path} has < 66 features. Padding with zeros.")
        pad_size = 66 - poses_tensor.shape[1]
        padding = torch.zeros(poses_tensor.shape[0], pad_size)
        poses_tensor = torch.cat([poses_tensor, padding], dim=1)

    torch.save(poses_tensor, pose_output_path)
    logging.info(f"Saved {poses_tensor.shape[0]} processed poses to {os.path.basename(pose_output_path)}")

def extract_and_save_frames_wrapper(args):
    """Wrapper to unpack arguments for image processing pool."""
    video_path, image_output_dir = args
    try:
        extract_and_save_frames(video_path, image_output_dir)
    except Exception as e:
        logging.error(f"Error extracting frames from {video_path}: {e}")

def process_and_save_poses_wrapper(args):
    """Wrapper to unpack arguments for pose processing pool."""
    pose_path, pose_output_path = args
    try:
        process_and_save_poses(pose_path, pose_output_path)
    except Exception as e:
        logging.error(f"Error processing poses from {pose_path}: {e}")

def main():
    """Orchestrates the preprocessing of all configured subjects and takes."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # --- Create distinct lists for image and pose processing tasks ---
    image_tasks = []
    pose_tasks = []
    for subject_id, take_range in SUBJECT_TAKE_MAPPING.items():
        for take_id in take_range:
            # Add image processing task
            video_path = os.path.join(SOURCE_VIDEO_ROOT, f"Subject{subject_id}", f"Video_V2_{take_id}.mp4")
            image_output_dir = os.path.join(OUTPUT_ROOT, "images", f"subject_{subject_id}", f"take_{take_id}")
            image_tasks.append((video_path, image_output_dir))

            # Add pose processing task
            pose_path = os.path.join(SOURCE_POSE_ROOT, f"Subject{subject_id}", f"Subject{subject_id}_MOCAP_MRK_{take_id}_gt_stageii.npz")
            pose_output_dir = os.path.join(OUTPUT_ROOT, "poses")
            pose_output_path = os.path.join(pose_output_dir, f"poses_subject{subject_id}_take{take_id}.pt")
            pose_tasks.append((pose_path, pose_output_path))

    if not image_tasks and not pose_tasks:
        logging.warning("No tasks to process. Check SUBJECT_TAKE_MAPPING configuration.")
        return
        
    logging.info(f"Generated {len(image_tasks)} image tasks and {len(pose_tasks)} pose tasks.")
    
    # --- Use a single process pool to run all tasks concurrently ---
    # The pool will work on both fast (pose) and slow (image) tasks at the same time.
    num_processes = 6
    with Pool(processes=num_processes) as pool:
        # Submit pose tasks
        pose_results = pool.map_async(process_and_save_poses_wrapper, pose_tasks)
        logging.info("Submitted all pose processing tasks to the pool.")

        # Submit image tasks
        image_results = pool.map_async(extract_and_save_frames_wrapper, image_tasks)
        logging.info("Submitted all image processing tasks to the pool.")

        # Wait for all tasks to complete
        pose_results.get() # Wait for all pose tasks to finish
        logging.info("--- All pose processing tasks are complete. ---")
        
        image_results.get() # Wait for all image tasks to finish
        logging.info("--- All image processing tasks are complete. ---")

    logging.info("--- All preprocessing tasks complete. ---")

if __name__ == "__main__":
    main()
