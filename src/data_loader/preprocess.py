import torch
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import multiprocessing


image_transform = transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def _extract_and_process_frames(video_path: str):
        """Extract frames from video and preprocess them."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(f"Total frames to process: {total_frames}")
        
        # Create progress bar
        # with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply preprocessing (resize, normalize, etc.)
            processed_frame = image_transform(pil_image)
            frames.append(processed_frame)
            
            frame_count += 1
            # pbar.update(1)  # Update progress bar
            
        cap.release()
        print(f"Successfully extracted {frame_count} frames")
        return frames


def _load_and_process_poses(pose_path: str):
    """Load and process pose data."""
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    
    # Load poses (assuming numpy array)
    poses_data = np.load(pose_path)
    poses_np = poses_data['poses']
    print(f"Loaded poses with shape: {poses_np.shape}")
    
    # Convert to tensor and ensure correct shape
    # Assuming input is (n_frames, 22, 3) - adjust as needed for your data
    poses_tensor = torch.from_numpy(poses_np).float()
    
    # Flatten each pose to (66,) for model input
    poses_tensor = poses_tensor[:,:66]
    print(f"Processed poses with shape: {poses_tensor.shape}")

    poses_flattened = []
    for i in range(poses_tensor.shape[0]):
        # pose_flat = poses_tensor[i].view(-1)  # (22, 3) -> (66,)
        poses_flattened.append(poses_tensor[i])

    # print(f"Flattened poses with shape: {len(poses_flattened)}")
    return poses_flattened


def process_videos(start, finish, subject):
	for i in tqdm(range(start,finish), desc=f"Processing videos for subject {subject}", position=0):
		video_path = f"/scratch/avs7793/PSU100/Modality_wise/Video/Subject{subject}/Video_V2_{i}.mp4"
		frames = _extract_and_process_frames(video_path)
		torch.save(frames, f"frames_subject{subject}_take{i}_view2.pt") 

def process_poses(start, finish, subject):
	for i in tqdm(range(start,finish), desc=f"Processing poses for subject {subject}", position=1):
		pose_path = f"/scratch/avs7793/soma/work/SOMA/amass_npzs/gt/Subject{subject}/Subject{subject}_MOCAP_MRK_{i}_gt_stageii.npz"
		poses = _load_and_process_poses(pose_path)
		torch.save(poses, f"poses_subject{subject}_take{i}.pt")


if __name__ == "__main__":
	# video_path = "/scratch/avs7793/PSU100/Modality_wise/Video/Subject1/Video_V2_1.mp4"
	# pose_path = "/scratch/avs7793/soma/work/SOMA/amass_npzs/gt/Subject1/Subject1_MOCAP_MRK_2_gt_stageii.npz"
    # # frames = _extract_and_process_frames(video_path)
	# poses = _load_and_process_poses(pose_path)
    # # print(f"Total frames extracted: {len(frames)}")
	# print(f"Total poses extracted: {len(poses)}")
    # # torch.save(frames, "frames_subject1_take1.pt")
	# torch.save(poses, "poses_subject1_take2.pt")
    # p_videos = multiprocessing.Process(target=process_videos, args=(1,9,5))
    p_poses = multiprocessing.Process(target=process_poses, args=(1,9,5))
  
    # p_videos.start()
    p_poses.start()

    # p_videos.join()
    p_poses.join()

    print("Done")
