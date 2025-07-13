import torch
from tqdm import tqdm

def combine_modalities(image_pt_path: str, pose_pt_path: str, pressure_pt_path: str, output_path: str):
    """
    Combine image, pose, and pressure data into synchronized triplets.
    
    Args:
        image_pt_path: Path to images .pt file
        pose_pt_path: Path to poses .pt file  
        pressure_pt_path: Path to pressure .pt file
        output_path: Path to save combined data
    """
    print("Loading data files...")
    
    # Load all three modalities
    images = torch.load(image_pt_path)
    poses = torch.load(pose_pt_path)
    pressure = torch.load(pressure_pt_path)
    
    print(f"Loaded {len(images)} images")
    print(f"Loaded {len(poses)} poses") 
    print(f"Loaded {len(pressure)} pressure maps")
    
    # Check that all have the same length (synchronized)
    min_length = min(len(images), len(poses), len(pressure))
    if not (len(images) == len(poses) == len(pressure)):
        print(f"WARNING: Mismatched lengths. Using minimum length: {min_length}")
        images = images[:min_length]
        poses = poses[:min_length]
        pressure = pressure[:min_length]
    
    print(f"Creating {min_length} synchronized triplets...")
    
    # Create synchronized triplets
    images_list, poses_list, pressure_list = [], [], []
    for i in tqdm(range(min_length), desc=f"Combining {output_path}"):
        images_list.append(images[i])
        poses_list.append(poses[i])
        pressure_list.append(pressure[i])
    
    combined_data = {
        'image': torch.stack(images_list),      # Shape: (N, 3, 256, 192)
        'pose': torch.stack(poses_list),        # Shape: (N, 66)
        'pressure': torch.stack(pressure_list)  # Shape: (N, 2520)
    }

    # Save combined data
    print(f"Saving combined data to {output_path}")
    torch.save(combined_data, output_path)
    print(f"Successfully saved {len(images_list)} synchronized triplets to {output_path}")
    
    return combined_data

# Usage example
if __name__ == "__main__":
    # all_inputs = [(4,11),(5,8),(6,10),(7,8),(8,10),(9,13),(10,10)] # (sub, end) do 1 again, take 8 of sub 2 is not done, data of frames, & poses is missing for subject 7 take 9
    all_inputs = [(1,9)]
    for sub, end in all_inputs:
        sub = sub
        end = end
        for i in range(1, end + 1):
            
            
            combine_modalities(
                image_pt_path=f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/subject {sub}/frames_subject{sub}_take{i}.pt",
                pose_pt_path=f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/subject {sub}/poses_subject{sub}_take{i}.pt", 
                pressure_pt_path=f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/subject {sub}/pressure_subject{sub}_take{i}.pt",
                output_path=f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/combined/combined_subject{sub}_take{i}.pt"
            )
    
    print("\nAll takes have been processed and combined.")