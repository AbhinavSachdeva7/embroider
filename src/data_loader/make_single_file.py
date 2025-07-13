import torch
import os
import glob
from tqdm import tqdm

def combine_all_takes(input_dir: str, output_path: str):
    
    # Find all .pt files in the input directory
    file_paths = sorted(glob.glob(os.path.join(input_dir, "combined_*.pt")))
    
    if not file_paths:
        print(f"No '.pt' files found in {input_dir}. Exiting.")
        return

    print(f"Found {len(file_paths)} files to combine.")

    total_samples = 0
    image_shape, pose_shape, pressure_shape = None, None, None
    dtype_image, dtype_pose, dtype_pressure = None, None, None

    for file_path in tqdm(file_paths, desc="Pass 1/2: Collecting metadata"):
        # Set weights_only=True for security and to prepare for future PyTorch versions.
        data = torch.load(file_path)
        
        num_in_file = data['image'].shape[0]
        total_samples += num_in_file
        
        # On the first file, grab the shapes and dtypes
        if image_shape is None:
            image_shape = data['image'].shape[1:]
            pose_shape = data['pose'].shape[1:]
            pressure_shape = data['pressure'].shape[1:]
            dtype_image = data['image'].dtype
            dtype_pose = data['pose'].dtype
            dtype_pressure = data['pressure'].dtype
        
        del data # Explicitly release memory

    if total_samples == 0:
        print("No samples found in any files. Exiting.")
        return

    print(f"\nMetadata collected: Total samples = {total_samples}")
    print(f"  Image shape: {image_shape}, Pose shape: {pose_shape}, Pressure shape: {pressure_shape}")

    # --- Pre-allocate final tensors ---
    print("\nPre-allocating final tensors...")
    final_images = torch.empty((total_samples, *image_shape), dtype=dtype_image)
    final_poses = torch.empty((total_samples, *pose_shape), dtype=dtype_pose)
    final_pressures = torch.empty((total_samples, *pressure_shape), dtype=dtype_pressure)

    # --- Pass 2: Load data and fill the tensors ---
    current_pos = 0
    for file_path in tqdm(file_paths, desc="Pass 2/2: Loading and filling data"):
        data = torch.load(file_path, weights_only=True)
        
        num_in_file = data['image'].shape[0]
        
        # Copy data into the correct slice of the pre-allocated tensors
        final_images[current_pos : current_pos + num_in_file] = data['image']
        final_poses[current_pos : current_pos + num_in_file] = data['pose']
        final_pressures[current_pos : current_pos + num_in_file] = data['pressure']
        
        current_pos += num_in_file
        del data # Explicitly release memory

    # Create the final combined dictionary
    final_data = {
        'image': final_images,
        'pose': final_poses,
        'pressure': final_pressures
    }

    # Save the single, large data file
    print(f"\nSaving final combined data to {output_path}...")
    torch.save(final_data, output_path)
    
    print("\nSuccessfully combined all data into a single file.")
    print(f"Final image tensor shape: {final_images.shape}")
    print(f"Final pose tensor shape: {final_poses.shape}")
    print(f"Final pressure tensor shape: {final_pressures.shape}")


if __name__ == '__main__':
    # Directory where the output of make_triplets.py is stored
    input_directory = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/combined/"
    
    # Path for the final output file
    output_file = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/ALL_DATA.pt"
    
    combine_all_takes(input_dir=input_directory, output_path=output_file)