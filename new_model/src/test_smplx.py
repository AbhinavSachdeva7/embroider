import numpy as np
import matplotlib
import torch
import smplx
import trimesh
import cv2
import os
from tqdm import tqdm

matplotlib.use('Agg')

import matplotlib.pyplot as plt

def visualize_smplx_pose(pt_path, model_path, output_path_prefix):
    """
    Load a single pose from a .pt file, create an SMPLX model, and save visualizations from two 3D views.
    
    Args:
        pt_path: Path to .pt file containing a single pose tensor.
        model_path: Path to SMPLX model files directory.
        output_path_prefix: Prefix for the output image paths.
    """
    
    # Load pose parameters from .pt file
    single_pose = torch.load(pt_path)
    
    print(f'Pose parameters shape: {single_pose.shape}')
    
    # Reshape to batch format (1, num_params) if it's not already
    if len(single_pose.shape) == 1:
        single_pose = single_pose.unsqueeze(0)
    
    # Initialize SMPLX model
    device = torch.device('cpu')
    print(f'Using device: {device}')
    
    model = smplx.SMPLX(
        model_path=model_path,
        gender='neutral',
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
    ).to(device)
    
    # Convert to torch tensor (already a tensor, but ensure it's on the right device and dtype)
    pose_tensor = single_pose.to(device, dtype=torch.float32)
    
    # Handle 55-parameter format by mapping to SMPLX parameter structure
    batch_size = 1
    
    if pose_tensor.shape[1] == 55:
        # Map the 55 parameters to SMPLX format
        global_orient = pose_tensor[:, :3]
        # Take the remaining 52 parameters for body pose
        remaining_params = pose_tensor[:, 3:]
        body_pose = torch.zeros(batch_size, 21, 3).to(device)
        
        # Map remaining parameters to body joints (52 params = ~17 joints * 3)
        n_params = min(remaining_params.shape[1], 21*3)
        n_joints = n_params // 3
        body_pose[:, :n_joints, :] = remaining_params[:, :n_joints*3].reshape(batch_size, n_joints, 3)
        
    elif pose_tensor.shape[1] == 72:  # Standard SMPL format
        global_orient = pose_tensor[:, :3]
        body_pose = pose_tensor[:, 3:72].reshape(batch_size, 23, 3)[:, :21, :]  # Take first 21 joints
        
    else:
        # Handle other parameter counts
        global_orient = pose_tensor[:, :3] if pose_tensor.shape[1] >= 3 else torch.zeros(batch_size, 3).to(device)
        remaining = pose_tensor[:, 3:] if pose_tensor.shape[1] > 3 else torch.zeros(batch_size, 0).to(device)
        body_pose = torch.zeros(batch_size, 21, 3).to(device)
        if remaining.shape[1] > 0:
            n_joints = min(remaining.shape[1] // 3, 21)
            body_pose[:, :n_joints, :] = remaining[:, :n_joints*3].reshape(batch_size, n_joints, 3)
    
    print(f'Global orient shape: {global_orient.shape}')
    print(f'Body pose shape: {body_pose.shape}')
    
    # Generate mesh with SMPLX model
    output = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=torch.zeros(batch_size, 10).to(device),  # Default shape
        expression=torch.zeros(batch_size, 10).to(device),  # Default expression
        left_hand_pose=torch.zeros(batch_size, 15, 3).to(device),  # Default hand pose
        right_hand_pose=torch.zeros(batch_size, 15, 3).to(device),  # Default hand pose
        jaw_pose=torch.zeros(batch_size, 3).to(device),  # Default jaw
        leye_pose=torch.zeros(batch_size, 3).to(device),  # Default eye
        reye_pose=torch.zeros(batch_size, 3).to(device),  # Default eye
        return_verts=True
    )
    
    # Extract vertices
    vertices = output.vertices.detach().cpu().numpy()[0]  # Take first (and only) batch
    faces = model.faces

    # Create visualizations for two different 3D views
    views = [
        (45, 45, '3D View 1'),
        (-45, 45, '3D View 2')
    ]

    for azim, elev, title in views:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface model
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=[0.8, 0.8, 1.0], alpha=0.9)
        
        ax.set_title(f'{title}')
        ax.view_init(elev=elev, azim=azim)
        
        # Set equal aspect ratio
        max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
        mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Remove axis labels for cleaner look
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
    
        # Construct output path and save image
        view_name = title.lower().replace(' ', '_')
        output_image_path = f"{output_path_prefix}_{view_name}.png"
        
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f'SMPLX pose visualization saved as: {output_image_path}')

    print(f'Vertices shape: {vertices.shape}')
    
    return vertices





# Usage example
if __name__ == "__main__":
    # Base path for visualization results
    base_path = '/scratch/avs7793/work_done/poseembroider/visualization_results/image_to_pressure'
    smplx_model_path = '/scratch/avs7793/models/smplx'  # Path to SMPLX model directory

    # Folder categories to iterate over
    folders = [
        "best_rank_1",
        "good_rank_2_5",
        "moderate_rank_6_10",
        "poor_rank_gt_10",
        "random",
    ]

    # Loop over each folder category
    for folder_name in folders:
        # Loop over ranks from 1 to 10
        for i in range(1, 12):
            # Format rank number with leading zero (e.g., 01, 02, ..., 10)
            rank_str = f"rank_{i:02d}"
            
            # Construct paths
            if i == 11:
                pt_file_path = os.path.join(base_path, folder_name, 'query_pose.pt')
                output_prefix = os.path.join(base_path, folder_name, 'smplx_pose_solid_render')
            else:
                pt_file_path = os.path.join(base_path, folder_name, rank_str, 'pose.pt')
                output_prefix = os.path.join(base_path, folder_name, rank_str, 'smplx_pose_solid_render')

            print(f"Processing: {pt_file_path}")

            try:
                # Check if the pose file exists before trying to process it
                if os.path.exists(pt_file_path):
                    # visualize_smplx_pose(
                    #     pt_file_path,
                    #     smplx_model_path,
                    #     output_prefix
                    # )
                    pose = torch.load(pt_file_path)
                    print(pose.shape)
                else:
                    print(f"File not found, skipping: {pt_file_path}")
            except Exception as e:
                print(f"Error processing {pt_file_path}: {e}")
                print("\nMake sure you have:")
                print("1. Downloaded SMPLX model files from https://smpl-x.is.tue.mpg.de/")
                print("2. Correct paths to your .pt file and model directory")


