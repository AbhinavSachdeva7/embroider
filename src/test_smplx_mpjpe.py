import numpy as np
import torch
import smplx
import os
import json
from tqdm import tqdm

def extract_joint_positions(pt_path, model_path):
    """
    Load a pose from a .pt file, create an SMPLX model, and extract joint positions.
    
    Args:
        pt_path: Path to .pt file containing a single pose tensor.
        model_path: Path to SMPLX model files directory.
    
    Returns:
        Joint positions as numpy array of shape (num_joints, 3)
    """
    
    # Load pose parameters from .pt file
    single_pose = torch.load(pt_path)
    
    # Reshape to batch format (1, num_params) if it's not already
    if len(single_pose.shape) == 1:
        single_pose = single_pose.unsqueeze(0)
    
    # Initialize SMPLX model
    device = torch.device('cpu')
    
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
    
    # Convert to torch tensor
    pose_tensor = single_pose.to(device, dtype=torch.float32)
    
    # Handle 66-parameter format (22 joints × 3 dimensions)
    batch_size = 1
    
    if pose_tensor.shape[1] == 66:
        # First 3 parameters are global orientation
        global_orient = pose_tensor[:, :3]
        # Remaining 63 parameters are body pose (21 joints × 3 dimensions)
        body_pose_params = pose_tensor[:, 3:66]  # 63 parameters
        body_pose = body_pose_params.reshape(batch_size, 21, 3)
        
        
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
    
    # Extract joint positions (SMPLX has joints attribute)
    joint_positions = output.joints.detach().cpu().numpy()[0]  # Take first (and only) batch
    
    return joint_positions

def calculate_mpjpe(joints1, joints2):
    """
    Calculate Mean Per Joint Position Error (MPJPE) between two sets of joint positions.
    
    Args:
        joints1: Joint positions of shape (num_joints, 3)
        joints2: Joint positions of shape (num_joints, 3)
    
    Returns:
        MPJPE value in millimeters
    """
    # Calculate Euclidean distance for each joint
    joint_errors = np.linalg.norm(joints1 - joints2, axis=1)
    
    # Calculate mean error
    mpjpe = np.mean(joint_errors)
    
    # Convert to millimeters (assuming input is in meters)
    mpjpe_mm = mpjpe * 1000
    
    return mpjpe_mm, joint_errors * 1000  # Return both mean and per-joint errors

def calculate_mpjpe_for_all_poses():
    """
    Calculate MPJPE for all poses against the query pose and save results.
    """
    # Base path for visualization results
    base_path = '/scratch/avs7793/work_done/poseembroider/visualization_results/image+pressure_to_pose'
    smplx_model_path = '/scratch/avs7793/models/smplx'  # Path to SMPLX model directory

    # Folder categories to iterate over
    folders = [
        "best_rank_1",
        "good_rank_2_5", 
        "moderate_rank_6_10",
        "poor_rank_gt_10",
        "random",
    ]

    # Dictionary to store all MPJPE results
    mpjpe_results = {}

    # Loop over each folder category
    for folder_name in folders:
        print(f"\nProcessing folder: {folder_name}")
        
        # Load query pose
        query_pose_path = os.path.join(base_path, folder_name, 'query_pose.pt')
        
        if not os.path.exists(query_pose_path):
            print(f"Query pose not found: {query_pose_path}")
            continue
            
        try:
            query_joints = extract_joint_positions(query_pose_path, smplx_model_path)
            print(f"Query pose loaded. Joint positions shape: {query_joints.shape}")
        except Exception as e:
            print(f"Error loading query pose {query_pose_path}: {e}")
            continue
        
        # Initialize results for this folder
        mpjpe_results[folder_name] = {
            'query_pose_path': query_pose_path,
            'ranks': {}
        }
        
        # Loop over ranks from 1 to 10
        for i in range(1, 11):
            rank_str = f"rank_{i:02d}"
            
            # Construct path to rank pose
            pt_file_path = os.path.join(base_path, folder_name, rank_str, 'pose.pt')
            
            if not os.path.exists(pt_file_path):
                print(f"Rank pose not found: {pt_file_path}")
                continue
                
            try:
                # Extract joint positions for this rank
                rank_joints = extract_joint_positions(pt_file_path, smplx_model_path)
                
                # Calculate MPJPE against query pose
                mpjpe_mean, joint_errors = calculate_mpjpe(query_joints, rank_joints)
                
                # Store results
                mpjpe_results[folder_name]['ranks'][rank_str] = {
                    'pose_path': pt_file_path,
                    'mpjpe_mm': float(mpjpe_mean),
                    'per_joint_errors_mm': joint_errors.tolist()
                }
                
                print(f"  {rank_str}: MPJPE = {mpjpe_mean:.2f} mm")
                
            except Exception as e:
                print(f"Error processing {pt_file_path}: {e}")
                mpjpe_results[folder_name]['ranks'][rank_str] = {
                    'pose_path': pt_file_path,
                    'error': str(e)
                }

    # Save results to JSON file
    output_file = os.path.join(base_path, 'mpjpe_results.json')
    with open(output_file, 'w') as f:
        json.dump(mpjpe_results, f, indent=2)
    
    print(f"\nMPJPE results saved to: {output_file}")
    
    # Also save a summary file with just the mean MPJPE values
    summary_results = {}
    for folder_name, folder_data in mpjpe_results.items():
        summary_results[folder_name] = {}
        for rank_str, rank_data in folder_data.get('ranks', {}).items():
            if 'mpjpe_mm' in rank_data:
                summary_results[folder_name][rank_str] = rank_data['mpjpe_mm']
    
    summary_file = os.path.join(base_path, 'mpjpe_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"MPJPE summary saved to: {summary_file}")
    
    # Print overall statistics
    print("\n=== MPJPE Summary ===")
    for folder_name, folder_data in summary_results.items():
        if folder_data:
            mpjpe_values = list(folder_data.values())
            print(f"{folder_name}:")
            print(f"  Mean MPJPE: {np.mean(mpjpe_values):.2f} mm")
            print(f"  Min MPJPE: {np.min(mpjpe_values):.2f} mm")
            print(f"  Max MPJPE: {np.max(mpjpe_values):.2f} mm")
            print(f"  Std MPJPE: {np.std(mpjpe_values):.2f} mm")

if __name__ == "__main__":
    calculate_mpjpe_for_all_poses() 