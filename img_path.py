import os
import json
from new_model.src.data_loader.dataset import ImagePosePressureDataset
import new_model.src.config as config
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import smplx
import trimesh
import cv2
import os
from tqdm import tqdm


def find_path(original_path, dataset, combination, folder):
	base_path = os.path.join('/scratch/avs7793/work_done/poseembroider/new_model/target_results', folder, combination)
	print(f"Processing combination: {combination} from {base_path}")

	image_path = {}

	json_file = os.path.join(base_path, 'summary.json')

	with open(json_file, 'r') as f:
		summary = json.load(f)

	# og_path = original_path

	# image_path["query"] = og_path

	for x in summary['top_3_full_dataset_indices']:
		path = dataset.get_image_path(x)
		image_path[x] = path


	output_json = os.path.join(base_path,"img_path.json")

	with open(output_json, 'w') as f:
		json.dump(image_path, f, indent=4)


def find_foot_pressure_mse(dataset, combination, folder, loss, og_foot_p):
	base_path = os.path.join('/scratch/avs7793/work_done/poseembroider/new_model/target_results', folder, combination)
	print(f"Processing combination: {combination} from {base_path}")

	pressure_mse = {}

	json_file = os.path.join(base_path, 'summary.json')

	with open(json_file, 'r') as f:
		summary = json.load(f)


	for x in summary['top_3_full_dataset_indices']:
		rmse = torch.sqrt(loss(dataset[x]['pressure_map'], dataset[og_foot_p]['pressure_map']))
		pressure_mse[x] = [rmse.item(), x]

	output_json = os.path.join(base_path, 'pressure_rmse.json')
	with open(output_json, 'w') as f:
		json.dump(pressure_mse, f, indent=4)




def visualize_smplx_pose(pt_path, model_path, output_path_prefix, model):
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
    
    model = model.to(device)
    
    # Convert to torch tensor (already a tensor, but ensure it's on the right device and dtype)
    pose_tensor = single_pose.to(device, dtype=torch.float32)
    
    # Handle 55-parameter format by mapping to SMPLX parameter structure
    batch_size = 1
    
    if pose_tensor.shape[1] == 66:
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
        (135, 0, '3D View 1'),
        (-45, 0, '3D View 2')
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
        output_image_path = f"{view_name}.png"
        output_image_path = os.path.join(output_path_prefix,output_image_path)
        
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f'SMPLX pose visualization saved as: {output_image_path}')

    print(f'Vertices shape: {vertices.shape}')
    
    return vertices



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

def calculate_and_save_mpjpe(index, combination, smplx_model_path, output_base_path):
    """
    Calculate MPJPE for all retrieved poses against the query pose and save results.
    """
    print(f"Calculating MPJPE for sample_{index:05d}, combination: {combination}")
    
    # Path to query pose
    query_pose_path = os.path.join(output_base_path, f'sample_{index:05d}', 'query_pose.pt')
    
    if not os.path.exists(query_pose_path):
        print(f"Query pose not found: {query_pose_path}")
        return
    
    try:
        # Extract joint positions for query pose
        query_joints = extract_joint_positions(query_pose_path, smplx_model_path)
        print(f"Query pose loaded. Joint positions shape: {query_joints.shape}")
    except Exception as e:
        print(f"Error loading query pose {query_pose_path}: {e}")
        return
    
    # Initialize results for this combination
    mpjpe_results = {
        'sample_index': index,
        'combination': combination,
        'query_pose_path': query_pose_path,
        'ranks': {}
    }
    
    # Loop over ranks 1-3 (as in your original code)
    for rank in range(1, 4):
        rank_str = f"rank_{rank:02d}"
        
        # Path to retrieved pose
        retrieved_pose_path = os.path.join(output_base_path, f'sample_{index:05d}', combination, rank_str, 'pose.pt')
        
        if not os.path.exists(retrieved_pose_path):
            print(f"Retrieved pose not found: {retrieved_pose_path}")
            continue
        
        try:
            # Extract joint positions for retrieved pose
            retrieved_joints = extract_joint_positions(retrieved_pose_path, smplx_model_path)
            
            # Calculate MPJPE against query pose
            mpjpe_mean, joint_errors = calculate_mpjpe(query_joints, retrieved_joints)
            
            # Store results
            mpjpe_results['ranks'][rank_str] = {
                'pose_path': retrieved_pose_path,
                'mpjpe_mm': float(mpjpe_mean),
                'per_joint_errors_mm': joint_errors.tolist()
            }
            
            print(f"  {rank_str}: MPJPE = {mpjpe_mean:.2f} mm")
            
        except Exception as e:
            print(f"Error processing {retrieved_pose_path}: {e}")
            mpjpe_results['ranks'][rank_str] = {
                'pose_path': retrieved_pose_path,
                'error': str(e)
            }
    
    # Save results to JSON file for this specific combination
    combo_dir = os.path.join(output_base_path, f'sample_{index:05d}', combination)
    os.makedirs(combo_dir, exist_ok=True)
    
    # Detailed results
    output_file = os.path.join(combo_dir, 'mpjpe_results.json')
    with open(output_file, 'w') as f:
        json.dump(mpjpe_results, f, indent=2)
    
    # Summary results (just MPJPE values)
    summary_results = {}
    for rank_str, rank_data in mpjpe_results.get('ranks', {}).items():
        if 'mpjpe_mm' in rank_data:
            summary_results[rank_str] = rank_data['mpjpe_mm']
    
    summary_file = os.path.join(combo_dir, 'mpjpe_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"MPJPE results saved to: {output_file}")
    print(f"MPJPE summary saved to: {summary_file}")





if __name__ == "__main__":
	full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE)
	combinations = [
		"image_to_pose",
		# "image_to_pressure",
		# "pose_to_image",
		# "pose_to_pressure",
		# "pressure_to_image",
		"pressure_to_pose",
		"image+pressure_to_pose",
		# "pose+pressure_to_image",
		# "image+pose_to_pressure",
	]
	
	IMAGE_PATH_OG = '/scratch/avs7793/work_done/poseembroider/new_model/image_path.json'
	SMPLX_MODEL_PATH = '/scratch/avs7793/models/smplx'
	OUTPUT_BASE_PATH = '/scratch/avs7793/work_done/poseembroider/new_model/target_results'
	model = smplx.SMPLX(
        model_path=SMPLX_MODEL_PATH,
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
    )
	
	
	with open(IMAGE_PATH_OG, 'r') as f:
		x = json.load(f)

	x = list(x.values())

	indices = [140,5159,5463,5601,5800,6305,6667,726,7187,7390,9866,10103,10104,1167,10473,11147,11501,2445]
	count = 0
	loss = nn.MSELoss(reduction='mean')

	for index in indices:

		for combination in combinations:
			calculate_and_save_mpjpe(index, combination, SMPLX_MODEL_PATH, OUTPUT_BASE_PATH)
			# find_path(original_path=None, dataset=full_dataset, combination=combination, folder=f'sample_{index:05d}')
			# find_foot_pressure_mse(dataset=full_dataset, combination=combination, folder=f'sample_{index:05d}', loss=loss, og_foot_p=x[count])
			# for rank in range(1,4):
			# 	output_prefix = os.path.join(OUTPUT_BASE_PATH, f'sample_{index:05d}', combination, f'rank_{rank:02d}')
			# 	pt_file = os.path.join(output_prefix, 'pose.pt')
			# 	visualize_smplx_pose(pt_file, SMPLX_MODEL_PATH, output_prefix, model)
			# 	print(pt_file)
				# break
			# break
		count += 1
		# break



	