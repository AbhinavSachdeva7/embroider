import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from new_model.src.pressure_estimation.model import PressureEstimatorNew
import new_model.src.config as config
from new_model.src.data_loader.dataset import ImagePosePressureDataset
from tqdm import tqdm
import os
import re
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import smplx
import trimesh
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
This script provides functions to visualize pressure estimation model outputs.
It can load a model, predict pressure maps from images and/or poses,
and save comparative visualizations of predicted versus original pressure maps,
including 3D SMPLX pose renderings.
"""


IMAGE_SIZE = (192, 256)
SMPLX_MODEL_PATH = '/scratch/avs7793/models/smplx'

def get_image_tensor(image_path):
    """
    Load an image from the given path, transform it to a tensor, and normalize it.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The processed image tensor of shape [3, 256, 192].
    """

    transform = transforms.Compose([
                # Images are already resized, so we only need to convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    image = Image.open(image_path)
	# uncomment the following lines to convert the image to RGB and resize it

	# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# image = cv2.resize(image, IMAGE_SIZE)
	# image = Image.fromarray(image)
	
    image_tensor = transform(image)
    return image_tensor # shape = [3,256,192]

def get_pose_tensor(pose):
    """
    Extracts the first 66 elements from a pose tensor.

    Args:
        pose (torch.Tensor): The input pose tensor.

    Returns:
        torch.Tensor: A tensor containing the first 66 elements of the input pose, shape [66].
    """
	# pose_tensor = torch.tensor(pose)
    pose_tensor = pose[:66]
    return pose_tensor # shape = [66]

def visualize(image_path = None, pose = None, model= None):
    """
    Performs inference using the pressure estimation model on an image and/or pose.

    This function loads a pre-trained model, processes the input image and/or pose,
    and returns the predicted pressure map.

    Args:
        image_path (str, optional): Path to the input image. Defaults to None.
        pose (torch.Tensor, optional): Input pose tensor. Defaults to None.
        model_checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        torch.Tensor: The predicted pressure map from the model. Returns None if both image_path and pose are None.
    """
    
    device = torch.device("cuda:0")
    model = model
    model.eval()
	
    with torch.no_grad():
        if image_path is not None:
            image_tensor = get_image_tensor(image_path)
            image_tensor = image_tensor.unsqueeze(0).to(device) # shape = [1,3,256,192]		
        if pose is not None:
            pose_tensor = get_pose_tensor(pose)
            pose_tensor = pose_tensor.unsqueeze(0).to(device) # shape = [1,66]
		
        if image_path is None and pose is None:
            print("Both image and pose are not provided")
            return
        input_image = image_tensor if image_path is not None else None
        input_pose = pose_tensor if pose is not None else None
		
        outputs = model(pose=input_pose, image=input_image)
        outputs = torch.expm1(outputs)
        outputs = outputs.squeeze(0)
        return outputs
		
def save_pressure_heatmap(image_path, pose_tensor, pressure_tensor, original_pressure, loss_fn, filepath, title="Pressure Map", pressure_shape=(60, 42), subject_id=7, take_id=3, index=139):
        """
        Saves a visualization comparing generated and original pressure maps.

        The visualization includes the original image, the original pressure heatmap,
        the generated pressure heatmap, and an optional 3D pose visualization.

        Args:
            image_path (str): Path to the original image.
            pose_tensor (torch.Tensor): The pose tensor used for 3D visualization.
            pressure_tensor (torch.Tensor): The generated pressure tensor.
            original_pressure (torch.Tensor): The ground truth pressure tensor.
            filepath (str): The path to save the output image.
            title (str, optional): Title for the pressure maps. Defaults to "Pressure Map".
            pressure_shape (tuple, optional): The shape to reshape the pressure tensors into. Defaults to (60, 42).
            subject_id (int, optional): Subject ID for display. Defaults to 7.
            take_id (int, optional): Take ID for display. Defaults to 3.
            index (int, optional): Frame index for display. Defaults to 139.
        """
        if pressure_tensor.is_cuda:
            pressure_tensor = pressure_tensor.cpu()
        if original_pressure.is_cuda:
            original_pressure = original_pressure.cpu()

        has_image = None
        has_pose = None
        if image_path is not None:
            has_image = image_path 
        if pose_tensor is not None:
            has_pose = pose_tensor 

        if has_pose is not None:
            vertices, faces, views = visualize_smplx_pose(pose=pose_tensor, model_path=SMPLX_MODEL_PATH)
		
        loss_fn = loss_fn
        total_loss = loss_fn(pressure_tensor, original_pressure)
        total_rmse_loss = torch.sqrt(total_loss)

        non_zero_mask = original_pressure > 0
        contact_rmse_str = "N/A"
        if torch.any(non_zero_mask):
            contact_loss = loss_fn(pressure_tensor[non_zero_mask], original_pressure[non_zero_mask])
            contact_rmse_loss = torch.sqrt(contact_loss)
            contact_rmse_str = f"{contact_rmse_loss.item():.4f} kPa"


        pressure_map = pressure_tensor.numpy().reshape(pressure_shape)
        original_pressure_map = original_pressure.numpy().reshape(pressure_shape)
        difference_pressure_map = pressure_map - original_pressure_map
        
        if has_image is not None and has_pose is not None:
            fig = plt.figure(figsize=(12, 18))
            gs = fig.add_gridspec(3, 2)
        else:
            fig = plt.figure(figsize=(12, 12))
            gs = fig.add_gridspec(2, 2)

        fig.suptitle(f"Subject-{subject_id} Take-{take_id} Index-{index:05d} \nOverall RMSE: {total_rmse_loss.item():.4f} kPa \nContact Area RMSE: {contact_rmse_str}")

        # Column 2: Generated Pressure and Differnece map(common to all cases)
        gs_col2 = gs[:, 1].subgridspec(2, 1, hspace=0.3)

        ax_gen = fig.add_subplot(gs_col2[0])
        im2 = ax_gen.imshow(pressure_map, cmap='viridis', interpolation='nearest')
        fig.colorbar(im2, ax=ax_gen)
        ax_gen.set_title(f"Generated {title}")

        ax_diff = fig.add_subplot(gs_col2[1])
        vmax = np.abs(difference_pressure_map).max()
        vmin = -vmax
        im3 = ax_diff.imshow(difference_pressure_map, cmap='coolwarm', interpolation='nearest', vmin=vmin, vmax=vmax)
        fig.colorbar(im3, ax=ax_diff)
        ax_diff.set_title("Difference Map (Generated - Original)")


        if has_image is not None and has_pose is not None:
            # Case: Image and Pose
            ax_img = fig.add_subplot(gs[0, 0])
            original_image = Image.open(image_path)
            ax_img.imshow(original_image)
            ax_img.set_title("Original Image")
            ax_img.axis('off')

            ax_pose = fig.add_subplot(gs[1, 0], projection='3d')
            ax_pose.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=[0.8, 0.8, 1.0], alpha=0.9)
            ax_pose.set_title("SMPL Pose")
            ax_pose.view_init(elev=0, azim=90)
            max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
            mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
            ax_pose.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_pose.set_ylim(mid_y - max_range, mid_y + max_range)
            ax_pose.set_zlim(mid_z - max_range, mid_z + max_range)

            ax_orig = fig.add_subplot(gs[2, 0])
            im1 = ax_orig.imshow(original_pressure_map, cmap='viridis', interpolation='nearest')
            fig.colorbar(im1, ax=ax_orig)
            ax_orig.set_title(f"Original {title}")

        elif has_image is not None:
            # Case: Image only
            ax_img = fig.add_subplot(gs[0, 0])
            original_image = Image.open(image_path)
            ax_img.imshow(original_image)
            ax_img.set_title("Original Image")
            ax_img.axis('off')

            ax_orig = fig.add_subplot(gs[1, 0])
            im1 = ax_orig.imshow(original_pressure_map, cmap='viridis', interpolation='nearest')
            fig.colorbar(im1, ax=ax_orig)
            ax_orig.set_title(f"Original {title}")
        
        elif has_pose is not None:
            # Case: Pose only
            ax_pose = fig.add_subplot(gs[0, 0], projection='3d')
            ax_pose.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=[0.8, 0.8, 1.0], alpha=0.9)
            ax_pose.set_title("SMPL Pose")
            ax_pose.view_init(elev=0, azim=90)
            max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                             vertices[:, 1].max()-vertices[:, 1].min(),
                             vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
            mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
            ax_pose.set_xlim(mid_x - max_range, mid_x + max_range)
            ax_pose.set_ylim(mid_y - max_range, mid_y + max_range)
            ax_pose.set_zlim(mid_z - max_range, mid_z + max_range)

            ax_orig = fig.add_subplot(gs[1, 0])
            im1 = ax_orig.imshow(original_pressure_map, cmap='viridis', interpolation='nearest')
            fig.colorbar(im1, ax=ax_orig)
            ax_orig.set_title(f"Original {title}")
            print("Saving")


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)		


def visualize_smplx_pose(pose, model_path):
    """
    Generates a 3D mesh from a given pose using the SMPLX model.

    This function loads an SMPLX model, processes a pose tensor, and returns the
    vertices and faces of the resulting 3D mesh. The visualization part is currently
    commented out but can be enabled to save images of the 3D model.

    Args:
        pose (torch.Tensor): A tensor representing the pose parameters.
        model_path (str): Path to the directory containing the SMPLX model files.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The vertices of the generated mesh.
            - np.ndarray: The faces of the generated mesh.
            - list: A list of view parameters for visualization.
    """
    
    # Load pose parameters from .pt file
    single_pose = pose
    
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
    
    if pose_tensor.shape[1] == 66:
        # First 3 parameters are global orientation
        global_orient = pose_tensor[:, :3]
        # Remaining 63 parameters are body pose (21 joints × 3 dimensions)
        body_pose_params = pose_tensor[:, 3:66]  # 63 parameters
        body_pose = body_pose_params.reshape(batch_size, 21, 3)
        
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

    # for azim, elev, title in views:
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Plot the surface model
    #     ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color=[0.8, 0.8, 1.0], alpha=0.9)
        
    #     ax.set_title(f'{title}')
    #     ax.view_init(elev=elev, azim=azim)
        
    #     # Set equal aspect ratio
    #     max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
    #                          vertices[:, 1].max()-vertices[:, 1].min(),
    #                          vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    #     mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    #     mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    #     mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
        
    #     ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #     ax.set_ylim(mid_y - max_range, mid_y + max_range)
    #     ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    #     # Remove axis labels for cleaner look
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
    #     ax.set_zlabel('')
    
    #     # Construct output path and save image
    #     view_name = title.lower().replace(' ', '_')
    #     output_image_path = f"{output_path_prefix}_{view_name}.png"
        
    #     plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    #     plt.close(fig)
        
    #     print(f'SMPLX pose visualization saved as: {output_image_path}')

    # print(f'Vertices shape: {vertices.shape}')
    
    return vertices,faces,views


if __name__ == "__main__":

    # --- Configuration ---
    # Define the subject, take, and specific frame indices for visualization.
    subject_id = 7
    take_id = 3
    indices = [139,4174,4536,4865,5160,5464,5600,5800,6020,6306,6668,727,7013,7186,7391,7652,7986,8610,9129,9507,9867,10104,1166,10472,11147,11500,12034,12671,13306,13734,14153,14470,14749,1629,15391,15723,16310,16836,2095,2444,2963,3392,3810] #keypose indexes of subject 7 take 3
    
    MODEL_CHECKPOINT_PATH = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_75_pressure_estimation_new.pth"
    BASE_PATH = f"/scratch/avs7793/work_done/poseembroider/new_model/inference_results/pressure_estimation_results_image"
    USE_IMAGE = True
    USE_POSE = False
    DEVICE = torch.device("cuda:0")
    LOSS_FN = nn.MSELoss(reduction='mean')
    
    checkpoint_model = torch.load(MODEL_CHECKPOINT_PATH, map_location=torch.device('cpu'), weights_only=False)
    model = PressureEstimatorNew(latentD=config.LATENT_D).to(DEVICE)
    model.load_state_dict(checkpoint_model['model_state_dict'])

    print(f"Using device: {DEVICE}")


    # --- Main Loop ---
    # Iterate through each specified index to generate and save pressure map visualizations.
    for index in indices:
        index = index - 1 
        
        # --- Data Loading ---
        # Construct file paths and load the image, original pressure, and pose data.
        image_path = f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/images/subject_{subject_id}/take_{take_id:}/{index:05d}.png"
    
        pressure_path = f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/pressure/pressure_subject{subject_id}_take{take_id}.pt"
        pressure = torch.load(pressure_path)
        original_pressure = pressure[index]

        pose_path = f'/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/poses/poses_subject{subject_id}_take{take_id}.pt'
        pose = torch.load(pose_path)
        pose_tensor = pose[index]

                
        # --- Save Visualization ---
        # Define output path and save the comparison heatmap.
        save_file_name = f"pressure_estimation_image_only_subject{subject_id}_take{take_id}_index{index}.png"    
        save_path = os.path.join(BASE_PATH, save_file_name)



        # --- Inference ---
        # Load the model checkpoint and run inference to get the predicted pressure map.
        
        if USE_IMAGE and USE_POSE:
            outputs = visualize(image_path=image_path, pose = pose_tensor, model = model)
            save_pressure_heatmap(image_path=image_path, pose_tensor=pose_tensor, pressure_tensor=outputs, original_pressure=original_pressure, loss_fn=LOSS_FN, filepath=save_path, title="Pressure Map", pressure_shape=(60, 42), subject_id=subject_id, take_id=take_id, index=index)

        elif USE_IMAGE:
            outputs = visualize(image_path=image_path, pose = None, model = model)
            save_pressure_heatmap(image_path=image_path, pose_tensor=None, pressure_tensor=outputs, original_pressure=original_pressure, loss_fn=LOSS_FN, filepath=save_path, title="Pressure Map", pressure_shape=(60, 42), subject_id=subject_id, take_id=take_id, index=index)

        elif USE_POSE:
            outputs = visualize(image_path=None, pose = pose_tensor, model = model)
            save_pressure_heatmap(image_path=None, pose_tensor=pose_tensor, pressure_tensor=outputs, original_pressure=original_pressure, loss_fn=LOSS_FN, filepath=save_path, title="Pressure Map", pressure_shape=(60, 42), subject_id=subject_id, take_id=take_id, index=index)

        else:
            print("Either Image or Pose is required for this to work")
            


        
        
        
        # break