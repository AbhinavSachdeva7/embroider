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

IMAGE_SIZE = (192, 256)

def get_image_tensor(image_path):

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
	# pose_tensor = torch.tensor(pose)
	pose_tensor = pose[:66]
	return pose_tensor # shape = [66]

def visualize(image_path = None, pose = None, model_checkpoint_path= None):
	device = torch.device("cuda:0")
	print(f"Using device: {device}")
	
	checkpoint_model = torch.load(model_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
	model = PressureEstimatorNew(latentD=config.LATENT_D).to(device)
	model.load_state_dict(checkpoint_model['model_state_dict'])

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
		
		outputs = model(image=input_image, pose=input_pose)
		outputs = torch.expm1(outputs)
		outputs = outputs.squeeze(0)
		return outputs
		
def save_pressure_heatmap(image_path, pressure_tensor, original_pressure, filepath, title="Pressure Map", pressure_shape=(60, 42), subject_id=7, take_id=3, index=139):
        """Save pressure map as heatmap, displaying generated and original side-by-side"""
        if pressure_tensor.is_cuda:
            pressure_tensor = pressure_tensor.cpu()
        if original_pressure.is_cuda:
            original_pressure = original_pressure.cpu()
		
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(pressure_tensor, original_pressure)

        pressure_map = pressure_tensor.numpy().reshape(pressure_shape)
        original_pressure_map = original_pressure.numpy().reshape(pressure_shape)
        
        original_image = Image.open(image_path)

        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 2)
        fig.suptitle(f"Subject-{subject_id} Take-{take_id} Index-{index:05d} \nMSE: {loss.item():.4f} kPa")

        # Column 1: Original Image and Pressure
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(original_image)
        ax_img.set_title("Original Image")
        ax_img.axis('off')

        ax_orig = fig.add_subplot(gs[1, 0])
        im1 = ax_orig.imshow(original_pressure_map, cmap='viridis', interpolation='nearest')
        fig.colorbar(im1, ax=ax_orig)
        ax_orig.set_title(f"Original {title}")

        # Column 2: Generated Pressure
        ax_gen = fig.add_subplot(gs[:, 1])
        im2 = ax_gen.imshow(pressure_map, cmap='viridis', interpolation='nearest')
        fig.colorbar(im2, ax=ax_gen)
        ax_gen.set_title(f"Generated {title}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)		


if __name__ == "__main__":

    subject_id = 7
    take_id = 3
    indices = [139,4174,4536,4865,5160,5464,5600,5800,6020,6306,6668,727,7013,7186,7391,7652,7986,8610,9129,9507,9867,10104,1166,10472,11147,11500,12034,12671,13306,13734,14153,14470,14749,1629,15391,15723,16310,16836,2095,2444,2963,3392,3810] #keypose indexes of subject 7 take 3
    
    for index in indices:
        index = index - 1 
        image_path = f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/images/subject_{subject_id}/take_{take_id:}/{index:05d}.png"
    
        pressure_path = f"/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/pressure/pressure_subject{subject_id}_take{take_id}.pt"
        pressure = torch.load(pressure_path)
        original_pressure = pressure[index]

        pose_path = f'/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/poses/poses_subject{subject_id}_take{take_id}.pt'
        pose = torch.load(pose_path)
        pose = pose[index]

        model_checkpoint_path = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_75_pressure_estimation_new.pth"
        outputs = visualize(image_path = image_path, pose = None, model_checkpoint_path = model_checkpoint_path)
        save_file_name = f"pressure_estimation_image_only_subject{subject_id}_take{take_id}_index{index}.png"
        base_path = f"/scratch/avs7793/work_done/poseembroider/new_model/src/subject_7_pressure_estimation"
        save_path = os.path.join(base_path, save_file_name)


        save_pressure_heatmap(image_path, outputs, original_pressure, filepath=save_path, title="Pressure Map", pressure_shape=(60, 42), subject_id=subject_id, take_id=take_id, index=index)
		