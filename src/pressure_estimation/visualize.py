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
		
def save_pressure_heatmap(pressure_tensor, original_pressure, filepath, title="Pressure Map", pressure_shape=(60, 42)):
        """Save pressure map as heatmap"""
        if pressure_tensor.is_cuda:
            pressure_tensor = pressure_tensor.cpu()
		
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(pressure_tensor, original_pressure)

        pressure_map = pressure_tensor.numpy().reshape(pressure_shape)
        fig, ax = plt.subplots(figsize=(6, 8))
        im = ax.imshow(pressure_map, cmap='viridis', interpolation='nearest')
        fig.colorbar(im)
        ax.set_title(title)
        ax.set_xlabel(f"MSE: {loss.item():.4f}")
        
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)		


if __name__ == "__main__":
	image_path = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/images/subject_3/take_3/08593.png"
	
	pressure_path = "/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/pressure/pressure_subject3_take3.pt"
	pressure = torch.load(pressure_path)
	original_pressure = pressure[8592]
	
	pose_path = '/scratch/avs7793/work_done/poseembroider/new_model/src/data/processed/poses/poses_subject3_take3.pt'
	pose = torch.load(pose_path)
	pose = pose[8592]
	
	model_checkpoint_path = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_75_pressure_estimation_new.pth"
	outputs = visualize(image_path = image_path, pose = None, model_checkpoint_path = model_checkpoint_path)

	save_file_name = "pressure_estimation_image_only.png"
	base_path = "/scratch/avs7793/work_done/poseembroider/new_model/src/"
	save_path = os.path.join(base_path, save_file_name)

	save_pressure_heatmap(outputs, original_pressure, filepath=save_path, title="Pressure Map", pressure_shape=(60, 42))
		