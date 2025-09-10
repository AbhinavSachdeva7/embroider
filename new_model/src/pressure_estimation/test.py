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

def test(x):
    # --- Configuration ---
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    torch.manual_seed(config.SEED + x)
    torch.cuda.manual_seed(config.SEED + x)
    BATCH_SIZE = 512
    NUM_WORKERS = 8
    VALIDATION_SPLIT = 0.2
    loss_fn = nn.MSELoss(reduction="mean")

    print("Loading and splitting the dataset...")
    full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE, transform = None)
    checkkpoint = torch.load("/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_75_pressure_estimation_new.pth", map_location=torch.device('cpu'), weights_only=False)

    
    # Create the split sizes
    test_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - test_size

    # Split the dataset
    _, test_dataset = data.random_split(full_dataset, [train_size, test_size])

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Initialize model
    model = PressureEstimatorNew(latentD=config.LATENT_D).to(device)
    model.load_state_dict(checkkpoint['model_state_dict'])

    model.eval()
    total_squared_error = 0
    total_elements = 0
    all_errors = []  # Store all individual errors for standard deviation calculation
    
    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Testing")
        for i, batch in enumerate(test_progress_bar):
            poses = batch['pose'].to(device) # Shape: (512, 66)
            images = batch['image'].to(device) # Shape: (512, 3, 256, 192)
            pressures = batch['pressure_map'].to(device) # Shape: (512, 2520)
            # print(images.shape)
            # print(poses.shape)
            # print(pressures.shape)
            outputs = model(pose=poses)
            outputs = torch.expm1(outputs)

            # Calculate individual errors (absolute differences)
            individual_errors = torch.abs(outputs - pressures)  # Element-wise absolute errors
            all_errors.append(individual_errors.cpu())  # Move to CPU and store
            
            # This calculates the average MSE for the current batch
            loss = loss_fn(outputs, pressures)
            loss = torch.sqrt(loss)

            # `pressures.numel()` is 512 * 1 * 42 * 60 = 1,290,240
            # We multiply the batch's average error by its number of elements
            # to get the sum of all squared errors for this batch.
            total_squared_error += loss.item() * pressures.numel()

            # We keep a running total of all elements processed
            total_elements += pressures.numel()
            test_progress_bar.set_postfix(batch_mse=loss.item())

    # Calculate RMSE
    average_mse = total_squared_error / total_elements if total_elements > 0 else 0.0
    
    # Calculate standard deviation of errors
    all_errors = torch.cat(all_errors, dim=0)  # Concatenate all batch errors
    error_std = torch.std(all_errors).item()   # Calculate standard deviation
    error_mean = torch.mean(all_errors).item() # Mean absolute error
    
    return {
        'rmse': average_mse,
        'error_std': error_std,
        'mae': error_mean  # Mean Absolute Error as bonus
    }

if __name__ == "__main__":
    train_results = test(0)
    test_results = test(232)
    
    print(f"Train RMSE: {train_results['rmse']:.6f}")
    print(f"Train Error Std: {train_results['error_std']:.6f}")
    print(f"Train MAE: {train_results['mae']:.6f}")
    
    print(f"Test RMSE: {test_results['rmse']:.6f}")
    print(f"Test Error Std: {test_results['error_std']:.6f}")
    print(f"Test MAE: {test_results['mae']:.6f}")
