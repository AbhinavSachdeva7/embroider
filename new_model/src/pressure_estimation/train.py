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

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save model and optimizer state to checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def train():
    # --- Configuration ---
    LEARNING_RATE = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    LOG_STEP = 10
    SAVE_EPOCH_INTERVAL = 5
    EPOCHS = 100
    BATCH_SIZE = 512
    NUM_WORKERS = 8
    VALIDATION_SPLIT = 0.2
    loss_fn = nn.MSELoss(reduction="sum")

    LOG_FILE = os.path.join(config.CHECKPOINT_DIR, "training_log_pressure_estimation_new.csv")
    print("Loading and splitting the dataset...")
    full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE, transform = None)
    checkkpoint = torch.load("/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_50_pressure_estimation_new_final.pth", map_location=torch.device('cpu'))

    
    # Create the split sizes
    test_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - test_size

    # Split the dataset
    _, test_dataset = data.random_split(full_dataset, [train_size, test_size])

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Initialize model
    model = PressureEstimatorNew(latentD=config.LATENT_D).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    model.load_state_dict(checkkpoint['model_state_dict'])
    
    # Load checkpoint
    start_epoch = checkkpoint['epoch']
    
    # --- Setup Logging ---
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # If starting fresh, create new log file. If resuming, append to existing log.
    if start_epoch == 0:
        with open(LOG_FILE, "w") as f:
            f.write("Epoch,Train Loss\n")
        print(f"Starting fresh training. Logging to {LOG_FILE}")
    else:
        print(f"Resuming training from epoch {start_epoch + 1}. Appending to {LOG_FILE}")

    # --- Training Loop ---
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0
        
        # --- TRAINING LOOP with TQDM ---
        train_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for batch in train_progress_bar:
            # Move data to the correct device
            images = batch['image'].to(device)
            
            pressure_maps = batch['pressure_map'].to(device)

            # Zero the gradients from the last step
            optimizer.zero_grad()

            z_decoded = model(images)
            z_reconstructed = torch.expm1(z_decoded)

            loss = loss_fn(z_reconstructed, pressure_maps)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(test_loader)

        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f}\n")
        print(f"--- Epoch {epoch+1}/{EPOCHS} | Avg Train Loss: {avg_train_loss:.4f} ---")

        # --- Save Regular Checkpoints ---
        if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}_pressure_estimation_new.pth')
            save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_path)

    # --- Save Final Checkpoint ---
    final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'model_epoch_{EPOCHS}_pressure_estimation_new_final.pth')
    save_checkpoint(model, optimizer, EPOCHS-1, avg_train_loss, final_checkpoint_path)
    print("Training completed!")

if __name__ == "__main__":
    train()