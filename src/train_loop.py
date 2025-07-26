import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from new_model.src.full_model import PoseImagePressureEmbroider
import new_model.src.config as config
from new_model.src.data_loader.dataset import ImagePosePressureDataset
from tqdm import tqdm
import os
import re

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer state from checkpoint.
    Handles both old format (just model weights) and new format (full checkpoint).
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        start_epoch: The epoch to resume training from
    """
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if this is the old format (just model state_dict) or new format (full checkpoint)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            # New format - full checkpoint with optimizer state and epoch info
            print("Loading from new checkpoint format (includes optimizer state and epoch)")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']  # Start from next epoch
            
            if 'loss' in checkpoint:
                print(f"Resuming from epoch {checkpoint['epoch']}, last loss: {checkpoint['loss']:.4f}")
            
            return start_epoch
            
        else:
            # Old format - just model state_dict
            print("Loading from old checkpoint format (model weights only)")
            model.load_state_dict(checkpoint)
            
            # Extract epoch number from filename for old format
            epoch_match = re.search(r'epoch_(\d+)', checkpoint_path)
            if epoch_match:
                start_epoch = int(epoch_match.group(1))
                print(f"Extracted epoch {start_epoch} from filename. Resuming from epoch {start_epoch + 1}")
                return start_epoch
            else:
                print("Could not extract epoch from filename. Starting from epoch 1")
                return 1
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0

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
    BATCH_SIZE = 2048
    NUM_WORKERS = 8
    VALIDATION_SPLIT = 0.2
    
    # Checkpoint configuration
    RESUME_TRAINING = True  # Set to True to resume from checkpoint, False to start fresh
    RESUME_CHECKPOINT_PATH = "/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_44.pth"  # Path to checkpoint to resume from
    
    # Logging configuration
    LOG_FILE = os.path.join(config.CHECKPOINT_DIR, "training_log_beyond_30.csv")

    # --- Load and Split Data ---
    print("Loading and splitting the dataset...")
    full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE, transform = None)
    
    # Create the split sizes
    test_size = int(VALIDATION_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - test_size
    
    # Use a generator for reproducible splits
    generator = torch.Generator().manual_seed(config.SEED)
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=generator)
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Initialize Model and Optimizer ---
    model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # --- Load Checkpoint (if resuming) ---
    start_epoch = 0
    if RESUME_TRAINING and RESUME_CHECKPOINT_PATH:
        start_epoch = load_checkpoint(model, optimizer, RESUME_CHECKPOINT_PATH)

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
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for batch in train_progress_bar:
            # Move data to the correct device
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            pressure_maps = batch['pressure_map'].to(device)

            # Zero the gradients from the last step
            optimizer.zero_grad()
            
            # --- FORWARD PASS ---
            # Call the model to get the dictionary of all contrastive losses
            loss_dict = model(images=images, poses=poses, pressure_maps=pressure_maps, 
                              single_partials=True, dual_partials=True, triplet_partial=False)
            
            if not loss_dict:
                continue
            
            # Aggregate the losses from all partials
            loss = sum(loss_dict.values()) / len(loss_dict)

            # --- BACKWARD PASS ---
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            # Update the progress bar with the latest loss
            train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"--- Epoch {epoch+1}/{EPOCHS} | Average Train Loss: {avg_train_loss:.4f} ---")       

        # --- Log Results ---
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f}\n")
        print(f"--- Epoch {epoch+1}/{EPOCHS} | Avg Train Loss: {avg_train_loss:.4f} ---")

        # --- Save Regular Checkpoints ---
        if (epoch + 1) % SAVE_EPOCH_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_path)

    # --- Save Final Checkpoint ---
    final_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'model_epoch_{EPOCHS}_final.pth')
    save_checkpoint(model, optimizer, EPOCHS-1, avg_train_loss, final_checkpoint_path)
    print("Training completed!")

if __name__ == "__main__":
    train()



