import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import new_model.src.config as config
from new_model.src.full_model import PoseImagePressureEmbroider
from new_model.src.data_loader.dataset import ImagePosePressureDataset
import os

def validate_model(checkpoint_path: str, data_loader: DataLoader):
    """
    Loads a trained model checkpoint and evaluates its loss on the test set.

    Args:
        checkpoint_path (str): The path to the saved model .pth file.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # --- Initialize Model and Load Trained Weights ---
    model = PoseImagePressureEmbroider(latentD=config.LATENT_D).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded model from {checkpoint_path}")

    # --- Load and Split Data ---
    # This must use the same logic and seed as the training script
    test_loader = data_loader

    # --- Evaluation Loop ---
    model.eval()  # Set the model to evaluation mode (important!)
    total_val_loss = 0
    
    # Wrap the entire loop in torch.no_grad() for efficiency
    with torch.no_grad():
        test_progress_bar = tqdm(test_loader, desc="Validating")
        for batch in test_progress_bar:
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            pressure_maps = batch['pressure_map'].to(device)
            
            # During validation, we typically use the full triplet for the most complete loss signal
            loss_dict = model(images=images, poses=poses, pressure_maps=pressure_maps, 
                              single_partials=True, dual_partials=True, triplet_partial=False)
            
            if not loss_dict:
                continue

            loss = sum(loss_dict.values()) / len(loss_dict)
            total_val_loss += loss.item()
            test_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_test_loss = total_val_loss / len(test_loader)
    print(f"\n--- Validation Complete ---")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    
    return avg_test_loss


if __name__ == '__main__':
    # --- Configuration ---
    # Set the path to the model checkpoint you want to evaluate
    LOG_FILE = os.path.join('/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints', "testing_log.csv")

    full_dataset = ImagePosePressureDataset(metadata_file=config.DATA_FILE, transform = None)
    test_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - test_size
    generator = torch.Generator().manual_seed(config.SEED)
    _, test_dataset = data.random_split(full_dataset, [train_size, test_size], generator=generator)
    
    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        f.write("Epoch,Validation Loss\n")
    print(f"Logging testing progress to {LOG_FILE}")
    
    for i in range(5,31,5):
        model_checkpoint = f"/scratch/avs7793/work_done/poseembroider/new_model/src/checkpoints/model_epoch_{i}.pth"
        test_loss = validate_model(model_checkpoint, test_loader)
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"{i},{test_loss}\n")


    